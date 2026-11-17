from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from typing_extensions import override
from ..model.config import Config
from . import Module, Linear
from .quant import LinearFP16

# Inkling 2 (model_type inkling2) building blocks. Reference: transformers models/inkling2 (PR #50682).
# Only the cache-free forward (attn_mode flash_attn_nc) is implemented: this is the converter's and the
# reference engine's path. The residual stream and the attention-residual block stack are kept in bf16, as
# the reference keeps them; matmuls run in fp16 with fp32 outputs, elementwise work in fp32.
#
# Block attention residuals: each layer mixes the outputs of the completed 6-layer blocks (the input of every
# block-boundary layer, starting with the embedding) with the partial block sum. The stack travels in
# params["ink_blocks"] (a list of (bsz, seq, hidden) bf16 tensors): Model.forward shares one params dict across
# modules, and the converter carries it from module to module per calibration row through
# params["quant_preserve"].

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant_nvfp4(values: torch.Tensor, scales: torch.Tensor, amax: torch.Tensor, denominator: torch.Tensor):
    """(n, k/2) e2m1 pairs, low nibble first, (n, k/16) e4m3 block scales, per-matrix amax and the layer's
    outer_scale_denominator -> (n, k) fp32. W = e2m1 * block_scale * amax / denominator (vLLM/SGLang inkling2)"""
    lut = torch.tensor(E2M1, dtype = torch.float, device = values.device)
    v = values.view(torch.uint8)
    w = torch.stack((lut[(v & 0x0F).long()], lut[(v >> 4).long()]), dim = -1).flatten(-2)
    w *= scales.float().repeat_interleave(16, dim = -1)
    w *= amax.float() / denominator.float()
    return w


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor):
    """(n, k) e4m3 values with (n, k/32) ue8m0 scales -> (n, k) fp32"""
    return values.float() * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim = -1)


_expert_cache = {}

def load_expert_matrix(stc, fkey: str, e: int, device) -> torch.Tensor:
    """One expert's (out, in) fp32 matrix from a stacked checkpoint tensor (NVFP4, MXFP8 or plain)"""
    ck = (id(stc), fkey, e)
    if ck in _expert_cache:
        return _expert_cache[ck]
    get = lambda k: stc.get_tensor(k, device, no_defer = True, fidx = e, arena = False)
    w = get(fkey)
    if w.dtype == torch.uint8:
        den = stc.get_tensor(fkey + "_outer_scale_denominator", device, no_defer = True, arena = False)
        m = dequant_nvfp4(w, get(fkey + "_scale"), get(fkey + "_amax"), den.view(-1)[0])
    elif w.dtype == torch.float8_e4m3fn:
        m = dequant_mxfp8(w, get(fkey + "_scale"))
    else:
        m = w.float()
    _expert_cache.clear()
    _expert_cache[ck] = m
    return m


class Inkling2ExpertLinear(Linear):
    """Routed expert projection. Source: rows of the stacked w13 (gate/up interleaved: gate = even rows) or w2"""

    def __init__(self, config, key, fkey, fidx, part, in_features, out_features, **kwargs):
        super().__init__(config, key, in_features, out_features, **kwargs)
        self.src_fkey, self.src_fidx, self.src_part = fkey, fidx, part

    @override
    def load_fp16(self, key) -> bool:
        stc = self.config.stc
        if key != self.key or not stc.has_tensor(self.src_fkey):
            return False
        w = load_expert_matrix(stc, self.src_fkey, self.src_fidx, self.device)
        if self.src_part == "gate":
            w = w[0::2]
        elif self.src_part == "up":
            w = w[1::2]
        weight = self.pad_out(w.T.contiguous().half())
        self.inner = LinearFP16(
            self.in_features, self.out_features, weight, None, self.full_in_features, self.full_out_features,
            self.first_in_feature, self.first_out_feature, self.out_dtype, key = self.key
        )
        self.quant_type = "fp16"
        return True


class RawTensors(Module):
    """Module that owns unquantized tensors under fixed keys, stored as found in the checkpoint"""

    def __init__(self, config, key, qmap = None):
        super().__init__(config, key, qmap)
        self.raw_keys = []
        self.t = {}

    def add_raw(self, *keys):
        self.raw_keys += [k for k in keys if k is not None]

    def load_raw(self, device):
        stc = self.config.stc
        for k in self.raw_keys:
            self.t[k] = stc.get_tensor(k, device, allow_bf16 = True, no_defer = True, arena = False)

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        self.load_raw(device)

    @override
    def unload(self):
        super().unload()
        self.t = {}

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for m in self.modules:
            t.update(m.get_tensors())
        t.update({k: v.contiguous() for k, v in self.t.items()})
        return t

    @override
    def weights_numel(self):
        return super().weights_numel() + sum(v.numel() for v in self.t.values())

    @override
    def optimizer_targets(self):
        return []

    def f32(self, k):
        return self.t[k].float()

    def _outside_keys(self):
        """Raw tensors not under this module's key prefix (attention-residual queries live at model level)"""
        return [k for k in self.raw_keys if not k.startswith(self.key + ".")]

    @override
    def get_compile_sizes(self, stc):
        return super().get_compile_sizes(stc) + [stc.get_tensor_size(k) for k in self._outside_keys()]

    @override
    def get_compile_tensors(self, stc):
        t = super().get_compile_tensors(stc)
        t.update({k: stc.get_tensor(k, allow_bf16 = True) for k in self._outside_keys()})
        return t


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Inkling2RMSNorm (weight given) or the unweighted norm; fp32 math, fp32 result"""
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim = True) + eps)
    return x if weight is None else weight.float() * x


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    x = x.float()
    if scale is not None:
        x = x * scale.float()
    norm = torch.linalg.vector_norm(x, dim = -1, keepdim = True)
    return x * (threshold / (norm + 1e-6)).clamp(max = 1.0)


def static_conv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Inkling2ShortConvolution without cache: causal depthwise conv (w: (C, 1, K), tap K-1 = current token),
    plus the input, over sqrt(K + 1). fp32 in and out"""
    k = w.shape[-1]
    y = F.conv1d(x.transpose(1, 2), w.float().view(w.shape[0], 1, k), padding = k - 1, groups = w.shape[0])
    y = y[..., :x.shape[1]].transpose(1, 2)
    return (y + x) / math.sqrt(k + 1)


def dynamic_conv(x: torch.Tensor, dyn_in: torch.Tensor, static_w: torch.Tensor, weight_proj: torch.Tensor,
                 heads: int) -> torch.Tensor:
    """Inkling2DynamicShortConvolution without cache: taps = static (K, C) + per-token, per-head weights
    projected from dyn_in; tap 0 = current token. Plus the input, over sqrt(2K + 1). fp32 in and out"""
    b, s, c = x.shape
    k = static_w.shape[0]
    dyn = (dyn_in.float() @ weight_proj.float().T).view(b, s, heads, 1, k)
    w = static_w.float().T.reshape(heads, c // heads, k) + dyn
    ctx = F.pad(x.transpose(1, 2), (k - 1, 0))
    win = ctx.unfold(-1, k, 1).flip(-1).transpose(1, 2).reshape(b, s, heads, c // heads, k)
    y = (win * w).sum(-1).reshape(b, s, c)
    return (y + x) / math.sqrt(2 * k + 1)


def attn_res_mix(query: torch.Tensor, blocks: list, partial: torch.Tensor) -> torch.Tensor:
    """Inkling2BlockAttentionResidual: softmax over the RMS-normed stack [blocks..., partial] against learned
    queries (nq, hidden); returns (bsz, seq, nq, hidden) fp32, a mix of the normed states"""
    st = torch.stack([b.to(partial.device) for b in blocks] + [partial], dim = -2).float()
    st = rms_norm(st, None, 1e-6)
    q = query.float().view(-1, st.shape[-1])
    w = torch.softmax(torch.einsum("bsnh,qh->bsqn", st, q) * st.shape[-1] ** -0.5, dim = -1)
    return torch.einsum("bsqn,bsnh->bsqh", w, st)


def get_blocks(params: dict) -> list:
    return params.get("ink_blocks") or []


def put_blocks(params: dict, blocks: list):
    params["ink_blocks"] = blocks
    if "quant_preserve" in params:
        params["quant_preserve"]["ink_blocks"] = blocks


class Inkling2Embedding(RawTensors):
    """Token embedding followed by its RMSNorm (Inkling2NormedEmbedding); bf16 output"""

    def __init__(self, config, key, norm_key, eps):
        super().__init__(config, key)
        self.wkey, self.nkey, self.eps = key + ".weight", norm_key + ".weight", eps
        self.add_raw(self.wkey, self.nkey)

    @override
    def forward(self, x, params, out_dtype = None):
        e = F.embedding(x.to(self.device), self.t[self.wkey])
        put_blocks(params, [])
        return rms_norm(e, self.t[self.nkey], self.eps).to(torch.bfloat16)


class Inkling2Final(RawTensors):
    """Final attention-residual mix (output_query) and the final RMSNorm; fp16 output for the head"""

    def __init__(self, config, key, query_key, norm_key, eps):
        super().__init__(config, key)
        self.qkey, self.nkey, self.eps = query_key, norm_key + ".weight", eps
        self.add_raw(self.qkey, self.nkey)
        self.plain_residual = False     # planted control: skip the mix

    @override
    def forward(self, x, params, out_dtype = None):
        if self.plain_residual:
            h = x.float()
        else:
            h = attn_res_mix(self.t[self.qkey], get_blocks(params), x)[..., 0, :]
        return rms_norm(h, self.t[self.nkey], self.eps).to(torch.bfloat16).half()


class Inkling2Head(Linear):
    """Unembedding with the final logit softcap. Rows past unpadded_vocab_size are padding: consumers drop those
    logits, as the reference does (they stay finite here so the converter's error checks keep working)"""

    def __init__(self, config, key, hidden, vocab, softcap, **kwargs):
        super().__init__(config, key, hidden, vocab, qmap = "block", qbits_key = "head_bits",
                         caps = {"logits_output": True}, out_dtype = torch.float, trim_padded_out = True,
                         **kwargs)
        self.logit_softcap = softcap

    @override
    def forward(self, x, params, out_dtype = None):
        y = super().forward(x, params, out_dtype = torch.float)
        return torch.tanh(y / self.logit_softcap) * self.logit_softcap


class Inkling2Attention(RawTensors):

    def __init__(self, config, key, layer_idx, is_global, hidden, heads, kv_heads, head_dim, d_rel, rel_extent,
                 window, qk_clip, rel_clip, eps, log_floor, log_alpha):
        super().__init__(config, key)
        self.layer_idx, self.is_global = layer_idx, is_global
        self.heads, self.kv_heads, self.head_dim, self.d_rel = heads, kv_heads, head_dim, d_rel
        self.rel_extent, self.window = rel_extent, (None if is_global else window)
        self.qk_clip, self.rel_clip, self.eps = qk_clip, rel_clip, eps
        self.log_floor, self.log_alpha = log_floor, log_alpha
        qm = "block.attn"
        qq, qk, qv = (qm + ".q", qm + ".k", qm + ".v") if is_global else (qm, qm, qm)
        lin = lambda name, i, o, q: Linear(config, f"{key}.{name}", i, o, qmap = q, out_dtype = torch.float,
                                           select_hq_bits = 2)
        self.q_proj = lin("wq_du", hidden, heads * head_dim, qq)
        self.k_proj = lin("wk_dv", hidden, kv_heads * head_dim, qk)
        self.v_proj = lin("wv_dv", hidden, kv_heads * head_dim, qv)
        self.r_proj = lin("wr_du", hidden, heads * d_rel, qq)
        self.g_proj = lin("gate_attn", hidden, heads * head_dim, qq)
        self.o_proj = lin("wo_ud", heads * head_dim, hidden, qm + ".o")
        for m in (self.q_proj, self.k_proj, self.v_proj, self.r_proj, self.g_proj, self.o_proj):
            self.register_submodule(m)
        self.k_sink = f"{key}.attn_sink_proj.weight"
        self.k_qscale, self.k_rscale = f"{key}.q_l2_clip.scale", f"{key}.rel_l2_clip.scale"
        self.k_rel = f"{key}.rel_logits_proj.proj"
        self.add_raw(self.k_sink, self.k_qscale, self.k_rscale, self.k_rel)
        if is_global:
            self.k_conv = [f"{key}.{n}_sconv.{t}" for n in ("k", "v") for t in ("static_w", "weight_proj")]
        else:
            self.k_conv = [f"{key}.k_sconv.weight", f"{key}.v_sconv.weight"]
        self.add_raw(*self.k_conv)

    @override
    def optimizer_targets(self):
        t = lambda m: m.optimizer_targets()
        return [[t(self.q_proj) + t(self.r_proj) + t(self.g_proj), t(self.k_proj) + t(self.v_proj), t(self.o_proj)]]

    def core(self, q, k, v, rel, sinks, pos, chunk = 256):
        """Eager attention with the relative bias, per-token sinks and the causal / sliding mask, query-chunked.
        q (b, s, H, D), k/v (b, s, KV, D), rel (b, s, H, extent), sinks (b, s, H), all fp32"""
        b, s, H, D = q.shape
        g = H // self.kv_heads
        q, k, v = q.transpose(1, 2), k.repeat_interleave(g, 2).transpose(1, 2), v.repeat_interleave(g, 2).transpose(1, 2)
        rel = rel.transpose(1, 2)
        sinks = sinks.transpose(1, 2).unsqueeze(-1)
        out = torch.empty_like(q)
        for a in range(0, s, chunk):
            e = min(a + chunk, s)
            k0 = 0 if self.window is None else max(0, a - self.window + 1)
            dist = pos[a:e, None] - pos[None, k0:e]
            sc = q[:, :, a:e] @ k[:, :, k0:e].transpose(-1, -2) * D ** -0.5
            idx = dist.clamp(0, self.rel_extent - 1).expand(b, H, -1, -1)
            bias = rel[:, :, a:e].gather(-1, idx)
            sc += bias.masked_fill((dist < 0) | (dist >= self.rel_extent), 0.0)
            masked = dist < 0
            if self.window is not None:
                masked = masked | (dist >= self.window)
            sc = sc.masked_fill(masked, float("-inf"))
            p = torch.softmax(torch.cat((sc, sinks[:, :, a:e]), dim = -1), dim = -1)[..., :-1]
            out[:, :, a:e] = p @ v[:, :, k0:e]
        return out.transpose(1, 2)

    @override
    def forward(self, x, params, out_dtype = None):
        xq, xk, xv = x if self.is_global else (x, x, x)
        b, s, _ = xq.shape
        H, KV, D = self.heads, self.kv_heads, self.head_dim
        q = self.q_proj.forward(xq.half(), params)
        k = self.k_proj.forward(xk.half(), params)
        v = self.v_proj.forward(xv.half(), params)
        r = self.r_proj.forward(xq.half(), params)
        gate = self.g_proj.forward(xq.half(), params)
        sinks = xq.float() @ self.f32(self.k_sink).T
        if self.is_global:
            ksw, kwp, vsw, vwp = (self.t[n] for n in self.k_conv)
            k = dynamic_conv(k, xk, ksw, kwp, KV)
            v = dynamic_conv(v, xv, vsw, vwp, KV)
        else:
            k = static_conv(k, self.t[self.k_conv[0]])
            v = static_conv(v, self.t[self.k_conv[1]])
        pos = torch.arange(s, device = xq.device) + params.get("ink_pos0", 0)
        q = l2_clip(q.view(b, s, H, D), self.qk_clip, self.t[self.k_qscale])
        k = l2_clip(k.view(b, s, KV, D), self.qk_clip)
        v = rms_norm(v.view(b, s, KV, D), None, self.eps)
        r = l2_clip(r.view(b, s, H, self.d_rel), self.rel_clip, self.t[self.k_rscale])
        if self.is_global:
            tau = 1.0 + self.log_alpha * torch.log(((pos + 1).float() / self.log_floor).clamp(min = 1.0))
            q = q * tau[None, :, None, None]
            r = r * tau[None, :, None, None]
        rel = r @ self.f32(self.k_rel)
        o = self.core(q, k, v, rel, sinks, pos)
        o = o.reshape(b, s, H * D) * torch.sigmoid(gate)
        return self.o_proj.forward(o.half(), params)


class Inkling2MoE(RawTensors):
    """384 routed experts, sigmoid router with selection bias, top-k renormalised, x route_scale, softcapped
    SwiGLU: gate_cap * tanh(g / gate_cap) * sigmoid(g) * up_cap * tanh(u / up_cap)"""

    def __init__(self, config, key, hidden, inter, num_experts, top_k, route_scale, gate_cap, up_cap):
        super().__init__(config, key)
        self.num_experts, self.top_k, self.route_scale = num_experts, top_k, route_scale
        self.gate_cap, self.up_cap, self.hidden = gate_cap, up_cap, hidden
        self.k_router, self.k_bias = f"{key}.gate.weight", f"{key}.gate.bias"
        self.add_raw(self.k_router, self.k_bias)
        self.gates, self.ups, self.downs = [], [], []
        qin, grp = "block.mlp.input", key + ".block_gud"
        for e in range(num_experts):
            ek = f"{key}.experts.{e}"
            w13, w2 = f"{key}.experts.w13_weight", f"{key}.experts.w2_weight"
            g = Inkling2ExpertLinear(config, f"{ek}.gate_proj", w13, e, "gate", hidden, inter, qmap = qin,
                                     out_dtype = torch.float, qgroup = grp)
            u = Inkling2ExpertLinear(config, f"{ek}.up_proj", w13, e, "up", hidden, inter, qmap = qin,
                                     out_dtype = torch.float, qgroup = grp)
            d = Inkling2ExpertLinear(config, f"{ek}.down_proj", w2, e, "down", inter, hidden,
                                     qmap = f"block.mlp.{e}.down", out_dtype = torch.float, qgroup = grp)
            for m in (g, u, d):
                self.register_submodule(m)
            self.gates.append(g); self.ups.append(u); self.downs.append(d)

    @override
    def optimizer_targets(self):
        gu, d = [], []
        for g, u, dn in zip(self.gates, self.ups, self.downs):
            gu += g.optimizer_targets() + u.optimizer_targets()
            d += dn.optimizer_targets()
        return [[gu, d]]

    def act(self, g, u):
        return self.gate_cap * torch.tanh(g / self.gate_cap) * torch.sigmoid(g) * self.up_cap * torch.tanh(u / self.up_cap)

    @override
    def forward(self, x, params, out_dtype = None):
        shape = x.shape
        y = x.reshape(-1, self.hidden)
        scores = torch.sigmoid(y.float() @ self.f32(self.k_router).T)
        yh = y.half()

        if params.get("activate_all_experts") and "capture" in params:
            # Calibration: every expert sees every row (gate/up share one Hessian, captured by the first
            # expert run; each down projection gets its own). Only the experts of this capture pass run
            pq = params.get("capture_qmaps")
            run = [e for e in range(self.num_experts) if pq is None or self.downs[e].qmap in pq] or [0]
            for e in run:
                a = self.act(self.gates[e].forward(yh, params), self.ups[e].forward(yh, params)).half()
                dn = self.downs[e]
                if pq is None or dn.qmap in pq:
                    dn.capture_H(a, params)
            return torch.zeros(shape, dtype = torch.bfloat16, device = x.device)

        sel = torch.topk(scores + self.f32(self.k_bias), self.top_k, dim = -1, sorted = False).indices
        w = scores.gather(1, sel)
        w = w / (w.sum(dim = -1, keepdim = True) + 1e-20) * self.route_scale
        out = torch.zeros((y.shape[0], self.hidden), dtype = torch.float, device = x.device)
        for e in torch.unique(sel).tolist():
            tok, slot = torch.where(sel == e)
            xe = yh[tok]
            a = self.act(self.gates[e].forward(xe, params), self.ups[e].forward(xe, params))
            d = self.downs[e].forward(a.half(), params)
            out.index_add_(0, tok, d * w[tok, slot, None])
        return out.to(torch.bfloat16).view(shape)


class Inkling2Block(RawTensors):
    """One decoder layer, attention-residual mixes included. State in and out: (bsz, seq, hidden) bf16"""

    def __init__(self, config, key, layer_idx, is_global, is_boundary, attn_query_key, mlp_query_key,
                 attn, moe, eps):
        super().__init__(config, key)
        self.layer_idx, self.is_global, self.is_boundary, self.eps = layer_idx, is_global, is_boundary, eps
        self.k_aq, self.k_mq = attn_query_key, mlp_query_key
        self.k_norms = [f"{key}.{n}.weight" for n in (("q_input_norm", "k_input_norm", "v_input_norm")
                                                      if is_global else ("attn_norm",))]
        self.k_mlp_norm = f"{key}.mlp_norm.weight"
        self.k_attn_conv, self.k_mlp_conv = f"{key}.attn_sconv.weight", f"{key}.mlp_sconv.weight"
        self.add_raw(self.k_aq, self.k_mq, *self.k_norms, self.k_mlp_norm, self.k_attn_conv, self.k_mlp_conv)
        self.attn, self.mlp = attn, moe
        self.register_submodule(attn)
        self.register_submodule(moe)
        self.plain_residual = False     # planted control: replace both mixes with the plain residual

    @override
    def optimizer_targets(self):
        return [self.attn.optimizer_targets(), self.mlp.optimizer_targets()]

    @override
    def forward(self, x, params, out_dtype = None):
        bf = torch.bfloat16
        blocks = get_blocks(params)
        x = x.to(bf)

        if self.k_aq is None or self.plain_residual:
            ins = [x] * (3 if self.is_global else 1)
        else:
            mix = attn_res_mix(self.t[self.k_aq], blocks, x)
            ins = list(mix.unbind(-2))
        normed = [rms_norm(h, self.t[k], self.eps).to(bf) for h, k in zip(ins, self.k_norms)]
        a = self.attn.forward(normed if self.is_global else normed[0], params)
        a = static_conv(a.to(bf).float(), self.t[self.k_attn_conv]).to(bf).float()

        if self.is_boundary:
            stored = x.cpu() if "quant_preserve" in params else x
            blocks = blocks + [stored]
        if self.plain_residual:
            residual = h = x.float() + a
        else:
            residual = a if self.is_boundary else x.float() + a
            h = attn_res_mix(self.t[self.k_mq], blocks, residual)[..., 0, :]
        m = self.mlp.forward(rms_norm(h, self.t[self.k_mlp_norm], self.eps).to(bf), params)
        m = static_conv(m.float(), self.t[self.k_mlp_conv]).to(bf).float()
        put_blocks(params, blocks)
        return (residual + m).to(bf)
