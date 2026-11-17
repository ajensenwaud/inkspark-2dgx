# InkSpark: Inkling-2-Small on 2x DGX Spark (GB10) with TensorFold

Thinking Machines' Inkling-2-Small (594B parameters, MoE, 48 layers) served across two DGX Sparks with 
tensor parallelism, from our own EXL3 quantisation of
Inkling's NVFP4 release, on TensorFold 1.2.7, with the model's own DSpark drafter. 

Key settings: ~2.2 bpw model, 1M context window, drafter on, images on.

## Running it

What you need: 
- Two GB10 DGX Sparks joined by their CX7 ports
- Docker with the NVIDIA runtime on both
- SSH configured from the head to the worker.

Ports: 8888 (API), 29591 (between the Sparks). 

Set `WORKER` (user@address) and `MASTER` (the head's CX7 address) in
`scripts/local.sh`.

Commands:
```bash
scripts/prepare.sh        # the image, then TensorFold v1.2.7 + patches-zig/ compiled in it, on both Sparks
scripts/fetch_pack.sh     # the model from Hugging Face (pinned, about 170 GB), on both Sparks, verified
./start.sh                # OpenAI API on :8888, model id Inkling-2-Small
./stop.sh
```

The model: [ajensenwaud/inkspark-2dgx](https://huggingface.co/ajensenwaud/inkspark-2dgx).

Settings in `scripts/config.sh` (override in `scripts/local.sh`): `SHAPE=long` (one conversation at 1M) or
`SHAPE=agents` (eight at 128K each), `DRAFTER=1|0`, `KV=fp8|bf16`.

## Model

| | |
|---|---|
| Size | 170 GB, about 84 GiB on each Spark |
| Experts | 2.17 bpw: gate and up 2 bits; down 3 bits on the first and last 12 layers, 2 bits between |
| Attention, head | 6 bpw, 8 bpw |
| Drafter, vision | bf16 as released by Thinking Machines |

## Quality

Settings: Two Sparks, thinking on, at Thinking Machines' sampling (temperature 1.0, top-p 0.95, top-k 20):

| Task | InkSpark | Thinking Machines' card |
|---|---|---|
| Global-MMLU-Lite (500 questions, 23 languages) | 89.6 % | 91.0 % |
| GSM8K (500) | 97.4 % | not reported |
| MMLU-Pro (500) | 84.6 % | not reported |

## Speed

One conversation at the 1M window, greedy, 512-token replies:

| Context | Plain decode | With the drafter | Reading the prompt |
|---|---|---|---|
| 128K | 25.9 tok/s | 37.9 tok/s | 734 tok/s |
| 256K | 22.6 | 32.7 | 540 |
| 512K | 18.1 | 29.2 | 429 |
| 1M | 13.3 | 21.1 | 295 |

Eight conversations at 128K each (`SHAPE=agents`):

| | |
|---|---|
| One stream, short prompt, code / prose (drafter on) | 41.4 / 32.4 tok/s |
| Eight streams at once | 90 tok/s in total |
| Reading a cold 64K prompt | 774 tok/s (84 s) |

Performance varies for cold long prompts.

## Credits and licence

Thanks to Thinking Machines for Inkling-2-Small, its vision tower and its DSpark drafter; Ash Hart
([@ashxhart](https://x.com/ashxhart)) for [TensorFold](https://github.com/ashhart/TensorFold); turboderp for
[exllamav3](https://github.com/turboderp-org/exllamav3), which made the pack; the transformers, vLLM and SGLang
contributors whose Inkling 2 code was the reference for the port; Mia AI Lab for the TensorFold CUDA base image.
Evaluation data: GSM8K (MIT), MMLU-Pro (MIT), Global-MMLU-Lite (Apache-2.0). Full list in `CREDITS.md`.

Recipe code: BSD 3-Clause. `patches-zig/` keep TensorFold's Apache-2.0. The model is Apache-2.0.
