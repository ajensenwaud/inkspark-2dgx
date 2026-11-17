---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Small
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: exllamav3
tags:
- exl3
- inkling2
- tensorfold
- dgx-spark
- gb10
---

# Inkling-2-Small, EXL3 at ~2.2 bpw

[thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (594B mixture of experts)
quantised to EXL3 from its NVFP4 release, made to run across two NVIDIA DGX Sparks (GB10) with tensor parallelism at
the model's full 1M-token context, with its vision tower and its DSpark drafter. The project that serves it:
[ajensenwaud/inkspark-2dgx](https://github.com/ajensenwaud/inkspark-2dgx).

| | |
|---|---|
| Weights | 170 GB, 25 shards (about 84 GiB a Spark at TP=2) |
| Bitrate | experts 2.17 bpw (gate and up 2 bits; down 3 bits on the first and last 12 layers, 2 bits between); attention 6 bpw; head 8 bpw; embedding bf16 |
| Conversion | exllamav3 1.6.6 with an Inkling 2 architecture overlay, calibration 250 x 2048, mul1 codebook; source @ `aac97412` |
| Drafter | the model's own DSpark drafter (`mtp.safetensors`), bf16 as released |
| Vision | the model's vision tower (`vision.safetensors`), bf16 as released |

## Quality

Served on two Sparks with InkSpark's engine, thinking on, temperature 1.0, top-p 0.95, top-k 20.

| Task | This pack | Thinking Machines' card |
|---|---|---|
| Global-MMLU-Lite (500 items, 23 languages) | 89.6 % | 91.0 % |
| GSM8K (500) | 97.4 % | not reported |
| MMLU-Pro (500) | 84.6 % | not reported |

Global-MMLU-Lite lands inside its ±2.7-point interval of the card's figure.

## Speed (two DGX Sparks, one stream unless stated)

| | 1M-token window | 8 x 128K windows |
|---|---|---|
| Decode at 128K context, plain / with the drafter | 25.9 / 37.9 tok/s | 23.3 / 36.9 tok/s (120K) |
| Decode at 1M context, plain / with the drafter | 13.3 / 21.1 tok/s | |
| Reading a cold 64K prompt | | 774 tok/s (84 s) |
| Eight streams at once | | 89-90 tok/s in total |

A cold 1M-token prompt takes about 46 minutes to read; follow-up turns resume from the prompt cache in seconds.

## Use

With InkSpark on two Sparks: `scripts/prepare.sh`, `scripts/fetch_pack.sh`, then `./start.sh`. The pack needs
InkSpark's TensorFold patches (the `inkling2` family); upstream TensorFold and exllamav3 cannot load it yet.

## Acknowledgements

Thinking Machines for the model, its vision tower and the DSpark drafter; turboderp for exllamav3, which made this
pack; Ash Hart for TensorFold, the engine it was evaluated on. Evaluation data: GSM8K (MIT), MMLU-Pro (MIT),
Global-MMLU-Lite (Apache-2.0).

## Licence

Apache-2.0, as the source model.
