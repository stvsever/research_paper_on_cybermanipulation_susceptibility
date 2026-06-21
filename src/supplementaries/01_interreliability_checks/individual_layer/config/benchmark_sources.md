# Benchmark Sources

The analysis config stores model benchmark metadata used for the MMLU-Pro
specification plots. Values are taken from model-card, technical-report, or
benchmark-provider sources:

- DeepSeek V4 Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- GPT-OSS 120B MMLU-Pro: https://www.vals.ai/comparison?modelA=fireworks%2Fgpt-oss-120b
- GPT-OSS 120B model card: https://arxiv.org/html/2508.10925v1
- Qwen3 32B: https://arxiv.org/html/2505.09388v1
- Nemotron 3 Nano 30B-A3B: https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard
- Llama 3.3 70B Instruct: https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md

`mmlu` is original MMLU where directly reported. `mmlu_pro` is used as the
single plotted latent benchmark for every model so the 3D specification curve is
interpretable on one benchmark scale. Nemotron 3 Nano's public card reports
MMLU-Pro but not original MMLU. GPT-OSS uses the Vals AI MMLU-Pro value because
the OpenAI model card reports MMLU but not MMLU-Pro.
