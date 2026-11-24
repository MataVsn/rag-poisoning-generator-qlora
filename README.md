# QLoRA document generator for RAG poisoning evaluation

Course project (GenAI Endterm, VAIS Labs team). We fine-tune Qwen3 models with QLoRA to write **one controlled test document** from three inputs: an original fact, a target (replacement) fact and a question. The documents are then injected into an **isolated** RAG test collection to measure how often the target assistant adopts the replaced fact, with and without defenses.

This repository contains the **training side** only: dataset construction, QLoRA training, candidate generation, scoring, and the trained adapters. The RAG evaluation harness and the site backend live in a separate project.

> **Research use only.** The generator is meant for testing your own RAG systems in a sandbox. Do not use it against systems you do not own or have permission to test. Synthetic entities are fictional; public examples come from PoisonedRAG (MIT).

Подробная инструкция на русском: [README_RU.md](README_RU.md).

## Pipeline

```
data sources ──► build_datasets.py ──► validate_datasets.py ──► tokenize_data.py
                                                                     │
                       configs/qlora_*.json ──► train_qlora.py ◄─────┘
                                                     │  (best validation checkpoint)
                                                     ▼
                                    generate_candidates.py ──► score_candidates.py
                                                     │
                                         make_rag_jobs.py ──► external RAG harness
```

| Step | Script | What it does |
|---|---|---|
| Data | `scripts/build_datasets.py` | Builds Russian synthetic examples (750 fictional entities × 2 facts, entity-disjoint splits) and normalises published PoisonedRAG documents grouped by query. 20 % of synthetic rows are clean controls (target = original). |
| Checks | `scripts/validate_datasets.py`, `tests/` | No shared group, exact question or exact target document across splits; checksums. |
| Tokens | `scripts/tokenize_data.py`, `check_tokens.py` | Chat template; prompt tokens get label `-100`, only the assistant document is the loss target; over-length rows raise instead of being truncated (max 345 tokens vs 512 limit). |
| Train | `scripts/train_qlora.py` | Transformers + PEFT + bitsandbytes NF4. Early stopping on validation loss, best checkpoint restored. Test split never used. |
| Campaign | `scripts/run_campaign.py`, `preflight_gpu.py`, `monitor_campaign_gpu.py` | GPU preflight, batch-size benchmark, time-boxed training run, status journal. |
| Generate / score | `scripts/generate_candidates.py`, `score_candidates.py`, `compare_generators.py` | Greedy generation for base vs tuned on the same prompts; literal target presence, length, latency, token-cap rate. |
| RAG jobs | `scripts/make_rag_jobs.py` | Exports paired clean/poisoned jobs for the evaluation harness. |

## Dataset

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| Synthetic Russian | 1,200 | 150 | 150 |
| PoisonedRAG (EN, frozen commit `f660d72`) | 1,092 | 132 | 140 |
| **Total** | **2,292** | **282** | **290** |

`data/sft/` – chat messages for training; `data/mixed/` – same rows with provenance; `data/generation_prompts/` – inputs without reference answers; `data/rag/` – clean corpora and cases for RAG tests. See `docs/DATA_CARD_RU.md`.

## Training runs

Both runs: one RTX 5090 (32 GB), 4-bit NF4 backbone, BF16 compute.

| | 4B | 14B |
|---|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` | `unsloth/Qwen3-14B-bnb-4bit` (upstream `Qwen/Qwen3-14B`) |
| LoRA | r 16, α 32, attention q/k/v/o | r 64, α 128, all linear layers |
| Trainable params | 11.8 M | 256.9 M |
| Batch / LR | — / see `configs/qlora_4b.json` | 8 / 5e-5 (selected by benchmark, `docs/RTX5090_BENCHMARK.json`) |
| Steps | 574 (2 epochs) | 600 (early stop; best = step 300) |
| Train runtime | 27.1 min | 19.6 min |
| Train loss | 0.611 | 0.509 |
| Best validation loss | 0.690 | 0.554 |

Raw manifests, histories and metrics: `training_results/`. Loss is token-level prediction loss on the document completion, **not** an attack success rate.

## Trained adapters (Git LFS)

| Path | Format | SHA-256 |
|---|---|---|
| `adapters/qwen3-4b-qlora/adapter_model.safetensors` | FP32, as saved by training | `7ea8a70bdf8bb6142c705d0a6c46ac7bacf7608c7ca8e2b40ae9396d92ccc1a5` |
| `adapters/qwen3-14b-qlora/adapter_model.safetensors` | **FP16 cast** of the saved FP32 adapter (to fit LFS quota; max abs cast error 3.1e-5) | `4cac567d6ab9c6f1c3d7d6c0ebb3e6ec48b084c04381c83d8126c18c959ad55a` |

Original FP32 14B file saved by the training run: SHA-256 `af6684556c0e878f73eaf7c5ae23298ede6b8646787bb7b0fbc954a3281ab132`.
Note: the cloud generator used in the later RAG experiments reported adapter hash `3d8f3dfcf12da677cbaff2a0997589c5d816eec09ef4a3116d039ad8c4b859a4`; the cause of the difference (re-save vs. different file) is not resolved from the local artefacts.

Load with PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-14B-bnb-4bit", device_map="auto")
model = PeftModel.from_pretrained(base, "adapters/qwen3-14b-qlora")
tok = AutoTokenizer.from_pretrained("adapters/qwen3-14b-qlora")
```

## RAG evaluation results (summary)

Measured in the separate harness: target assistant Qwen3 8B (Ollama), `qwen3-embedding:0.6b`, top-k 6; 20 held-out cases (18 conflicts + 2 controls), 3 modes (none / defensive prompt / prompt + context-checking model), paired clean vs poisoned, 240 calls per run. Labels are assistant semantic review, human confirmation pending.

Target ASR (out of 18):

| Generator | Corpus | none | prompt | guard |
|---|---|---|---|---|
| base 4B → tuned 4B | short fact cards | 8 → 6 | 6 → 2 | 3 → 9 |
| base 14B → tuned 14B | short fact cards | 4 → 3 | 1 → 1 | 2 → 6 |
| base 14B → tuned 14B | dense encyclopedia articles | 9 → 12 | 7 → 8 | 4 → 4 |

Takeaways: fine-tuning mostly changed *format* (tuned 14B documents had far fewer invented extra details: 0 vs 12 and 3 vs 10 of 18); the corpus structure mattered more than tuning — a short direct card beats a fact buried in a long article. 20 cases is a small subset, not a robustness estimate.

## Reproduce

```sh
python3 scripts/validate_datasets.py
python3 -m unittest discover -s tests -v
# GPU machine
sh scripts/bootstrap_gpu.sh          # CUDA torch + requirements-train.txt
python3 scripts/train_qlora.py --config configs/qlora_4b.json
```

Server workflow: `docs/SERVER_RU.md`, 14B campaign: `docs/CAMPAIGN_14B_RU.md`. Remote helper scripts read host/key from environment variables (`GPU_SSH_HOST`, `GPU_SSH_PORT`, `GPU_SSH_KEY`).

Not included: the private archive of earlier attack trials (never used for training), secrets, virtual environments.

## References

- Chang et al., *One Shot Dominance*, Findings of EMNLP 2025 (AuthChain threat model; not reproduced here).
- Zou et al., *PoisonedRAG* — https://github.com/sleeepeer/PoisonedRAG (MIT).
- Qwen3 model cards; Hugging Face PEFT / bitsandbytes QLoRA docs.
