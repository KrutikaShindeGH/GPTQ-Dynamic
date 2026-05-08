# Dynamic Hessian-Based Precision Scaling for Post-Training Quantization

The deployment of Large Language Models
(LLMs) is fundamentally constrained by high
computational overhead. Post-Training Quantization
(PTQ), specifically the Generative Pre-
Trained Transformer Quantization (GPTQ) algorithm,
has emerged as a leading method for
compressing model weights by utilizing Hessian
information to compensate for quantization
error. However, standard GPTQ applies
a uniform bit-width across all layers, which
leads to suboptimal performance. In this work,
we propose a dynamic, per-layer precision scaling
method based on global loss sensitivity to
optimize bit-width allocation. Our framework
(Algorithm 1) profiles each layer’s sensitivity
using the Fisher Information Matrix (FIM), allocates
bit-widths via a greedy constrained optimization,
and compensates for quantization
error using a modified GPTQ second-order update
rule. In our experiments the method is
labelled FP16 (Dynamic) to reflect that it operates
at a mixed average bit-rate. We evaluate on
the OPT and BLOOM model families and show
that FP16 (Dynamic) achieves lower perplexity
than uniform GPTQ at matched bit-rates
across WikiText2, PTB, and C4 benchmarks,
with competitive zero-shot accuracy on LAMBADA,
ARC, PIQA, and StoryCloze.


## Repository Structure

```
combined_project/
├── baseline_GPTQ/          # Uniform GPTQ baseline (Frantar et al., 2022)
│   ├── gptq.py               # Core GPTQ quantization algorithm
│   ├── opt.py                # OPT model quantization entry point
│   ├── bloom.py              # BLOOM model quantization entry point
│   ├── quant.py              # Quantization primitives
│   ├── datautils.py          # Calibration data loading (WikiText2, PTB, C4)
│   ├── modelutils.py         # Model utility functions
│   ├── llama.py              # LLaMA model support
│   ├── quant_cuda.cpp        # CUDA kernel bindings
│   ├── quant_cuda_kernel.cu  # CUDA quantization kernel
│   ├── setup_cuda.py         # CUDA build script
│   └── zeroShot/             # Zero-shot evaluation harness
│       ├── main.py           # Evaluation entry point
│       ├── evaluator.py      # Task evaluator
│       ├── tasks/            # LAMBADA, ARC, PIQA, StoryCloze tasks
│       └── models/           # Model wrappers for zero-shot eval
│
├── dynamic_GPTQ/           # Our proposed method: Dynamic Hessian-Based Precision Scaling
│   ├── main.py               # Entry point — runs Algorithm 1 end-to-end
│   ├── gptq.py               # Modified GPTQ with per-layer bit-width support
│   ├── opt.py                # OPT model with dynamic precision scaling
│   ├── bloom.py              # BLOOM model with dynamic precision scaling
│   ├── quant.py              # Quantization primitives (extended for mixed precision)
│   ├── utils.py              # FIM sensitivity profiling & bit allocation (Algorithm 1)
│   ├── datautils.py          # Calibration data loading
│   └── modelutils.py         # Model utility functions
│
└── results/                # Experimental results (Tables 1–11 in the report)
    ├── baseline/
    │   ├── perplexity/       # WikiText2, PTB, C4 perplexity for FP16, RTN, GPTQ
    │   │   ├── OPT_family/   # OPT-125M, 350M, 1.3B, 2.7B, 6.7B
    │   │   └── BLOOM_family/ # BLOOM-560M, 1.1B, 1.7B, 3B, 7.1B
    │   └── zeroshot/         # LAMBADA, ARC, PIQA, StoryCloze accuracy
    │       ├── OPT_family/
    │       └── BLOOM_family/
    └── dynamic/
        ├── perplexity/       # FP16 (Dynamic) perplexity results
        │   ├── OPT_family/
        │   └── BLOOM_family/
        └── zeroshot/         # FP16 (Dynamic) zero-shot accuracy
            ├── OPT_family/
            └── BLOOM_family/
```

---

## Method Summary (Algorithm 1)

Our framework implements three stages:

1. **Sensitivity Profiling** (`utils.py` → `PROFILESENSITIVITY`): Computes per-layer
   Fisher Information Matrix (FIM) traces as a proxy for quantization sensitivity (Eq. 1).

2. **Bit Allocation** (`utils.py` → `ALLOCATEBITS`): Greedy sequential bit-reduction
   assigns discrete bit-widths {2, 3, 4, 8} subject to a global average bit-rate
   constraint `Btarget` (Eq. 2).

3. **Quantization with Error Compensation** (`gptq.py` → `QUANTIZEMODEL`): Applies
   GPTQ's second-order weight update per layer using its assigned bit-width (Eq. 3).

---

## Baselines

| Method        | Description                                      |
|---------------|--------------------------------------------------|
| FP16          | Uncompressed full-precision model                |
| RTN-4 / RTN-3 | Round-to-nearest at 4-bit / 3-bit               |
| GPTQ-4        | Uniform 4-bit GPTQ (Frantar et al., 2022)       |
| GPTQ-3        | Uniform 3-bit GPTQ (Frantar et al., 2022)       |
| FP16 (Dyn.)  | **Ours** — mixed-precision, Hessian-guided      |

---

## Evaluation

- **Perplexity**: WikiText2, PTB, C4 (Tables 1–3 in report)
- **Zero-Shot Accuracy**: LAMBADA, ARC-Easy, ARC-Challenge, PIQA, StoryCloze (Tables 4–8)
- **Hardware**: Single NVIDIA A100 80GB GPU

---

## Execution Instructions

### Requirements

```bash
pip install torch transformers datasets numpy
```

To enable the fast CUDA quantization kernel (recommended for models ≥ 1.3B):

```bash
cd baseline_GPTQ/
python setup_cuda.py build_ext --inplace
```

---

### 1. Baseline GPTQ

All baseline commands are run from the `baseline_GPTQ/` directory.

#### Perplexity Evaluation

**OPT models — FP16 baseline:**
```bash
python opt.py facebook/opt-125m c4
python opt.py facebook/opt-350m c4
python opt.py facebook/opt-1.3b c4
python opt.py facebook/opt-2.7b c4
python opt.py facebook/opt-6.7b c4
```

**OPT models — RTN 4-bit:**
```bash
python opt.py facebook/opt-125m c4 --wbits 4 --nearest
python opt.py facebook/opt-350m c4 --wbits 4 --nearest
python opt.py facebook/opt-1.3b  c4 --wbits 4 --nearest
python opt.py facebook/opt-2.7b  c4 --wbits 4 --nearest
python opt.py facebook/opt-6.7b  c4 --wbits 4 --nearest
```

**OPT models — GPTQ 4-bit:**
```bash
python opt.py facebook/opt-125m c4 --wbits 4
python opt.py facebook/opt-350m c4 --wbits 4
python opt.py facebook/opt-1.3b  c4 --wbits 4
python opt.py facebook/opt-2.7b  c4 --wbits 4
python opt.py facebook/opt-6.7b  c4 --wbits 4
```

**OPT models — GPTQ 3-bit:**
```bash
python opt.py facebook/opt-125m c4 --wbits 3
python opt.py facebook/opt-350m c4 --wbits 3
python opt.py facebook/opt-1.3b  c4 --wbits 3
python opt.py facebook/opt-2.7b  c4 --wbits 3
python opt.py facebook/opt-6.7b  c4 --wbits 3
```

**OPT models — GPTQ 4-bit with group-size 128:**
```bash
python opt.py facebook/opt-125m c4 --wbits 4 --groupsize 128
python opt.py facebook/opt-350m c4 --wbits 4 --groupsize 128
python opt.py facebook/opt-1.3b  c4 --wbits 4 --groupsize 128
python opt.py facebook/opt-2.7b  c4 --wbits 4 --groupsize 128
python opt.py facebook/opt-6.7b  c4 --wbits 4 --groupsize 128
```

**BLOOM models — FP16 baseline:**
```bash
python bloom.py bigscience/bloom-560m  c4
python bloom.py bigscience/bloom-1b1   c4
python bloom.py bigscience/bloom-1b7   c4
python bloom.py bigscience/bloom-3b    c4
python bloom.py bigscience/bloom-7b1   c4
```

**BLOOM models — RTN 4-bit:**
```bash
python bloom.py bigscience/bloom-560m c4 --wbits 4 --nearest
python bloom.py bigscience/bloom-1b1  c4 --wbits 4 --nearest
python bloom.py bigscience/bloom-1b7  c4 --wbits 4 --nearest
python bloom.py bigscience/bloom-3b   c4 --wbits 4 --nearest
python bloom.py bigscience/bloom-7b1  c4 --wbits 4 --nearest
```

**BLOOM models — GPTQ 4-bit:**
```bash
python bloom.py bigscience/bloom-560m c4 --wbits 4
python bloom.py bigscience/bloom-1b1  c4 --wbits 4
python bloom.py bigscience/bloom-1b7  c4 --wbits 4
python bloom.py bigscience/bloom-3b   c4 --wbits 4
python bloom.py bigscience/bloom-7b1  c4 --wbits 4
```

**BLOOM models — GPTQ 3-bit:**
```bash
python bloom.py bigscience/bloom-560m c4 --wbits 3
python bloom.py bigscience/bloom-1b1  c4 --wbits 3
python bloom.py bigscience/bloom-1b7  c4 --wbits 3
python bloom.py bigscience/bloom-3b   c4 --wbits 3
python bloom.py bigscience/bloom-7b1  c4 --wbits 3
```

> The calibration dataset argument (`c4`) can be replaced with `wikitext2` or `ptb`
> to calibrate on a different dataset. Perplexity is always reported on all three.

#### Zero-Shot Evaluation

Zero-shot evaluation uses a saved quantized model checkpoint. First save the quantized
weights, then run the zero-shot harness.

**Save quantized weights (example — OPT-1.3B GPTQ 4-bit):**
```bash
python opt.py facebook/opt-1.3b c4 --wbits 4 --save opt-1.3b-4bit.pt
```

**Run zero-shot evaluation:**
```bash
python zeroShot/main.py \
    --model facebook/opt-1.3b \
    --load opt-1.3b-4bit.pt \
    --task lambada piqa arc_easy arc_challenge storycloze
```

Repeat the save + eval pattern for each model and bit-width. For the FP16 baseline,
omit `--load` and do not pass `--wbits`:
```bash
python zeroShot/main.py \
    --model facebook/opt-1.3b \
    --task lambada piqa arc_easy arc_challenge storycloze
```

---

### 2. Dynamic GPTQ (Ours)

All dynamic commands are run from the `dynamic_GPTQ/` directory.

#### Perplexity Evaluation

`main.py` runs the full Algorithm 1 pipeline (sensitivity profiling → bit allocation →
quantization) in a single call. The key argument is `--target_bits`, which sets the
global average bit-rate constraint `Btarget`.

**OPT models — Dynamic 4-bit (Btarget = 4):**
```bash
python main.py facebook/opt-125m c4 --target_bits 4
python main.py facebook/opt-350m c4 --target_bits 4
python main.py facebook/opt-1.3b  c4 --target_bits 4
python main.py facebook/opt-2.7b  c4 --target_bits 4
python main.py facebook/opt-6.7b  c4 --target_bits 4
```

**OPT models — Dynamic 3-bit (Btarget = 3):**
```bash
python main.py facebook/opt-125m c4 --target_bits 3
python main.py facebook/opt-350m c4 --target_bits 3
python main.py facebook/opt-1.3b  c4 --target_bits 3
python main.py facebook/opt-2.7b  c4 --target_bits 3
python main.py facebook/opt-6.7b  c4 --target_bits 3
```

**BLOOM models — Dynamic 4-bit (Btarget = 4):**
```bash
python main.py bigscience/bloom-560m c4 --target_bits 4
python main.py bigscience/bloom-1b1  c4 --target_bits 4
python main.py bigscience/bloom-1b7  c4 --target_bits 4
python main.py bigscience/bloom-3b   c4 --target_bits 4
python main.py bigscience/bloom-7b1  c4 --target_bits 4
```

**BLOOM models — Dynamic 3-bit (Btarget = 3):**
```bash
python main.py bigscience/bloom-560m c4 --target_bits 3
python main.py bigscience/bloom-1b1  c4 --target_bits 3
python main.py bigscience/bloom-1b7  c4 --target_bits 3
python main.py bigscience/bloom-3b   c4 --target_bits 3
python main.py bigscience/bloom-7b1  c4 --target_bits 3
```

#### Zero-Shot Evaluation

Save the dynamically quantized model, then evaluate with the same zero-shot harness
from `baseline_GPTQ/zeroShot/`.

**Save dynamic quantized weights:**
```bash
python main.py facebook/opt-1.3b c4 --target_bits 4 --save opt-1.3b-dynamic-4bit.pt
```

**Run zero-shot evaluation:**
```bash
python ../baseline_GPTQ/zeroShot/main.py \
    --model facebook/opt-1.3b \
    --load opt-1.3b-dynamic-4bit.pt \
    --task lambada piqa arc_easy arc_challenge storycloze
```

---

### 3. Results & Visualization Pipeline

After collecting all results into `3_results/`, run the three visualization scripts
from the project root (where `3_results/` lives).

**Step 1 — Parse all results into a single JSON:**
```bash
python script1_parse_results.py --results_root ./3_results
```
Output: `results_data.json`

**Step 2 — Generate all tables (PNG + CSV):**
```bash
python script2_tables.py --data results_data.json --out_dir tables/
```
Output: `tables/table_*.png` and `tables/table_*.csv`

**Step 3 — Generate all figures (PNG):**
```bash
python script3_figures.py --data results_data.json --out_dir figures/
```
Output: `figures/fig*.png`

> **Re-running after adding dynamic results:** once dynamic perplexity `.txt` files and
> zero-shot `.json` files are placed under `3_results/dynamic/`, simply re-run Steps 1–3
> above. The scripts auto-detect `Dynamic_4bit` and `Dynamic_3bit` experiments and
> include them in all tables and figures automatically.

#### Running in Google Colab

Use `colab_runner.py` as a guide. Copy each cell block into a separate Colab cell and
run top-to-bottom. Before running, set `REPO_ROOT` at the top of Cell 1 to point to
your project directory using one of the three options provided (GitHub clone, manual
zip upload, or Google Drive mount).

---

## Reference

Frantar et al. (2022). *GPTQ: Accurate Post-Training Quantization for Generative
Pre-Trained Transformers.* arXiv:2210.17323

