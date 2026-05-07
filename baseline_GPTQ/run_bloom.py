import subprocess
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
]

DATASETS = ["wikitext2", "ptb", "c4"]

EXPERIMENTS = [
    ("FP16 Baseline",        []),
    ("RTN 4-bit",            ["--wbits", "4", "--nearest"]),
    ("GPTQ 4-bit",           ["--wbits", "4"]),
    ("GPTQ 3-bit",           ["--wbits", "3"]),
    ("GPTQ 4-bit group-128", ["--wbits", "4", "--groupsize", "128"]),
]

OUTPUT_DIR = "BLOOM_family"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_ppl(output, dataset):
    lines = output.strip().split('\n')
    dataset_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == dataset:
            dataset_idx = i
            break
    if dataset_idx == -1:
        return None
    for line in lines[dataset_idx + 1:]:
        s = line.strip()
        if s in ['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new']:
            break
        try:
            val = float(s)
            if val > 1.0 and '.' in s:
                return val
        except:
            pass
    return None

# ── Runner ────────────────────────────────────────────────────────────────────
def run_one(model, dataset, extra_args=[]):
    cmd = ["python", "bloom.py", model, dataset] + extra_args
    print(f"    CMD: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        real_errors = [l for l in output.split('\n')
                      if 'error' in l.lower() and 'warning' not in l.lower()
                      and 'futurewarning' not in l.lower()]
        if real_errors:
            print(f"    ⚠️  Error: {real_errors[-1][:100]}")
    return parse_ppl(output, dataset)

# ── Main Loop ─────────────────────────────────────────────────────────────────
for idx, model in enumerate(MODELS):
    model_short = model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"  Model: {model_short}  ({idx+1}/{len(MODELS)})")
    print(f"{'='*60}")

    output_file = os.path.join(OUTPUT_DIR, f"results_{model_short}.txt")

    # Skip if already completed
    if os.path.exists(output_file):
        print(f"  ⏭️  Skipping {model_short} - results already exist")
        continue

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== GPTQ Experiments - {model_short} ===\n\n")

    for exp_name, extra_args in EXPERIMENTS:
        print(f"\n  Running: {exp_name}...")
        all_results = {}

        for dataset in DATASETS:
            print(f"    Dataset: {dataset} ", end="", flush=True)
            ppl = run_one(model, dataset, extra_args)
            if ppl is not None:
                all_results[dataset] = ppl
                print(f"✅ PPL={ppl:.4f}")
            else:
                print(f"⚠️  No result parsed")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"=== {exp_name} ===\n")
            for ds, ppl in all_results.items():
                f.write(f"{ds} : {ppl}\n")
            f.write("\n")

        print(f"  ✅ {exp_name}: {all_results}")

    print(f"\n  💾 Saved to {output_file}")

    # Ask to continue after each model (except last)
    if idx < len(MODELS) - 1:
        next_model = MODELS[idx + 1].split("/")[-1]
        print(f"\n{'='*60}")
        try:
            answer = input(f"  ✅ {model_short} done! Continue to {next_model}? (y/n): ").strip().lower()
        except EOFError:
            answer = 'y'
        if answer != 'y':
            print(f"\n  Stopping here. Run script again to continue from next model.")
            break

print("\n✅ Session complete!")
print(f"📁 Results saved in: {OUTPUT_DIR}/")
print(f"\nCompleted results:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")

