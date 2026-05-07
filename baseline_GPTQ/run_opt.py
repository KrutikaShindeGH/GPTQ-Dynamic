import subprocess
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
]

DATASETS = ["wikitext2", "ptb", "c4"]

EXPERIMENTS = [
    ("FP16 Baseline",        []),
    ("RTN 4-bit",            ["--wbits", "4", "--nearest"]),
    ("GPTQ 4-bit",           ["--wbits", "4"]),
    ("GPTQ 3-bit",           ["--wbits", "3"]),
    ("GPTQ 4-bit group-128", ["--wbits", "4", "--groupsize", "128"]),
]

OUTPUT_DIR = "OPT_family"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_results(output, dataset):
    """
    opt.py prints: dataset name on one line, then layer indices 0,1,2...
    then the PPL float on its own line.
    Strategy: find dataset label, then find next valid float after it.
    Fallback: grab last valid float in entire output.
    """
    results = {}
    lines = output.split('\n')

    # Find where dataset name is printed
    dataset_line_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == dataset:
            dataset_line_idx = i
            break

    if dataset_line_idx == -1:
        # Fallback: last valid float in output
        for line in reversed(lines):
            stripped = line.strip()
            if '.' in stripped:
                try:
                    val = float(stripped)
                    if val > 1.0:
                        results[dataset] = val
                        return results
                except:
                    pass
        return results

    # Search forward from dataset label for PPL
    for j in range(dataset_line_idx + 1, len(lines)):
        stripped = lines[j].strip()
        if '.' in stripped:
            try:
                val = float(stripped)
                if val > 1.0:
                    results[dataset] = val
                    return results
            except:
                pass
    return results

# ── Runner ────────────────────────────────────────────────────────────────────
def run_experiment(model, dataset, extra_args=[]):
    cmd = ["python", "opt.py", model, dataset] + extra_args
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"  ⚠️  Error:\n{output[-300:]}")
    return parse_results(output, dataset)

# ── Main Loop ─────────────────────────────────────────────────────────────────
for model in MODELS:
    model_short = model.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"  Model: {model_short}")
    print(f"{'='*60}")

    output_file = os.path.join(OUTPUT_DIR, f"results_{model_short}.txt")

    if os.path.exists(output_file):
        print(f"  ⏭️  Skipping {model_short} - results already exist")
        continue

    with open(output_file, "w") as f:
        f.write(f"=== GPTQ Experiments - {model_short} ===\n\n")

    for exp_name, extra_args in EXPERIMENTS:
        print(f"\n  Running: {exp_name}...")
        all_results = {}

        for dataset in DATASETS:
            print(f"    Dataset: {dataset} ", end="", flush=True)
            results = run_experiment(model, dataset, extra_args)
            all_results.update(results)
            if results:
                print(f"✅ PPL={list(results.values())[0]:.2f}")
            else:
                print(f"⚠️  No results parsed")

        with open(output_file, "a") as f:
            f.write(f"=== {exp_name} ===\n")
            for ds, ppl in all_results.items():
                f.write(f"{ds} : {ppl}\n")
            f.write("\n")

        print(f"  ✅ {exp_name} done: {all_results}")

    print(f"\n  💾 Saved to {output_file}")

print("\n✅ All experiments complete!")
print(f"📁 Results saved in: {OUTPUT_DIR}/")

