import cmd
import subprocess
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # force GPU 0

# ── Config ────────────────────────────────────────────────────────────────────
OPT_MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    #"facebook/opt-6.7b",
]


BLOOM_MODELS = [
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
]

EXPERIMENTS = [
    ("FP16",         []),
    ("RTN_4bit",     ["--wbits", "4", "--nearest"]),
    ("GPTQ_4bit",    ["--wbits", "4"]),
    ("GPTQ_3bit",    ["--wbits", "3"]),
]

# Tasks supported by the repo (lambada, piqa, arc_easy, arc_challenge, storycloze)
# storycloze needs manual download so skip for now
TASKS = ["lambada", "piqa", "arc_easy", "arc_challenge"]

BASE_OUTPUT_DIR = "zeroShot_results"

# ── Runner ────────────────────────────────────────────────────────────────────
def run_zeroshot(model, task, extra_args=[]):
    model_short = model.split("/")[-1]
    cmd = ["python", "main.py", model, "c4", "--task", task] + extra_args
    print(f"    CMD: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="zeroShot")
    output = result.stdout + result.stderr
    if result.returncode != 0:
        real_errors = [l for l in output.split('\n')
                      if 'error' in l.lower() and 'warning' not in l.lower()
                      and 'futurewarning' not in l.lower()
                      and 'userwarn' not in l.lower()]
        if real_errors:
            print(f"    ⚠️  Error: {real_errors[-1][:150]}")
        return None

    # Parse JSON result from output
    try:
        # Find JSON block in output
        lines = output.split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        if json_start != -1:
            json_str = '\n'.join(lines[json_start:])
            # Find end of JSON
            brace_count = 0
            end_idx = 0
            for i, ch in enumerate(json_str):
                if ch == '{': brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            data = json.loads(json_str[:end_idx])
            return data.get('results', {}).get(task, {})
    except Exception as e:
        print(f"    ⚠️  Parse error: {e}")
    return None

# ── Main ──────────────────────────────────────────────────────────────────────
def run_family(models, family_name):
    family_dir = os.path.join(BASE_OUTPUT_DIR, family_name)

    for model in models:
        model_short = model.split("/")[-1]
        model_dir = os.path.join(family_dir, model_short)
        os.makedirs(model_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Model: {model_short}  [{family_name}]")
        print(f"{'='*60}")

        for exp_name, extra_args in EXPERIMENTS:
            exp_file = os.path.join(model_dir, f"{exp_name}.json")

            # Skip if already done
            if os.path.exists(exp_file):
                print(f"\n  ⏭️  Skipping {exp_name} - already exists")
                continue

            print(f"\n  Running: {exp_name}...")
            all_results = {}

            for task in TASKS:
                print(f"    Task: {task} ", end="", flush=True)
                result = run_zeroshot(model, task, extra_args)
                if result:
                    all_results[task] = result
                    # Print key metric
                    acc = result.get('acc', result.get('acc_norm', '?'))
                    print(f"✅ acc={acc:.4f}" if isinstance(acc, float) else f"✅ {result}")
                else:
                    print(f"⚠️  Failed")

            # Save results
            with open(exp_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "model": model,
                    "experiment": exp_name,
                    "extra_args": extra_args,
                    "results": all_results
                }, f, indent=2)

            print(f"  💾 Saved to {exp_file}")

        # Ask to continue after each model
        model_idx = models.index(model)
        if model_idx < len(models) - 1:
            next_model = models[model_idx + 1].split("/")[-1]
            print(f"\n{'='*60}")
            try:
                answer = input(f"  ✅ {model_short} done! Continue to {next_model}? (y/n): ").strip().lower()
            except EOFError:
                answer = 'y'
            if answer != 'y':
                print(f"\n  ⏹️  Stopping. Run again to continue from next model.")
                return False
    return True

# ── Entry Point ───────────────────────────────────────────────────────────────
print("\n🚀 Starting Zero-Shot Evaluation")
print(f"📁 Results will be saved to: {BASE_OUTPUT_DIR}/")
print(f"   OPT_family/ → opt-125m/, opt-350m/, ...")
print(f"   BLOOM_family/ → bloom-560m/, bloom-1b1/, ...")
print(f"\nTasks: {TASKS}")
print(f"Experiments: {[e[0] for e in EXPERIMENTS]}")

try:
    answer = input("\nRun OPT family first? (y/n): ").strip().lower()
except EOFError:
    answer = 'y'

if answer == 'y':
    cont = run_family(OPT_MODELS, "OPT_family")
    if cont:
        try:
            answer2 = input("\nNow run BLOOM family? (y/n): ").strip().lower()
        except EOFError:
            answer2 = 'y'
        if answer2 == 'y':
            run_family(BLOOM_MODELS, "BLOOM_family")
else:
    run_family(BLOOM_MODELS, "BLOOM_family")

print("\n✅ Zero-shot evaluation session complete!")
print(f"\nFolder structure created:")
for root, dirs, files in os.walk(BASE_OUTPUT_DIR):
    level = root.replace(BASE_OUTPUT_DIR, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level >= 2:
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')


            