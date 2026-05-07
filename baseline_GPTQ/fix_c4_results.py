import subprocess, re, os

models = ['facebook/opt-125m', 'facebook/opt-350m']
experiments = [
    ('FP16 Baseline', []),
    ('RTN 4-bit',     ['--wbits', '4', '--nearest']),
]

for model in models:
    model_short = model.split('/')[-1]
    output_file = f'OPT_family/results_{model_short}.txt'
    print(f'\n=== {model_short} ===')

    for name, args in experiments:
        cmd = ['python', 'opt.py', model, 'c4'] + args
        print(f'Running {name} on c4...')
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        lines = output.strip().split('\n')

        # Parse correct C4 PPL
        ppl = None
        for i, line in enumerate(lines):
            if line.strip() == 'c4':
                for line2 in reversed(lines[i+1:]):
                    s = line2.strip()
                    try:
                        val = float(s)
                        if val > 1.0 and '.' in s:
                            ppl = val
                            break
                    except:
                        pass

        if ppl is None:
            print(f'  ⚠️  Could not parse C4 PPL')
            continue

        print(f'  ✅ {name} c4 PPL: {ppl}')

        # Update results file
        with open(output_file, 'r') as f:
            content = f.read()

        section_start = content.find(f'=== {name} ===')
        if section_start == -1:
            print(f'  ⚠️  Section not found in file')
            continue

        section_end = content.find('=== ', section_start + 1)
        if section_end == -1:
            section_end = len(content)
        section = content[section_start:section_end]

        new_section = re.sub(
            r'c4 : [\d.]+',
            f'c4 : {ppl}',
            section
        )

        content = content[:section_start] + new_section + content[section_end:]

        with open(output_file, 'w') as f:
            f.write(content)

        print(f'  💾 Updated {output_file}')

print('\n✅ Done! Verifying final results...')
for model_short in ['opt-125m', 'opt-350m']:
    print(f'\n--- OPT_family/results_{model_short}.txt ---')
    with open(f'OPT_family/results_{model_short}.txt') as f:
        print(f.read())



        