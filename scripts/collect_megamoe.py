"""Aggregate the MegaMoE fused vs non-fused sweep."""
import re
import glob
import os
import sys

d = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/deepep_results/megamoe')
KERN = re.compile(r'^\s+([\d.]+) us\s+(.*)$')
PERF = re.compile(r'EP\s+0/\s*8 \|\s*(\d+) TFLOPS \|.*?\|\s*(\d+) us,.*?\| ([\d.]+)x')

BUCKET = [
    ('GEMM (L1+L2, routed+shared)', lambda n: 'gemm' in n),
    ('SwiGLU', lambda n: 'swiglu' in n),
    ('EP dispatch', lambda n: 'dispatch_impl' in n),
    ('EP dispatch epilogue', lambda n: 'dispatch_copy_epilogue' in n),
    ('EP combine', lambda n: 'combine_impl' in n),
    ('EP combine epilogue', lambda n: 'combine_reduce_epilogue' in n),
]

rows = []
for p in sorted(glob.glob(f'{d}/mega_tok*.log'),
                key=lambda p: int(re.search(r'tok(\d+)', p).group(1))):
    tok = int(re.search(r'tok(\d+)', p).group(1))
    kern, fused_us, tflops, speed, total = {}, None, None, None, None
    for line in open(p, errors='ignore'):
        m = KERN.match(line.rstrip())
        if m:
            name = m.group(2)
            if 'sum of kernels' in name:
                total = float(m.group(1))
            else:
                kern[name] = float(m.group(1))
        g = PERF.search(line)
        if g and fused_us is None:
            tflops, fused_us, speed = int(g.group(1)), int(g.group(2)), float(g.group(3))
    if fused_us is None or total is None:
        continue
    buckets = {}
    for label, pred in BUCKET:
        buckets[label] = sum(v for k, v in kern.items() if pred(k.lower()))
    comm = sum(v for k, v in buckets.items() if k.startswith('EP'))
    comp = total - comm
    rows.append((tok, fused_us, total, speed, tflops, comp, comm, buckets))

print(f'{"tok/rank":>9} {"unfused":>9} {"fused":>9} {"speedup":>8} {"TFLOPS":>7} | '
      f'{"bl compute":>11} {"bl comm":>9} {"comm %":>7} | {"fused vs":>9}')
print(f'{"":>9} {"(total)":>9} {"":>9} {"":>8} {"(fused)":>7} | '
      f'{"GEMM+act":>11} {"EP":>9} {"of total":>7} | {"bl compute":>9}')
print('-' * 96)
for tok, fused, total, speed, tflops, comp, comm, _ in rows:
    print(f'{tok:>9} {total:8.0f}us {fused:8}us {speed:7.2f}x {tflops:7} | '
          f'{comp:10.0f}us {comm:8.0f}us {comm / total * 100:6.1f}% | '
          f'{comp / fused:8.2f}x')

print('\nUnfused breakdown (us):')
labels = [b[0] for b in BUCKET]
print(f'{"tok/rank":>9} | ' + ' | '.join(f'{l.replace("EP ", "").replace(" (L1+L2, routed+shared)", ""):>10}' for l in labels))
print('-' * 96)
for tok, _, total, _, _, _, _, buckets in rows:
    print(f'{tok:>9} | ' + ' | '.join(f'{buckets[l]:10.1f}' for l in labels))
