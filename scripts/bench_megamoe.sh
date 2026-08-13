#!/bin/bash
# MegaMoE fused vs non-fused (EP dispatch + GEMM1 + SwiGLU + GEMM2 + EP combine)
# across batch sizes, on 8xB200.
set -u
source ~/DeepEP/runenv.sh
OUT=${1:-~/deepep_results/megamoe}
mkdir -p "$OUT"
cd ~/DeepGEMM

for tok in 128 256 512 1024 2048 4096 8192; do
  f="$OUT/mega_tok${tok}.log"
  timeout 2400 python tests/test_mega_moe.py \
      --num-max-tokens-per-rank "$tok" --num-tokens "$tok" \
      --num-correctness-tests 1 > "$f" 2>&1
  echo "=== tokens/rank=$tok"
  grep -E "baseline breakdown|sum of kernels|EP  0/8" "$f" | head -3
done
