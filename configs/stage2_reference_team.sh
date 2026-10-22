#!/bin/bash
# Stage 2: one reference team. 1,150 train / 10 validation seats from selected versions of that team. Every epoch draws 300 shards with
# replacement with p ~ recency weight (make_split.py writes the weights; w3 and w5 are two settings of the same scheme). Export epochs: 30 / 33 / 30.
cd "$(dirname "$0")/../src"
python bc_train2.py --data ../data/ref_v3d --allow ../lists/ref_all.json --val-list ../lists/ref_cur_select.json --weights ../weights/v3e_w${W:-3}_weights.json \
  --epoch-draws 300 --init ../runs/s$SEED/s1/last.pt --out ../runs/s$SEED/s2 \
  --epochs 40 --lr 1.5e-4 --ch 256 --blocks 8 --device cuda --chunk 120 --tb-repeat 1 --seed $SEED --load-workers 2 --resume 1
