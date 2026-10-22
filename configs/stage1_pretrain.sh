#!/bin/bash
# Stage 1: behaviour cloning on many teams. 18,667 train / 2,074 validation game-seats. One shard per (game, seat) from bc_data.py.
# --init is an earlier widened clone checkpoint (bc_clean_w256m.pt); --allow lists the shards of the stage-1 corpus; SEED = 311 / 312 / 313.
cd "$(dirname "$0")/../src"
python bc_train2.py --data ../data/multi ../data/frontier ../data/multi_old ../data/multi_0923 ../data/gtnow_p \
  --allow ../lists/stage1_allow.json --init ../init/bc_clean_w256m.pt --out ../runs/s$SEED/s1 \
  --epochs 24 --lr 3e-4 --ch 256 --blocks 8 --device cuda --chunk 120 --tb-repeat 1 --val-last 1 --max-val 60 --seed $SEED --load-workers 8 --resume 1
