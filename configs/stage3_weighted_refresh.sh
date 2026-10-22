#!/bin/bash
# Stage 3: the reference team's recent games with extra weight on games against opponents rated 2,800 and above. 1,982 train / 23 validation seats.
# --late-shards lists shards whose late-game steps are drawn more often from day 20 (configs/late_shards_example.json shows the format).
# --init is the chosen stage-2 epoch (29 / 32 / 29 zero-based). Export epochs: 21 / 23 / 21 zero-based (22 / 24 / 22 in the write-up's 1-based figures).
cd "$(dirname "$0")/../src"
python bc_train_hp.py --data ../data/ref_v3f ../data/ref_refresh ../data/top3 ../data/newfam_r1 ../data/newfam_eval10 \
  --allow ../lists/stage3_allow.json --val-list ../lists/stage3_val.json --weights ../lists/stage3_weights.json --epoch-draws 300 \
  --init ../runs/s$SEED/s2/ck_epoch${E2:-29}.pt --out ../runs/s$SEED/s3 \
  --epochs 24 --lr 1.5e-4 --ch 256 --blocks 8 --device cuda --chunk 120 --tb-repeat 1 --seed $SEED --load-workers 2 --resume 1 --keep-from 0 --gpu-cache 1 \
  --late-shards ../configs/late_shards_example.json --late-from 20
python a5_select_v3e.py export --runs ../runs/s$SEED/s3 --eps ${E3:-21}     # -> runs/s$SEED/s3/np/ck_epoch21_t0.5_k2c35.npz (style token and T = 0.5 baked in)
python q8.py ../runs/s$SEED/s3/np/ck_epoch${E3:-21}_t0.5_k2c35.npz ../members/FS3W_s${SEED}_q8.npz   # int8 trunk for the 100 MiB upload cap
