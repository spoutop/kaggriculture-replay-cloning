#!/bin/bash
# Track A / A1: Kaggle daily episode dumps (CC0) kaggle/kaggriculture-episodes-YYYY-MM-DD -> replays/daily_YYYYMMDD/episodes/*.json.gz (+VALIDATION.json).
# usage: dl_daily.sh 2026-09-19 2026-09-20 ...
ROOT=${KGR_ROOT}
PY=$ROOT/.venv/bin/python; K=$ROOT/.venv/bin/kaggle
TA=$ROOT/ws/maclearn/rank1_20260923/trackA
LOG=$TA/logs/dl_daily.log
for D in "$@"; do
  DD=$(echo $D | tr -d '-'); OUT=$ROOT/replays/daily_$DD; mkdir -p $OUT
  echo "START $D $(date -u +%FT%TZ)" >> $LOG
  for attempt in 1 2 3; do
    ls $OUT/*.zip >/dev/null 2>&1 && break
    $K datasets download -d kaggle/kaggriculture-episodes-$D -p $OUT >> $LOG 2>&1 && break
    sleep 60
  done
  Z=$(ls $OUT/*.zip 2>/dev/null | head -1)
  if [ -z "$Z" ]; then echo "FAILED download $D" >> $LOG; continue; fi
  echo "  zip $(ls -la $Z) $(date -u +%FT%TZ)" >> $LOG
  $PY $TA/zip_to_gz.py $Z $OUT 2>&1 | tail -1 >> $LOG
  echo "END $D $(date -u +%FT%TZ)" >> $LOG
done
date -u +%FT%TZ > $TA/logs/dl_daily.done
