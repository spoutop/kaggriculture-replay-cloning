#!/bin/bash
# Build and check a submission bundle from five baked members plus the shipped offsets and opening book. The six alternates and the detector are optional.
cd "$(dirname "$0")/../src"
python make_bundle_ens.py ../bundles/NAME ../members/FS3W_s311_q8.npz ../members/FS3W_s312_q8.npz ../members/FS3W_s313_q8.npz ../members/PXLT_s299_q8.npz ../members/PXLT_s301_q8.npz \
  --weights 1,1,1,1,1 --correction ../configs/correction_final.json --book ../book/book_deploy_v2.npz --book-params 'hyb=1&pmin=0.05'
python first_step_latency.py ../bundles/NAME                                   # fresh-process load + first step
python preflight_bundle.py ../bundles/NAME.tar.gz --r1 ../bundles/REFERENCE    # Kaggle's own loader + one official self-play episode
python bundle_parity_book.py --bundle ../bundles/NAME --spec ../bundles/NAME.spec.json --corr ../configs/correction_final.json --book-spec '../book/book_deploy_v2.npz?hyb=1&pmin=0.05' --seeds 33074950:33074952
