#!/usr/bin/env bash
set -euo pipefail

rm -rf dataset output artifacts

make clean
mkdir -p artifacts

python3 generate_dataset.py \
    --out dataset \
    --count 200 \
    --width 128 \
    --height 128

make build

./batch_blur.exe \
    -i dataset \
    -o output \
    -n 200 \
    -r 2 \
    -t 256 \
    | tee artifacts/execution.log

cp dataset/image_0000.pgm artifacts/before_image_0000.pgm
cp output/image_0000.pgm artifacts/after_image_0000.pgm

echo "Proof artifacts prepared in artifacts/"
ls -lh artifacts/
