#!/usr/bin/env bash
set -x
# OFFSET=$RANDOM
for percent in 1 5 10; do
    for fold in 1 2 3 4 5; do
        python $(dirname "$0")/semi_coco.py --percent ${percent} --seed ${fold} --data-dir $1 --save-dir $2
    done
done
