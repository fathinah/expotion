#!/bin/bash

python3 train.py \
    -d exp_0114/ \
    -n face_must_b20_e20_lr1e-02 \
    -l 48 \
    -r 12 \
    -lr 1e-02 \
    -s 'prob-based' \
    -ds 2