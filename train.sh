#!/bin/bash

python3 train.py \
    -d exp_0317/ \
    -n raft_must_b10_e20_lr1e-02 \
    -l 48 \
    -t demo/prompt.txt \
    -r 48 \
    -lr 1e-02 \
    -s 'prob-based' \
    -ds 2