#! /usr/bin/env bash

# for csedu
srun --gres=gpu:1 --time=4:00:00 --mem=10G --cpus-per-task=5 -p csedu --pty bash

# for das
# srun --gres=gpu:1 --time=4:00:00 --mem=10G --cpus-per-task=5 -p das --pty bash
