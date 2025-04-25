#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --account=cseduproject # or your course code
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=./logs/experiment1_%j.out
#SBATCH --error=./logs/experiment1_%j.err
#SBATCH --mail-user=YOURNAME

project_dir=.
source "$project_dir"/venv/bin/activate
python3 "$project_dir"/train.py

