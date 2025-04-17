#! /usr/bin/env bash
set -e

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# set environment variables
source ../.env 2> /dev/null || source .env

# default directory to save files in
DIR="$SCRIPT_DIR"/../data/cifar10
mkdir -p "$DIR"

# download tar file
curl -C - https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz --output "$DIR"/cifar-10-python.tar.gz

# extract tar file
tar xfv "$DIR"/cifar-10-python.tar.gz -C "$DIR"

# move data out of "ugly" `cifar-10-batches-py` folder
mv "$DIR"/cifar-10-batches-py/* "$DIR"
rmdir "$DIR"/cifar-10-batches-py