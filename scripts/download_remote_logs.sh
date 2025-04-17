#! /usr/bin/env bash

# change working directory to folder where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set environment variables
source ../.env 2> /dev/null || source .env

# determine where to download to
TARGET_DIR=$SCRIPT_DIR/../logs/cluster
mkdir -p "$TARGET_DIR"

# rsync remote logs
echo "this script requires you to set variables SCIENCE_USERNAME and REPO_PATH"; exit 0 # delete line once you've set the username
SCIENCE_USERNAME=$USER # science username
CLUSTER_REPO_PATH=~/cluster-skeleton-python # path to folder of this repo on the cluster

rsync -azP "$SCIENCE_USERNAME"@cn84.science.ru.nl:"$CLUSTER_REPO_PATH"/logs/ "$TARGET_DIR"