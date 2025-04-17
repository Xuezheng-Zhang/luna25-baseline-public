#! /usr/bin/env bash
set -e

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# load the environment variables from the `.env` file
source ../.env 2> /dev/null || source .env

# check $NFS_USER_DIR and $PROJECT_NAME are defined in the .env file
if [ -z "$NFS_USER_DIR" ]; then
    echo "Please set $NFS_USER_DIR in .env file."
    exit
fi
if [ -z "$PROJECT_NAME" ]; then
    echo "Please set $PROJECT_NAME in .env file."
    exit
fi

echo "using NFS_USER_DIR=$NFS_USER_DIR and PROJECT_NAME=$PROJECT_NAME"

# make a directory on the network-attached file system to store logs and checkpoints
echo creating directory "$NFS_USER_DIR"
mkdir -p "$NFS_USER_DIR"
chmod 700 "$NFS_USER_DIR" # only you can access

echo creating directory "$NFS_USER_DIR"/"$PROJECT_NAME"/data
mkdir -p "$NFS_USER_DIR"/"$PROJECT_NAME"/data

echo creating directory "$NFS_USER_DIR"/"$PROJECT_NAME"/logs
mkdir -p "$NFS_USER_DIR"/"$PROJECT_NAME"/logs

# and make a symlink to access it directly from the root of the project
echo "creating symlink $PWD/data pointing to $NFS_USER_DIR/$PROJECT_NAME/data"
ln -sfn "$NFS_USER_DIR"/"$PROJECT_NAME"/data "$SCRIPT_DIR"/../data

echo "creating symlink $PWD/logs pointing to $NFS_USER_DIR/$PROJECT_NAME/logs"
ln -sfn "$NFS_USER_DIR"/"$PROJECT_NAME"/logs "$SCRIPT_DIR"/../logs
