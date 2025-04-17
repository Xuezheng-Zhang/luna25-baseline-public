run_rsync() {
  echo started syncing to node "$1"
  ssh "$1" mkdir -p /scratch/"$USER"/
  rsync /scratch/"$USER"/virtual_environments/ "$1":/scratch/"$USER"/virtual_environments/ -ah --delete
  echo completed syncing to node "$1"
}

if [[ "$HOSTNAME" != "cn84"* ]]; then
  echo run this script from cn84
  exit
fi

# DAS cluster
run_rsync cn79  # CPU-only
run_rsync cn104
run_rsync cn105
run_rsync cn117

# ICIS cluster
run_rsync cn114
run_rsync cn115