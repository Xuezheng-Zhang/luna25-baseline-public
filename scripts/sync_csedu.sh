# select an account which is allowed to submit to the csedu cluster
ACCOUNT=$(groups | grep -o '\bcsedu[^[:space:]]*' | head -n 1)

run_rsync() {
  echo started syncing to node "$1"
  echo running command 'mkdir -p /scratch/"$USER"'
  srun -p csedu-prio -A "$ACCOUNT" -w "$1" --qos csedu-small mkdir -p /scratch/"$USER"
  echo running command 'rsync cn84:/scratch/"$USER"/virtual_environments/ /scratch/"$USER"/virtual_environments/ -ah --delete'
  srun -p csedu-prio -A "$ACCOUNT" -w "$1" --qos csedu-small rsync cn84:/scratch/"$USER"/virtual_environments/ /scratch/"$USER"/virtual_environments/ -ah --delete
  echo completed syncing to node "$1"
}

if [[ "$HOSTNAME" != "cn84"* ]]; then
  echo run this script from cn84
  exit
fi

# gpu nodes
run_rsync cn47
run_rsync cn48

# cpu nodes
run_rsync cn77
run_rsync cn78

