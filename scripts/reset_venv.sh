# This script can be used to reset your virtual environment to a clean install without
# any dependencies installed. Please use this only AFTER you've called
# setup_virtual_environment.sh at least once.

VENV_DIR=$PWD/.venv

if [ -L "$VENV_DIR" ]; then
  echo "detected $VENV_DIR is a symlink to $(realpath "$VENV_DIR")"
  VENV_DIR="$(realpath "$VENV_DIR")"
fi

if [ -d "$VENV_DIR" ]; then
  echo "removing existing virtual environment $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR"/bin/activate
pip install --upgrade pip wheel
