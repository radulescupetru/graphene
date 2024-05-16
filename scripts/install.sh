#!/usr/bin/env bash

# bash strict mode
set -euo pipefail

# https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script#comment127065688_246128
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# shellcheck source-path=scripts
source "$SCRIPT_DIR"/lib.sh

if ! is_venv_activated; then
  exit 0
fi

pip install -U pip-tools

# Base pip-compile arguments
PIP_COMPILE_ARGS=(
  --resolver=backtracking
  --allow-unsafe
  --all-extras
  --no-emit-index-url
  --strip-extras
  --verbose
)


# Check for UPGRADE environment variable
if [[ ${UPGRADE_PIP_COMPILE:-false} == "true" ]]; then
  echo "Upgrading requirements to latest compatible versions"
  PIP_COMPILE_ARGS+=("--upgrade")
fi

compile_requirements() {
  local output_file=$1
  shift # remove first argument (output file) from $@

  pip-compile "${PIP_COMPILE_ARGS[@]}" "$@" -o "$output_file" pyproject.toml &
}

if is_macos; then
  compile_requirements "requirements/macos.lock"
fi

# wait for all the pip-compile processes to finish
wait
