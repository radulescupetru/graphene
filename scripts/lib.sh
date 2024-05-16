#!/usr/bin/env bash

set -euo pipefail

function is_venv_activated {
    # check if virtualenv is activated (conda or pip)
    if [[ -z ${VIRTUAL_ENV:-} ]] && [[ -z ${CONDA_DEFAULT_ENV:-} ]]; then
        echo "ERROR: virtualenv is not activated"
        return 1
    else
        echo "virtualenv is activated"
        return 0
    fi
}


function is_macos {
    if [[ $(uname) == "Darwin" ]]; then
        return 0
    else
        return 1
    fi
}
