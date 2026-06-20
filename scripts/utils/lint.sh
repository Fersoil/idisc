#!/bin/bash
set -euo pipefail
python3 -m black idisc scripts
python3 -m isort idisc scripts
