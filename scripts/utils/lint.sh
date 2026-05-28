#!/bin/bash
# Format and sort imports across the live Python sources.
set -euo pipefail
python3 -m black idisc scripts
python3 -m isort idisc scripts
