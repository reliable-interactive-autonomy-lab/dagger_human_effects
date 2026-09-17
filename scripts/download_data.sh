#!/usr/bin/env bash
# Fetch robomimic's official demonstration datasets for pretraining base policies.
#
# Lift (proficient-human, low-dim) is the validated default: run
# scripts/validate_dataset.py after downloading to confirm this robosuite build
# reproduces the dataset's observation conventions.
set -euo pipefail
DEST="${1:-data/robomimic}"
PY="${PYTHON:-python}"
RM_DIR="$("$PY" -c 'import robomimic,os;print(os.path.dirname(robomimic.__file__))')"

"$PY" "$RM_DIR/scripts/download_datasets.py" \
  --download_dir "$DEST" --tasks lift --dataset_types ph --hdf5_types low_dim

echo
echo "==> validating observation conventions against this robosuite build"
"$PY" scripts/validate_dataset.py \
  --dataset "$DEST/lift/ph/low_dim_v141.hdf5" --env Lift
