#!/usr/bin/env bash
set -euo pipefail

BENCH="${BENCH:-$HOME/NaVILA-Bench}"
MIRROR="${MP3D_MIRROR:-$HOME/NaVILA-Dataset/MP3D-mirror}"
ZIP_PATH="${MP3D_HABITAT_ZIP:-$MIRROR/downloads/mp3d_habitat.zip}"
SCENES_ROOT="${MP3D_SCENES_ROOT:-$MIRROR/extracted/MatterPort3D/mp3d}"
TRAIN_META="${TRAIN_META:-$HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz}"
MANIFEST="${MP3D_NAVMESH_MANIFEST:-$MIRROR/navmesh_manifest_train.json}"
SCENE_LIST="${MP3D_TRAIN_SCENE_LIST:-$MIRROR/train_scenes.txt}"
URL="${MP3D_HABITAT_URL:-http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip}"

mkdir -p "$(dirname "$ZIP_PATH")" "$SCENES_ROOT" "$(dirname "$MANIFEST")"

echo "[MP3D] $(date -Is) downloading Habitat archive"
curl \
  --location \
  --continue-at - \
  --fail \
  --retry 999 \
  --retry-delay 10 \
  --retry-connrefused \
  --speed-limit 1024 \
  --speed-time 60 \
  --output "$ZIP_PATH" \
  "$URL"

echo "[MP3D] $(date -Is) testing archive"
python -m zipfile --test "$ZIP_PATH"

echo "[MP3D] $(date -Is) extracting required train GLBs"
python "$BENCH/scripts/prepare_mp3d_train_assets.py" \
  --zip "$ZIP_PATH" \
  --metadata "$TRAIN_META" \
  --output-root "$SCENES_ROOT" \
  --scene-list "$SCENE_LIST" \
  --require-scene-count 61

echo "[MP3D] $(date -Is) generating train navmeshes through Slurm"
srun \
  -p "${SLURM_PARTITION:-gpuA800}" \
  --gres=gpu:1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --mem="${SLURM_MEM:-64G}" \
  bash -lc "
    set -euo pipefail
    module load anaconda/3-2024.02.01
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    conda activate \"${HABITAT_CONDA_ENV:-vlnce3}\"
    cd \"$BENCH\"
    export PYTHONPATH=\"$BENCH:\$PYTHONPATH\"
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_mp3d_navmeshes.py \
      --scenes-root \"$SCENES_ROOT\" \
      --metadata \"$TRAIN_META\" \
      --output-manifest \"$MANIFEST\" \
      --minimum-coverage 0.99
  "

echo "[MP3D] $(date -Is) complete"
