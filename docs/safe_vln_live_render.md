# Go2 Safe-VLN v4 strict live rendering and goal stopping

`--safe-live-render` keeps Isaac cameras disabled and renders NaViLA RGB frames
in a separate headless Habitat-Sim process from the current physical Go2 pose.
The image, proposed action, executed action, next physical state, geodesic reward, and Go2 safety cost
therefore belong to one executed transition.

The currently downloaded mirror is:

```text
$HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d
```

It contains the 11 R2R `val_unseen` GLBs. These scenes are evaluation-only and
must not be used to train the upper policy.

## 1. Install Habitat-Sim

Use the Python 3.8 `vlnce3` environment:

```bash
module load anaconda/3-2024.02.01
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vlnce3
conda install \
  /share/home/202430461770/habitat-sim-0.1.7-py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2
export PYTHONPATH=$HOME/NaVILA-Bench:$PYTHONPATH
```

## 2. Generate the missing navmeshes

Run this on an A800 allocation. Habitat-Sim needs an EGL-capable GPU even when
the RGB renderer is disabled during navmesh generation.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_mp3d_navmeshes.py \
  --scenes-root \
    $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --metadata \
    $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --output-manifest \
    $HOME/NaVILA-Dataset/MP3D-mirror/navmesh_manifest.json
```

The command exits with status 2 when fewer than 99% of the 1,839 episodes have
a start and goal within 0.25 m of the generated navmesh and a finite path.

## 3. Start the two model/render services

Habitat terminal (`vlnce3`):

```bash
cd $HOME/NaVILA-Bench
export PYTHONPATH=$HOME/NaVILA-Bench:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python scripts/habitat_render_server.py \
  --scenes-root \
    $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --host 127.0.0.1 \
  --port 54322 \
  --gpu-device-id 0
```

NaViLA terminal (`navila`):

```bash
cd $HOME/NaVILA-Bench
export PYTHONPATH=$HOME/NaVILA:$HOME/NaVILA-Bench:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python scripts/vlm_server.py \
  --model_path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --port 54321
```

## 4. Run the first strict episode

Activate `vlnce-isaac` and restore the same GLIBC variables used for
Safe-Replay. Do not pass `--enable_cameras`.

```bash
$GLIBC_RUN $CONDA_PREFIX/bin/python scripts/navila_eval.py \
  --task=go2_matterport_vision \
  --num_envs=1 \
  --history_length=9 \
  --load_run=2024-09-25_23-22-02 \
  --headless \
  --safe-vln \
  --safe-live-render \
  --vlnce-episode-id=1 \
  --vlnce-metadata=$HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --vlnce-gt=$HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz \
  --mp3d-scenes-root=$HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --render-host=127.0.0.1 \
  --render-port=54322 \
  --vlm_host=127.0.0.1 \
  --vlm_port=54321 \
  --dataset-role=eval \
  --safe-dataset-dir=$HOME/NaVILA-Bench/outputs/safe_live_val_unseen_ep1 \
  --max_vlm_calls=18
```

Batch collection uses the same contract:

```bash
python scripts/safe_vln_main.py evaluate \
  --safe-live-render \
  --vlnce-episode-ids 1 2 3 \
  --vlnce-metadata \
    $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --vlnce-gt \
    $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz \
  --mp3d-scenes-root \
    $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --dataset-role eval \
  --dataset-dir outputs/safe_live_val_unseen
```

## Strictness and failure behavior

- The renderer receives the Go2 root pose and independently verifies the
  Isaac/Habitat coordinate conversion.
- The client checks that the Go2 pose did not move while RGB was rendered.
- Horizontal round-trip error must be at most 2 cm and yaw error at most 1°.
- RGB history contains only live-rendered frames and is padded by repeating the
  earliest real frame.
- Renderer timeouts, wrong scenes, missing navmeshes, and pose mismatches abort
  the episode. No offline-image fallback is allowed.
- A complete episode is published atomically under `completed/<episode-id>`.
  Failed episodes remain absent from the dataset manifest.
- `dataset_role=eval` manifests are rejected by critic warmup and PPO training.

For full v4 training, replace the scene root and metadata with the authorized
complete MP3D train split, set `--dataset-role=train`, and recollect rollouts.
Do not mix v1/v2/v3 shards with v4 shards.

## v4 goal-stop contract

The episode-specific goal radius comes from the official VLN-CE metadata and is
sent to the renderer on every request. The renderer uses that same radius for
the dynamic oracle and echoes it back; a mismatch aborts the episode.

- `--goal-stop-mode=policy` reports the raw policy. Evaluation never overrides
  a movement action, even inside the goal radius.
- In train collection, a non-stop action inside the goal radius receives
  `-0.5` missed-stop reward. Three consecutive missed stops terminate the
  episode after the third macro action.
- `--goal-stop-mode=shield` still queries and records the model first, but
  executes STOP inside the goal radius. This is a system success and a shield
  intervention, not a policy success or PPO-eligible transition.
- Every transition stores `policy_action_id`, `executed_action_id`,
  `in_goal_radius`, `missed_stop`, `policy_success`, `system_success`, and
  `shield_intervened`.

## Fresh v4 training sequence

Full training requires all 61 authorized MP3D train GLBs and generated
navmeshes. The currently installed 11 `val_unseen` scenes remain evaluation
only.

Create three deterministic, scene-balanced, non-overlapping ID lists:

```bash
TRAIN_META=$HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
CALIBRATE_80_IDS=$(python scripts/select_vlnce_episode_ids.py \
  --metadata "$TRAIN_META" --count 80 --offset 0 --require-scene-count 61)
TRAIN_500_IDS=$(python scripts/select_vlnce_episode_ids.py \
  --metadata "$TRAIN_META" --count 500 --offset 80 --require-scene-count 61)
PPO_500_IDS=$(python scripts/select_vlnce_episode_ids.py \
  --metadata "$TRAIN_META" --count 500 --offset 580 --require-scene-count 61)
```

Calibrate the physical safety profile on the first 80 episodes:

```bash
python scripts/safe_vln_main.py calibrate-safety \
  --safe-live-render \
  --vlnce-episode-ids $CALIBRATE_80_IDS \
  --vlnce-metadata "$TRAIN_META" \
  --vlnce-gt $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --mp3d-scenes-root $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --dataset-role train \
  --goal-stop-mode policy \
  --calibration-dir outputs/safe_vln_v4_calibration_records \
  --output-profile checkpoints/safe_vln_v4_cost_profile.json
```

Collect 500 strictly aligned oracle episodes for behavior cloning:

```bash
python scripts/safe_vln_main.py collect \
  --safe-live-render \
  --vlnce-episode-ids $TRAIN_500_IDS \
  --vlnce-metadata $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz \
  --vlnce-gt $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --mp3d-scenes-root $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --dataset-role train \
  --collection-policy oracle \
  --goal-stop-mode policy \
  --safe-cost-profile checkpoints/safe_vln_v4_cost_profile.json \
  --dataset-dir outputs/safe_live_v4_oracle_500
```

Train a fresh LoRA actor with 5× STOP supervision, then warm its new critics:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py warmup-actor \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --dataset-dir outputs/safe_live_v4_oracle_500 \
  --output-dir checkpoints/safe_vln_v4_actor_bc \
  --training-dtype bfloat16 \
  --actor-lr 1e-6 \
  --oracle-stop-weight 5 \
  --epochs 1 \
  --mini-batch-size 1 \
  --max-samples 500

CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py warmup-critics \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --checkpoint checkpoints/safe_vln_v4_actor_bc \
  --reset-critics \
  --dataset-dir outputs/safe_live_v4_oracle_500 \
  --output-dir checkpoints/safe_vln_v4_warm \
  --training-dtype bfloat16 \
  --epochs 1 \
  --max-samples 500
```

Start `vlm_server.py` with `checkpoints/safe_vln_v4_warm` and
`--no-safe_deterministic`, collect 500 VLM-policy episodes in
`--goal-stop-mode=policy`, then run PPO:

```bash
python scripts/safe_vln_main.py collect \
  --safe-live-render \
  --vlnce-episode-ids $PPO_500_IDS \
  --vlnce-metadata "$TRAIN_META" \
  --vlnce-gt $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --mp3d-scenes-root $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --dataset-role train \
  --collection-policy vlm \
  --goal-stop-mode policy \
  --safe-cost-profile checkpoints/safe_vln_v4_cost_profile.json \
  --dataset-dir outputs/safe_live_v4_on_policy_500
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py train \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --checkpoint checkpoints/safe_vln_v4_warm \
  --rollout-dir outputs/safe_live_v4_on_policy_500 \
  --output-dir checkpoints/safe_vln_v4_ppo_v1 \
  --training-dtype bfloat16 \
  --actor-lr 1e-6 \
  --critic-lr 1e-4 \
  --ppo-epochs 1 \
  --mini-batch-size 1 \
  --oracle-ce-coef 0.05 \
  --oracle-stop-weight 5 \
  --policy-version 0
```

The PPO command inherits λ from the input checkpoint unless
`--initial-lagrange-multiplier` is explicitly supplied. It updates λ once per
rollout batch and prints `lambda_before`, `lambda_after`, mean episode cost,
cost limit, and constraint excess.

For the fixed 100-ID `val_unseen` list, run two separate output directories:
one with `--goal-stop-mode=policy`, then the same IDs with
`--goal-stop-mode=shield`. Compare `policy_success_rate`,
`system_success_rate`, missed-stop count, goal escape rate, shield intervention
rate, SPL, and cost. Never use either evaluation directory for training.

```bash
VAL_META=$HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
VAL_100_IDS=$(python scripts/select_vlnce_episode_ids.py \
  --metadata "$VAL_META" --count 100 --offset 0 --require-scene-count 11)

python scripts/safe_vln_main.py evaluate \
  --safe-live-render --vlnce-episode-ids $VAL_100_IDS \
  --vlnce-metadata "$VAL_META" \
  --vlnce-gt $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz \
  --mp3d-scenes-root $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --dataset-role eval --goal-stop-mode policy \
  --safe-cost-profile checkpoints/safe_vln_v4_cost_profile.json \
  --dataset-dir outputs/safe_vln_v4_eval_policy_100

python scripts/safe_vln_main.py evaluate \
  --safe-live-render --vlnce-episode-ids $VAL_100_IDS \
  --vlnce-metadata "$VAL_META" \
  --vlnce-gt $HOME/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz \
  --mp3d-scenes-root $HOME/NaVILA-Dataset/MP3D-mirror/extracted/MatterPort3D/mp3d \
  --dataset-role eval --goal-stop-mode shield \
  --safe-cost-profile checkpoints/safe_vln_v4_cost_profile.json \
  --dataset-dir outputs/safe_vln_v4_eval_shield_100
```
