# Go2 Safe-VLN v2

Safe-VLN v2 separates terminal safety events from bounded dense physical risk.
Safe-Replay continues to read eight NaViLA frames from R2R while Isaac PhysX,
contact sensing, robot state, and the Go2 RayCaster provide safety supervision.
By default, the replay ID is now resolved through the original
`R2R_VLNCE_v1-3_preprocessed/train/train.json.gz` metadata. Isaac therefore
loads the same Matterport scene, start pose, goal, reference path, and GT
trajectory instead of an unrelated physical episode. The replay objective
still uses graded oracle-action reward rather than physical navigation
progress.

The default metadata paths are:

```text
~/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
~/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz
```

They can be overridden with `--safe-replay-vlnce-metadata` and
`--safe-replay-vlnce-gt`. `--safe-replay-legacy-unpaired` is available only
for reproducing old datasets.

`offline_reference_same_episode` means the offline RGB and live Go2 physics
share the episode, scene, start, and goal. It does not claim strict per-step
alignment: if the learned action differs from the oracle action, the offline
reference video and the live robot pose can diverge. Records therefore also
store `strict_observation_state_alignment=false`.

## Objective

- Hard collision, fall, or confirmed blocking produces `hard_cost=1` and
  terminates immediately.
- Soft contact, tilt, proximity, blocking trend, speed near obstacles, and
  command discontinuity produce a non-terminal macro `dense_cost` in `[0, 0.1]`.
- The default episode cost limit is `0.25`.
- Replay rewards are `1/0.5/0.25` for exact/adjacent/two-step magnitudes,
  `0` across action families, `-1` for an incorrect stop, and `-0.5` for
  missing an oracle stop.
- Data uses `safe-vln-go2-v2`; objective fingerprints prevent mixing old or
  differently configured targets.

## Migration from PPO v2

Keep the PPO v2 actor, collect a new critic-calibration dataset, reset both
critics, and warm them up. Then collect a fresh rollout because PPO must use
value predictions from the warmed critics.

1. Start the PPO v2 VLM server with stochastic actions.
2. Run `calibrate-safety` on replay IDs 101 through 180 and save the generated
   cost profile.
3. Collect critic-only data on IDs 181 through 220 with that profile.
4. Run `warmup-critics --checkpoint .../safe_vln_ppo_v2 --reset-critics`.
5. Restart the server from the warmup checkpoint and collect PPO data on IDs
   221 through 300.
6. Train with `--policy-version 2`, `--cost-limit 0.25`, and a reset Lagrange
   multiplier of `0.001`.

The first v2 dataset is for critic warmup only. It must not be reused for PPO,
because its stored value predictions come from the old objective.

## Commands

Start the inherited PPO v2 actor in stochastic mode:

```bash
conda activate navila
cd ~/NaVILA-Bench
export PYTHONPATH=$HOME/NaVILA:$HOME/NaVILA-Bench:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python scripts/vlm_server.py \
  --model_path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --safe_checkpoint $HOME/NaVILA-Bench/checkpoints/safe_vln_ppo_v2 \
  --no-safe_deterministic \
  --port 54321
```

In the Isaac environment, define the replay ID groups once:

```bash
CALIB_IDS=$(seq -s ' ' 101 180)
CRITIC_IDS=$(seq -s ' ' 181 220)
PPO_IDS=$(seq -s ' ' 221 300)
EVAL_IDS=$(seq -s ' ' 301 320)
```

Collect 80 calibration episodes and fit the bounded soft-risk profile:

```bash
$GLIBC_RUN $CONDA_PREFIX/bin/python scripts/safe_vln_main.py calibrate-safety \
  --r2r-data-path isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz \
  --safe-replay \
  --safe-replay-root $HOME/NaVILA-Dataset/R2R \
  --safe-replay-ids $CALIB_IDS \
  --start-idx 100 \
  --vlm-host localhost \
  --vlm-port 54321 \
  --calibration-dir outputs/safe_v2_calibration_raw \
  --output-profile checkpoints/safe_v2_cost_profile.json \
  --max-vlm-calls 18
```

Collect a separate critic dataset with the fitted profile:

```bash
$GLIBC_RUN $CONDA_PREFIX/bin/python scripts/safe_vln_main.py collect \
  --safe-replay \
  --safe-replay-root $HOME/NaVILA-Dataset/R2R \
  --safe-replay-ids $CRITIC_IDS \
  --start-idx 180 \
  --vlm-host localhost \
  --vlm-port 54321 \
  --safe-cost-profile checkpoints/safe_v2_cost_profile.json \
  --dataset-dir outputs/safe_v2_critic_181_220 \
  --safe-policy-tag ppo2_v2critic \
  --max-vlm-calls 18
```

Reset both value heads while retaining the PPO v2 LoRA actor:

```bash
conda activate navila
CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py warmup-critics \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --checkpoint checkpoints/safe_vln_ppo_v2 \
  --reset-critics \
  --dataset-dir outputs/safe_v2_critic_181_220 \
  --output-dir checkpoints/safe_vln_v2_warmup \
  --device cuda \
  --training-dtype bfloat16 \
  --epochs 1
```

Restart the VLM server from `checkpoints/safe_vln_v2_warmup`, keep stochastic
sampling enabled, and collect IDs 221 through 300 into a new rollout directory:

```bash
$GLIBC_RUN $CONDA_PREFIX/bin/python scripts/safe_vln_main.py collect \
  --safe-replay \
  --safe-replay-root $HOME/NaVILA-Dataset/R2R \
  --safe-replay-ids $PPO_IDS \
  --start-idx 220 \
  --vlm-host localhost \
  --vlm-port 54321 \
  --safe-cost-profile checkpoints/safe_v2_cost_profile.json \
  --dataset-dir outputs/safe_v2_ppo_221_300 \
  --safe-policy-tag warmup_v2 \
  --max-vlm-calls 18
```

Then train policy version 2 into version 3:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py train \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --checkpoint checkpoints/safe_vln_v2_warmup \
  --rollout-dir outputs/safe_v2_ppo_221_300 \
  --output-dir checkpoints/safe_vln_ppo_v3 \
  --device cuda \
  --training-dtype bfloat16 \
  --actor-lr 1e-6 \
  --critic-lr 1e-4 \
  --cost-limit 0.25 \
  --lagrange-lr 0.035 \
  --initial-lagrange-multiplier 0.001 \
  --ppo-epochs 1 \
  --mini-batch-size 1 \
  --policy-version 2
```

Use a deterministic PPO v3 server for the held-out IDs 301 through 320:

```bash
$GLIBC_RUN $CONDA_PREFIX/bin/python scripts/safe_vln_main.py evaluate \
  --safe-replay \
  --safe-replay-root $HOME/NaVILA-Dataset/R2R \
  --safe-replay-ids $EVAL_IDS \
  --start-idx 300 \
  --vlm-host localhost \
  --vlm-port 54321 \
  --safe-cost-profile checkpoints/safe_v2_cost_profile.json \
  --safe-policy-tag ppo3_eval \
  --max-vlm-calls 18
```

Never append v2 samples to an old v1 directory; the manifest and objective
fingerprint checks deliberately reject that operation.
