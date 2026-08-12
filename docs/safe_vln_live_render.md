# Go2 Safe-VLN v5 strict live rendering and goal stopping

> Current CMDP semantics: the constraint uses cumulative cost over each
> episode, matching SafeVLA. Costs are not divided by the number of macro
> actions, so a collision, fall, or forward-blocked event cannot be diluted by
> extending the trajectory.

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
- Native Isaac-camera collection uses the same
  `navila_uniform_full_history_v1` sampler.  Older native shards marked
  `linear_8_of_available` remain valid for safety statistics, but are rejected
  when converting to the strict v5 Actor dataset; recollect them instead of
  relabeling the frame contract.
- Native collection restores the official episode start pose after the
  excluded locomotion warm-up, clears sensor histories and timeout counters,
  then renders and force-refreshes the attached cameras. Every real frame
  records both the Go2 root pose and the camera world pose.
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

## Historical v4 training sequence (do not run as mainline)

This section records the superseded live dynamic-Oracle experiment. Its
collection commands are intentionally rejected unless the explicit diagnostic
ablation flag is supplied. They are retained to interpret old artifacts, not
as current training instructions; use **Safety instrumentation reset (current
mainline)** below for new runs.

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

Train a fresh LoRA actor on a deterministic 2,000-transition subset. Balanced
sampling retains all STOP labels, covers all 500 episodes and 61 scenes, and
then fills the remaining slots across action classes. The post-training audit
must reach at least 50% STOP accuracy and 40% non-STOP macro accuracy before
the checkpoint is accepted. Then warm its new critics on a separately
risk-balanced 2,000-transition subset:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py warmup-actor \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --dataset-dir outputs/safe_live_v4_oracle_500 \
  --output-dir checkpoints/safe_vln_v4_actor_bc_v2 \
  --training-dtype bfloat16 \
  --actor-lr 1e-6 \
  --oracle-stop-weight 5 \
  --epochs 1 \
  --mini-batch-size 4 \
  --max-samples 2000 \
  --sampling-strategy balanced-oracle \
  --sampling-seed 20260729

CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py warmup-critics \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --checkpoint checkpoints/safe_vln_v4_actor_bc_v2 \
  --reset-critics \
  --dataset-dir outputs/safe_live_v4_oracle_500 \
  --output-dir checkpoints/safe_vln_v4_warm_v2 \
  --training-dtype bfloat16 \
  --epochs 1 \
  --max-samples 2000 \
  --sampling-strategy balanced-critic \
  --sampling-seed 20260729
```

Start `vlm_server.py` with `checkpoints/safe_vln_v4_warm_v2` and
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
  --checkpoint checkpoints/safe_vln_v4_warm_v2 \
  --rollout-dir outputs/safe_live_v4_on_policy_500 \
  --output-dir checkpoints/safe_vln_v4_ppo_v1 \
  --training-dtype bfloat16 \
  --actor-lr 1e-6 \
  --critic-lr 1e-4 \
  --ppo-epochs 1 \
  --mini-batch-size 1 \
  --oracle-ce-coef 0.05 \
  --oracle-stop-weight 5 \
  --max-samples 2000 \
  --sampling-strategy balanced-ppo \
  --sampling-seed 20260729 \
  --policy-version 0
```

The PPO command inherits λ from the input checkpoint unless
`--initial-lagrange-multiplier` is explicitly supplied. It updates λ once per
rollout batch using costs from every complete rollout episode, not only the
2,000 selected optimization transitions. It prints `lambda_before`,
`lambda_after`, mean episode cost, cost limit, and constraint excess. STOP
weighting is normalized by sample count, so the 5× factor remains effective
even with `--mini-batch-size=1`.

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

## Historical Safe-VLN v5 hierarchical actor

v5 is a new data contract. Existing v4 shards and checkpoints are legacy
read-only inputs and must not be appended to or used for v5 actor warmup. v5
matches original NaViLA frame preparation: repeated-first padding for short
histories, seven uniformly spaced observations over the complete history, then
the latest observation. Its dynamic Oracle quantizes the distance remaining
outside the episode success radius, so 25/50/75 cm actions are all supervised.

Run an eight-episode collection smoke test inside an allocated A800 node:

```bash
EPISODE_COUNT=8 DATASET_DIR=$HOME/NaVILA-Bench/outputs/safe_live_v5_smoke_8 \
  bash scripts/run_safe_vln_v5_collect_oracle.sh
```

After inspecting the smoke audit, collect the same deterministic 500 training
IDs used by v4 (offset 80) into a new directory:

```bash
bash scripts/run_safe_vln_v5_collect_oracle.sh
```

The script automatically runs `audit_safe_vln_v5.py`. Formal acceptance
requires exactly 500 episodes and 61 scenes, unique observation keys, strict
pose/frame alignment, all ten actions with at least 50 samples each, and at
least 150 STOP samples. It never appends to an existing v5 directory.

Train the hierarchical actor from the original NaViLA checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_safe_vln_v5_warmup_actor.sh
```

Stage 1 caches one multimodal state feature per scheduled sample and trains only
the STOP/direction/magnitude head for 20 epochs. Stage 2 unfreezes only the
fresh LoRA adapter for one epoch. Complete episodes are split into train,
calibration, and audit before sampling. Calibration selects the STOP threshold;
the untouched audit split performs final acceptance. Training batches contain
10% STOP samples; the remainder is uniform over the nine motion actions. The
motion pool reserves up to 25% per action for non-STOP samples 0--1 m outside
the goal radius, preventing premature image-only STOP. New data marks
repeated-first padding explicitly; old black-left shards are legacy and are
not reused.
checkpoint is accepted only when audit STOP recall is at least 0.50, non-goal
false STOP rate is at most 0.05, non-STOP macro accuracy is at least 0.40, and
all probabilities are finite and normalized. The calibration curve, 10x10
confusion matrix, and per-sample predictions are written beside the checkpoint.

The previous hierarchical checkpoint can be diagnosed without modifying it:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py audit-actor \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --checkpoint checkpoints/safe_vln_v5_actor_hierarchical_v1 \
  --dataset-dir outputs/safe_live_v5_oracle_500 \
  --output-dir outputs/safe_vln_v5_actor_hierarchical_v1_diagnostic \
  --training-dtype bfloat16 \
  --dev-episodes-per-scene 1 \
  --stop-threshold-grid-step 0.01
```

A 64-transition A800 training smoke can be run against the accepted full v5
dataset without weakening the full-dataset audit:

```bash
MAX_SAMPLES=64 \
OUTPUT_DIR=$HOME/NaVILA-Bench/checkpoints/safe_vln_v5_actor_smoke_64 \
LOG_ROOT=$HOME/NaVILA-Bench/outputs/safe_vln_v5_actor_smoke_64_logs \
  bash scripts/run_safe_vln_v5_warmup_actor.sh
```

New checkpoints contain `actor_head.pt` and `actor_config.json` in addition to
the LoRA adapter and critic heads. `vlm_server.py` selects this architecture
automatically; checkpoints without `actor_config.json` retain the legacy
candidate-response scorer. `actor_config.json` schema v2 records the calibrated
STOP threshold and direction-major action mapping.

## v6 sensor-gated safety elicitation

`--goal-stop-mode sensor-gated` makes the aligned geodesic distance
authoritative. Inside the success radius it executes STOP; outside the radius
it rejects a premature policy STOP and executes the highest-probability motion
action. Missing distance or missing motion probabilities terminates safely.
Every intervention remains visible in policy/system metrics and is excluded
from PPO because the executed action was not sampled by the policy.

Certify the existing factorized motion actor without changing its source
checkpoint:

```bash
sbatch slurm/run_safe_vln_v6_certify_gated_actor.sh
```

Build the fixed 80-episode safety-elicitation list from the strict v5 data:

```bash
python scripts/safe_vln_main.py select-risk-episodes \
  --dataset-dir outputs/safe_live_v5_oracle_500 \
  --output outputs/safe_vln_v6_risk80.json
```

The selector writes both JSON diagnostics and a newline-delimited ID file. It
selects 20 hard-event, 20 near-obstacle, 20 maneuver-risk, and 20 low-risk
control episodes while limiting each scene to two episodes when possible.
Start stochastic rollout inference with a reproducible stream:

```bash
python scripts/vlm_server.py \
  --model_path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --safe_checkpoint checkpoints/safe_vln_v5_actor_factorized_gated_v1 \
  --no-safe_deterministic --safe_sampling_seed 20260801 --port 54321
```

## Historical v6 online DAgger recovery loop (diagnostic only)

The commands in this section intentionally use the retired live navmesh Oracle.
They now require `--allow-online-oracle` in the collector and are not part of
the reported mainline; use the current-mainline section below for new data.

The initial actor was found to over-select turning macro-actions after its own
closed-loop mistakes. The recovery loop remains online in simulation: Go2
executes the served actor action, Habitat renders the next 8-frame RGB history
from that physical pose, and the dynamic shortest-path oracle labels the same
state. Pose, map, navmesh and geodesic distance are used only to produce the
label, reward, cost, and safety gate; they are never added to NaViLA inputs.

First collect 80 on-policy training episodes from the accepted sensor-gated
actor. The collector tags every transition with `recovery_category`; in
particular, `forward_after_turn` means the dynamic oracle requires a forward
action while the policy still proposed a turn.

```bash
SAFE_VLN_CHECKPOINT=$HOME/NaVILA-Bench/checkpoints/safe_vln_v6_actor_balanced_gated_v1 \
SAFE_VLN_DATASET_DIR=$HOME/NaVILA-Bench/outputs/safe_vln_v6_dagger_r1_rollout \
SAFE_VLN_LOG_ROOT=$HOME/NaVILA-Bench/outputs/safe_vln_v6_dagger_r1_logs \
SAFE_VLN_POLICY_TAG=v6-dagger-r1-p0 \
SAFE_VLN_ONLINE_ROUND=1 \
SAFE_VLN_REQUIRE_ONLINE_DAGGER=1 \
SAFE_VLN_STOCHASTIC=1 \
sbatch slurm/run_safe_vln_v6_collect_risk80.sh
```

The collection audit rejects non-VLM samples, invalid policy statistics,
misaligned frames, missing policy version, or a rollout containing no
forward-after-turn recovery state. Then train the actor with 60% online data
and 40% static expert anchors. Forward-after-turn errors are sampled 4x,
other action mismatches 2x, and matching states 1x.

```bash
SAFE_VLN_DAGGER_SOURCE_CHECKPOINT=$HOME/NaVILA-Bench/checkpoints/safe_vln_v6_actor_balanced_gated_v1 \
SAFE_VLN_DAGGER_ROLLOUT_DIR=$HOME/NaVILA-Bench/outputs/safe_vln_v6_dagger_r1_rollout \
SAFE_VLN_DAGGER_ANCHOR_DIR=$HOME/NaVILA-Bench/outputs/safe_live_v5_oracle_500 \
SAFE_VLN_DAGGER_OUTPUT_DIR=$HOME/NaVILA-Bench/checkpoints/safe_vln_v6_dagger_actor_r1 \
sbatch slurm/run_safe_vln_v6_dagger_actor.sh
```

This output is deliberately diagnostic-only. Certify it again before serving:

```bash
SAFE_VLN_ACTOR_CHECKPOINT=$HOME/NaVILA-Bench/checkpoints/safe_vln_v6_dagger_actor_r1 \
SAFE_VLN_ACTOR_DATASET=$HOME/NaVILA-Bench/outputs/safe_live_v5_oracle_500 \
SAFE_VLN_ACTOR_CERTIFIED_OUTPUT=$HOME/NaVILA-Bench/checkpoints/safe_vln_v6_dagger_actor_r1_gated \
sbatch slurm/run_safe_vln_v6_certify_gated_actor.sh
```

Evaluate the certified checkpoint on the fixed 22 `val_unseen` episodes. Only
continue to SafePPO when success reaches 2/22, mean cost is at most the v0
baseline (0.925326), hard-violation rate is at most 12/22, and blocked rate is
at most 1/22. Before SafePPO, collect another 80 on-policy DAgger episodes
from the certified actor (use a new `SAFE_VLN_DATASET_DIR` and set
`SAFE_VLN_REQUIRE_ONLINE_DAGGER=1` as above). Then set the four required
variables below. `run_safe_vln_v6_dagger_ppo.sh` re-audits that fresh rollout
and enforces the held-out gate before using an oracle cross-entropy coefficient
of 0.20; otherwise it exits without creating a PPO checkpoint.

```bash
SAFE_VLN_PPO_CHECKPOINT=$HOME/NaVILA-Bench/checkpoints/safe_vln_v6_dagger_actor_r1_gated \
SAFE_VLN_PPO_ROLLOUT_DIR=$HOME/NaVILA-Bench/outputs/safe_vln_v6_dagger_r1_ppo_rollout \
SAFE_VLN_PPO_EVAL_SUMMARY=$HOME/NaVILA-Bench/outputs/safe_vln_v6_eval_val_unseen_dagger_r1_logs/summary.json \
sbatch slurm/run_safe_vln_v6_dagger_ppo.sh
```

## Inspect online RGB histories

Each completed strict rollout stores the exact eight RGB inputs used by
NaViLA. Export annotated 4×2 contact sheets for one episode with:

```bash
PYTHONPATH=$HOME/NaVILA-Bench python scripts/visualize_safe_vln_episode.py \
  --dataset-dir outputs/safe_vln_v6_dagger_r1_rollout \
  --episode-id 1509 \
  --output-dir outputs/safe_vln_v6_dagger_r1_preview_1509
```

The sheet records the instruction, dynamic Oracle action, policy/executed
action and probabilities, reward, cost, recovery category, and turn streak.
Repeated entries marked `history_padding` are expected at the start of an
episode; Safe-VLN now repeats the first valid frame instead of injecting a
synthetic black observation. Add
`--only-recovery` to export only `forward_after_turn` states.

If a live rollout contains a terminal state without a valid Oracle, repair that
episode separately and merge it immutably before actor training:

```bash
SAFE_VLN_CHECKPOINT=$HOME/NaVILA-Bench/checkpoints/safe_vln_v6_actor_balanced_gated_v1 \
SAFE_VLN_EPISODE_IDS=9263 \
SAFE_VLN_DATASET_DIR=$HOME/NaVILA-Bench/outputs/safe_vln_v6_dagger_r1_repair_9263 \
SAFE_VLN_LOG_ROOT=$HOME/NaVILA-Bench/outputs/safe_vln_v6_dagger_r1_repair_9263_logs \
sbatch slurm/run_safe_vln_v6_collect_risk80.sh

PYTHONPATH=$HOME/NaVILA-Bench python scripts/merge_safe_vln_datasets.py \
  --source-dir outputs/safe_vln_v6_dagger_r1_rollout \
  --source-dir outputs/safe_vln_v6_dagger_r1_repair_9263 \
  --replace-episode 9263 \
  --output-dir outputs/safe_vln_v6_dagger_r1_repaired
```

## Safety instrumentation reset (current mainline)

The old factorized/online-DAgger commands above are historical ablations. The
current mainline does not train from the live dynamic Oracle: it is only a
diagnostic label and requires `--allow-online-oracle` to enable. Recollect
strict paired data after the PhysX contact self-test passes. Safe-VLN records
`contact_sensor_enabled`, `turn_execution`, `turn_tracking_failure` (with the
deprecated `turn_blocked` alias), and the achieved/requested yaw ratio for
every macro action. A yaw tracking miss is diagnostic only: only unsafe base
contact, fall, or sustained forward blockage is a hard cost/termination.

Use the calibration summary before training:

```bash
PYTHONPATH=$HOME/NaVILA-Bench python scripts/summarize_action_execution.py \
  --input outputs/safe_calibration_v7
```

Checkpoint lifecycle is fail closed. A critic-only checkpoint retains the
original deterministic NaViLA text Actor and exposes only reward/cost values;
it cannot create PPO-eligible action statistics. A replacement discrete Actor
is served only after an independent held-out audit with non-zero acceptance
thresholds. Diagnostic checkpoints are rejected unless an explicit ablation
flag is supplied.

The reproducible mainline is now deliberately shorter than the historical
online-DAgger branch:

1. Collect strict VLM rollouts with Isaac's native camera or
   `--safe-live-render`, contact processing enabled, and the original NaViLA
   Actor. Do not pass `--allow-online-oracle`.
2. Accept only transactional v5 episode directories whose audit passes, then
   warm up the two critics. Critic warmup uses one value-only multimodal
   forward per sample and writes an explicit `critic-only` checkpoint role.
3. Do not pass `--no-safe_deterministic` to a critic-only server and do not use
   its rollouts for PPO. First train/certify a discrete Actor on disjoint
   train/calibration/audit episodes. Only a checkpoint with role `policy` and
   interface `safe-vln-discrete-v1` may collect on-policy PPO data.
4. Evaluate that audited policy on held-out VLN-CE episodes. The `summarize`
   output reports cumulative episode cost (the CMDP constraint), collision,
   blockage, and turn-tracking diagnostics separately.

The old DAgger shell jobs exit unless `SAFE_VLN_ALLOW_ONLINE_ORACLE=1` is set;
that switch is only for a separately reported diagnostic ablation and must not
be mixed into the mainline training set.

The native-camera array job starts the VLM server without a Safe-VLN
checkpoint. This is intentional: original NaViLA supplies the actions, while
physical reward/cost returns collected from Go2 supervise the new critics
later. The job fails early if an inherited `SAFE_VLN_CHECKPOINT` is present,
so a stale shell variable cannot silently turn base collection into a random
or previously trained replacement-Actor rollout.
Each worker requests one GPU and nine CPUs, so the two array tasks can run on
the same compatible node without requiring a two-GPU gang allocation.
The job converts the official VLN-CE train metadata and GT into Isaac
coordinates, verifies all 61 USD scenes, scene-balances the episode order, and
records both source SHA-256 hashes in every sample. The bundled
`vln_ce_isaac_v1.json.gz` is an 11-scene evaluation asset and is rejected for
native training.

```bash
SAFE_VLN_NATIVE_OUTPUT_ROOT=$HOME/NaVILA-Bench/outputs/safe_vln_v10_native_base80 \
SAFE_VLN_NATIVE_LOG_ROOT=$HOME/NaVILA-Bench/outputs/safe_vln_v10_native_base80_logs \
SAFE_VLN_NATIVE_EPISODES_PER_WORKER=40 \
sbatch slurm/run_safe_vln_v8_native_camera_collect_2gpu.sh

PYTHONPATH=$HOME/NaVILA-Bench python scripts/merge_safe_vln_datasets.py \
  --source-dir outputs/safe_vln_v10_native_base80/gpu0 \
  --source-dir outputs/safe_vln_v10_native_base80/gpu1 \
  --output-dir outputs/safe_vln_v10_native_base80_merged

PYTHONPATH=$HOME/NaVILA-Bench python scripts/audit_safe_vln_v5.py \
  --dataset-dir outputs/safe_vln_v10_native_base80_merged \
  --allow-small-dataset --require-navila-teacher

SAFE_VLN_CRITIC_DATASET=$HOME/NaVILA-Bench/outputs/safe_vln_v10_native_base80_merged \
SAFE_VLN_CRITIC_OUTPUT=$HOME/NaVILA-Bench/checkpoints/safe_vln_v10_critics \
sbatch slurm/run_safe_vln_v6_warmup_critics.sh
```

Before SafePPO, train a discrete head to reproduce original NaViLA rather than
starting from a random candidate scorer. Use a larger native dataset with at
least three episodes per scene (500 episodes/61 scenes is the acceptance
target), because whole episodes from every scene are reserved independently
for threshold calibration and final audit. The command is:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/safe_vln_main.py warmup-actor \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --dataset-dir $HOME/NaVILA-Bench/outputs/safe_vln_v10_native_base500_merged \
  --output-dir $HOME/NaVILA-Bench/checkpoints/safe_vln_v10_actor_distilled \
  --device cuda --training-dtype bfloat16 \
  --actor-architecture hierarchical-stop-motion \
  --actor-target-source navila-policy \
  --sampling-strategy stratified \
  --head-warmup-epochs 20 --epochs 1 --max-samples 4000
```

Only an output whose `trainer_state.json` says `checkpoint_role=policy`,
`policy_interface=safe-vln-discrete-v1`, and
`actor/audit_target_source=original-navila-policy` is deployable. A failed
fidelity audit remains diagnostic and cannot be passed to PPO.
