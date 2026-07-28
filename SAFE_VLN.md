# Go2 Safe-VLN

This extension adds safety-aware hierarchical navigation to NaVILA-Bench:

- the upper NaViLA policy selects one of ten navigation macro actions;
- the released Go2 locomotion policy remains frozen and executes velocity commands;
- Go2 base contact and dangerous orientation are independent CMDP costs;
- a collision or fall terminates the episode at the post-physics state;
- every macro action records reward, cost, reward value, cost value, log probability, and returns;
- independent reward/cost GAE feeds a Lagrangian constrained PPO objective;
- only LoRA actor parameters and the two critic heads are optimized.

Only `go2_matterport_vision` is supported. Safe mode is opt-in and does not change the legacy benchmark.

For strict image/state pairing without Isaac RTX cameras, see
[`docs/safe_vln_live_render.md`](docs/safe_vln_live_render.md).

## Environments verified on this machine

Use `vlnce-isaac` for Isaac Lab collection/evaluation:

```bash
conda activate /share/home/202430461770/.conda/envs/vlnce-isaac
export PYTHONPATH=$HOME/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce:$HOME/NaVILA-Bench/isaaclab_exts/omni.isaac.matterport:$PYTHONPATH
export PYTHONPATH=$HOME/IsaacLab/source/extensions/omni.isaac.lab:$HOME/IsaacLab/source/extensions/omni.isaac.lab_tasks:$HOME/IsaacLab/source/extensions/omni.isaac.lab_assets:$PYTHONPATH
```

Use `navila` for the model server and training. The checkout exists at `$HOME/NaVILA` but is not discoverable in that environment unless it is installed editable or added to `PYTHONPATH`:

```bash
conda activate /share/home/202430461770/.conda/envs/navila
export PYTHONPATH=$HOME/NaVILA:$HOME/NaVILA-Bench:$PYTHONPATH
```

## A800 camera evaluation compatibility

Isaac Sim 4.1 camera rendering is verified on the A800 node `g02n06` with
NVIDIA driver `550.78`. A node with driver `595.58.03` crashed while
initializing the Hydra viewport, before the Go2 environment or Safe-VLN code
ran. Prefer a verified 550-series node for `--headless --enable_cameras`.

The released Go2 checkpoint
`2024-09-25_23-22-02/model_26499.pt` predates the lower-level cost critic.
The two inference entry points automatically load its actor and reward critic,
allow only missing `cost_critic.*` parameters, and skip optimizer restoration.
Any other missing or unexpected model key remains a fatal compatibility error.
The initialized lower-level cost critic is not a trained safety value model;
Safe-VLN cost/value predictions continue to come from the upper-level model.

Use `--vlm_host=localhost` only when `vlm_server.py` runs on the same compute
node. For separate Slurm allocations, pass the VLM server node hostname.

## Action and safety contract

The server chooses a canonical integer `action_id` in `[0, 9]`: left 15/30/45 degrees, right 15/30/45 degrees, forward 25/50/75 cm, or stop. Velocity and duration are resolved locally by the Isaac client; values supplied by a remote server cannot replace executable commands.

At every low-level simulation step, Safe-VLN checks contact forces on the Go2
`base`, `*_hip`, `*_thigh`, and `*_calf` bodies. Feet are excluded so
normal ground contact is not treated as a collision. It also checks
projected-gravity orientation and sustained forward progress. A non-foot force
above `1.0 N`, orientation above `0.8 rad`, or less than `0.10 m`
displacement during `2.0 s` of continuous forward commands terminates the
macro action and episode immediately. Reward remains separate:

```text
reward = distance_before - distance_after - 0.01 + 10 * success
cost   = unsafe_non_foot_contact + fall + blocked
```

Strict live-render v4 additionally applies `-0.5` to every movement decision
made inside the episode's official goal radius. Raw-policy and goal-shield
results are stored separately; a shield STOP is never labelled as a policy
success or PPO sample.

## Evaluation and collection

Start either a legacy NaViLA server or a server with a Safe-VLN checkpoint in the `navila` environment:

```bash
python scripts/vlm_server.py \
  --model_path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --safe_checkpoint outputs/policy_v1 \
  --port 54321
```

In another terminal, activate `vlnce-isaac`. Collect collision-labelled visual transitions:

```bash
python scripts/safe_vln_main.py collect \
  --dataset-dir outputs/safe_vln_dataset \
  --start-idx 0 --end-idx 100
```

Run deterministic evaluation without writing training shards:

```bash
python scripts/safe_vln_main.py evaluate \
  --start-idx 0 --end-idx 100 \
  --cost-limit 0
```

Safe trajectories are written under `eval_results/.../safe_trajectories`.
Transitions include contact body forces, orientation, and blocked-window diagnostics.
Complete training samples are written atomically as tar shards containing eight JPEG frames and one JSON transition. The existing measurement/video locations remain unchanged.

## Training

First collect data with a legacy server, then warm-start independent critics in the `navila` environment:

```bash
python scripts/safe_vln_main.py warmup-critics \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --dataset-dir outputs/safe_vln_dataset \
  --output-dir outputs/critic_warmup \
  --epochs 1
```

Start stochastic structured inference with the warm checkpoint, then collect fresh on-policy data from the `vlnce-isaac` environment:

```bash
python scripts/vlm_server.py \
  --model_path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --safe_checkpoint outputs/critic_warmup \
  --no-safe_deterministic

python scripts/safe_vln_main.py collect \
  --dataset-dir outputs/on_policy_v0 \
  --start-idx 0 --end-idx 100
```

Run constrained PPO back in the `navila` environment:

```bash
python scripts/safe_vln_main.py train \
  --model-path $HOME/NaVILA/checkpoints/navila-llama3-8b-8f \
  --checkpoint outputs/critic_warmup \
  --rollout-dir outputs/on_policy_v0 \
  --output-dir outputs/policy_v1 \
  --cost-limit 0.1 \
  --gamma 0.99 \
  --gae-lambda 0.95
```

Re-collect rollouts after every policy update. The trainer rejects samples lacking the structured `old_log_prob`, `reward_value`, or `cost_value`, preventing legacy data from being treated as on-policy PPO data.

Aggregate episode metrics:

```bash
python scripts/safe_vln_main.py summarize \
  --measurement-dir eval_results/go2_matterport_vision_loco_2024-09-25_23-22-02/measurements
```

The summary includes success/SPL, safe success, safe SPL, cumulative cost, collision rate, blocked rate, constraint satisfaction, zero-cost rate, and cost percentiles.

## Defaults

- non-foot contact threshold: `1.0 N`
- orientation limit: `0.8 rad`
- blocked window: `2.0 s` (100 steps at the current `0.02 s` control period)
- blocked displacement threshold: `0.10 m`
- reward discount / GAE lambda: `0.99 / 0.95`
- training/evaluation cost limits: `0.1 / 0.0`
- PPO clip: `0.1`
- Lagrange learning rate: `0.035`
- LoRA rank/alpha/dropout: `16 / 32 / 0.05`

Run the pure logic suite with the available Python 3.11 base environment:

```bash
/share/software/anaconda3/2024.02.01/bin/python -m pytest -q tests
```
