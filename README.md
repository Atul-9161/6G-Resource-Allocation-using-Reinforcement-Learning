# AI-Native 6G Resource Allocation — PPO Reinforcement Learning

A custom [Gymnasium](https://gymnasium.farama.org/) environment simulating dynamic
frequency-channel allocation in a 6G wireless cell with realistic wireless
metrics (SINR, co-channel interference, fairness), paired with a **PPO**
(Proximal Policy Optimization) actor-critic policy.

## Problem

Each scheduling frame, 4 mobile users (UEs) need to be assigned to one of 3
frequency sub-bands. Users assigned to the same channel interfere with one
another (co-channel interference), which lowers their SINR and achievable
throughput. Demands, per-channel noise, and per-user distance to the base
station all vary every frame. The agent must learn channel assignments that
maximize throughput **and** keep the allocation fair across users **and**
meet as many users' QoS demands as possible — three competing objectives.

## Project structure

```
tele6g_project/
├── app.py                  # Main entry point: train, plot, evaluate
├── requirements.txt
├── README.md
└── src/
    ├── environment.py      # Tele6GResourceEnv: SINR, interference, fairness, reward
    ├── model.py             # ActorCritic network (shared backbone, multi-head actor + critic)
    ├── gae.py                # Generalized Advantage Estimation
    ├── ppo_trainer.py       # PPO training loop (clipped objective)
    ├── baselines.py          # Random & greedy-heuristic baseline policies
    ├── evaluate.py            # Head-to-head policy comparison
    └── plotting.py            # Training-curve figure generation
```

## Wireless metrics modeled

- **SINR** (Signal-to-Interference-plus-Noise Ratio) — computed per user from
  transmit power, distance-based path loss, channel noise floor, and
  co-channel interference from other users sharing the same channel.
- **Shannon capacity** — per-user throughput via `C = B · log2(1 + SINR)`.
- **Co-channel interference** — users sharing a channel degrade each other's
  SINR (NOMA-style simplification); this is what makes naive "everyone picks
  the best channel" scheduling fail.
- **Jain's Fairness Index** — measures how evenly demand-satisfaction is
  spread across users (1.0 = perfectly fair, closer to 0 = starved users).
- **QoS satisfaction rate** — fraction of users whose achieved throughput met
  their demand.

## Algorithm: PPO

- **Actor-critic** network: shared backbone, one classification head per
  user (channel choice) plus a scalar value head.
- **GAE** (Generalized Advantage Estimation) for lower-variance advantage
  estimates.
- **Clipped surrogate objective** to prevent destructively large policy
  updates — the core PPO stability trick.
- **Entropy bonus**, linearly decayed over training, for early exploration.
- **Gradient clipping** for additional training stability.

## Baselines compared against

1. **Random assignment** — each user picks an independent random channel.
2. **Greedy lowest-noise** — every user grabs the least-noisy channel,
   deliberately causing congestion/interference — a realistic example of why
   naive heuristics fail in multi-user wireless scheduling.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

By default this runs **2000 training episodes** (30 frames each). To change
episode count, edit `NUM_EPISODES` at the top of `app.py` — 1000–5000 is a
reasonable range depending on how long you're willing to let it train.

## Output

- `tele6g_training_results.png` — two panels: (1) raw + smoothed PPO reward
  over training, (2) fairness index and QoS satisfaction rate over training.
- A printed comparison table at the end, e.g.:

```
Policy                          Reward    Fairness     QoS/4    SINR(dB)
PPO (learned)                    XXX.X       0.XXX      X.XX        XX.X
Random assignment                XXX.X       0.XXX      X.XX        XX.X
Greedy lowest-noise               XXX.X       0.XXX      X.XX        XX.X
```

## Possible extensions

- Swap PPO for a multi-agent formulation (one policy per user, or MADDPG).
- Add user mobility (distances evolve over time instead of resampling fresh
  each frame) for a truly sequential MDP rather than a per-frame context.
- Add more channels / users to test scalability.
