# engagement-autonomy

2D orbital-plane toy battlespace for:
- Simulation (two-body gravity, impulsive Δv burns)
- Tracking (EKF per target, optional Hungarian data association)
- Engagement management baseline (Hungarian assignment using hand-crafted costs)
- AutoML/NAS hooks (scaffold only)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/run_episode.py --config configs/episode.yaml
```

Outputs an episode JSONL under `runs/`.

## What’s implemented now

- Truth sim for agents + targets in 2D orbital plane.
- Simple sensor: noisy range/bearing from origin (can be extended to relative sensors).
- EKF for targets with analytic measurement Jacobian.
- Optional track association via Hungarian + Mahalanobis gating (scaffolded).
- Engagement assignment via Hungarian over (agent, track) pairs with a baseline cost.
