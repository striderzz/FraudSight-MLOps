# CLAUDE.md: Precog build guide

This file tells Claude (Claude Code) how to build this project. Read **README.md** first: it is the design spec. This file is the build plan.

## Goal

End-to-end MLOps for real-time credit card fraud detection (ULB dataset): Feast features, XGBoost/LightGBM with cost-based thresholds and SHAP, MLflow registry, FastAPI scoring on Kubernetes with canary rollouts, Evidently drift monitoring and automated retraining.

## Tech stack

Python 3.11, pandas, scikit-learn, XGBoost, LightGBM, imbalanced-learn, Optuna, SHAP, Feast, MLflow, FastAPI, Kafka (transaction replayer), Docker, kind/minikube + Kubernetes manifests or Helm, Evidently, Prometheus + Grafana, GitHub Actions.

## Build order (phases)

1. Scaffold + data download script; EDA notebook focused on class imbalance (about 0.17% fraud).
2. Baselines (logistic regression, random forest) evaluated with PR-AUC, recall at fixed precision and a cost matrix, using time-aware splits.
3. XGBoost/LightGBM with Optuna; imbalance strategies compared; SHAP explanations; everything tracked in MLflow.
4. Feast feature repo (velocity and rolling-amount features) with offline and online stores; training and serving share the same feature definitions.
5. FastAPI scoring service loading the MLflow production model; latency target documented; Dockerfile + tests.
6. Kafka transaction replayer, plus a consumer that scores events in real time.
7. Kubernetes deployment (kind/minikube) with HPA; canary rollout between two model versions.
8. Evidently drift reports + Prometheus metrics + Grafana dashboard; alert on drift.
9. GitHub Actions retraining workflow with model gating (only promote if it beats the current model).
10. Load test (Locust/k6) with p95 latency + throughput; results in docs/results and the README.

## Working rules for Claude

- Build **one phase at a time**, in the order below. Each phase must run end to end before the next starts.
- After each phase, update the **Roadmap checklist in README.md** and commit with a clear message.
- **Never invent results.** Every number in README.md (accuracy, latency, throughput, cost) must come from a run in this repo. Put benchmark output in `docs/results/`.
- Keep secrets out of git: use `.env` (gitignored) plus a committed `.env.example`.
- Never commit raw datasets or large model files. Write download scripts and add the data folders to `.gitignore`.
- Python: 3.11+, `uv` or `pip-tools` for dependencies, `ruff` + `black` for formatting, type hints, `pytest` for tests.
- Every service gets a `Dockerfile`. Local development runs with `docker compose up`.
- Add a GitHub Actions workflow early (lint + tests) and keep it green.
- Prefer small, readable modules over clever abstractions. Add docstrings where intent is not obvious.
- **Screenshots:** when a phase has something visible (UI, dashboard, Grafana, CLI output, test report), capture real screenshots into `docs/screenshots/` and embed them in README.md. Never mock them up.
- When a phase is done, add a short "How it works" note or diagram to `docs/`. This is what interviewers will read.

## Definition of done (whole project)

- `docker compose up` (or the documented command) brings up a working demo in one step
- README has a real architecture diagram, setup steps, screenshots or GIFs, and measured results
- CI is green, tests cover the core logic
- A 2-3 minute demo video or GIF is linked in the README

## Status

Only `README.md` and this file exist so far. Start at **Phase 1**.
