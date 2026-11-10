# CLAUDE.md: FraudSight MLOps build guide

This file tells Claude (Claude Code) how to build this project. Read **README.md** first: it is the design spec. This file is the build plan. **INFO.md** is the interview-prep companion; keep it consistent with what is actually built.

## Goal

End-to-end lifecycle for card-fraud detection on the public ULB credit card fraud dataset (284,807 transactions, about 0.17% fraud), sized for one person in 2–4 weeks: leakage-safe time-based split, a scikit-learn feature pipeline, class weights vs SMOTE compared, a logistic-regression baseline vs XGBoost, imbalance-aware metrics (PR-AUC, recall at 1% FPR and at fixed precision, cost-based threshold), SHAP explanations, MLflow tracking and registry (staging -> production aliases with a promotion gate), a FastAPI scoring service with input validation and latency logging, Evidently drift reports on a simulated shifted batch with a retrain trigger, Docker Compose (API + MLflow) and GitHub Actions CI (lint, tests, training smoke test).

ULB is the primary dataset because it is one small table with a time column and runs on a laptop and in CI. IEEE-CIS is a stretch goal.

## Tech stack

Python 3.11, pandas, scikit-learn, imbalanced-learn (SMOTE only), XGBoost, SHAP, MLflow (tracking server + registry with aliases), FastAPI + Pydantic + Uvicorn, Evidently, Docker Compose, pytest, ruff, GitHub Actions.

## Build order (phases)

1. Scaffold (`src/fraudsight/`, `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore` with data folders), Kaggle download script, schema checks, dedupe, time-based train/val/test split (about 60/20/20 by `Time`), EDA notebook on imbalance. CI workflow with ruff + unit tests from day one, including a no-overlap split test.
2. Feature pipeline (`log1p(Amount)`, hour of day, scaling) as a scikit-learn `Pipeline`; evaluation module (average precision, recall at 1% FPR, recall at fixed precision, bootstrap CIs); logistic regression baseline; MLflow tracking server and run logging.
3. XGBoost with early stopping on validation `aucpr`; imbalance comparison (none, `scale_pos_weight`, SMOTE inside an imbalanced-learn pipeline on the training fold only); cost-based threshold chosen on validation; SHAP global plot and example explanations logged to MLflow.
4. MLflow registry with `staging` and `production` aliases; `promote.py` gate (promote only if test PR-AUC beats the current production model on the same window); FastAPI service (`/score`, `/score/batch`, `/health`, `/model-info`, optional `?explain=true`) loading `models:/fraudsight-fraud@production`, Pydantic validation, JSON latency logging; `scripts/bench_latency.py` for p50/p95/p99.
5. Docker Compose (`mlflow` + `api`, health checks, `--serve-artifacts`); CI adds a training smoke test on synthetic data with the real schema (no dataset download in CI) and an API image build.
6. Shifted and control batch simulator; Evidently data drift report vs the training window (HTML + JSON in `docs/results/drift/`); `retrain_trigger.py` retrains when drift share passes `DRIFT_THRESHOLD` and sends the new model through the Phase 4 gate.
7. Fill the README results table from real runs, add screenshots (MLflow UI, Evidently report, API docs), how-it-works notes in `docs/`, demo GIF.

## Stretch goals

Only after Phase 7 is done and CI is green.

- Feast feature store (offline + online) for stateful features such as rolling counts and amount vs recent median
- Kubernetes deploy (kind or minikube) with a Horizontal Pod Autoscaler and a load test
- Fraud explainer: `claude-haiku-5-5` via the `anthropic` SDK turns top SHAP contributions into a one-sentence analyst reason; offline from the scoring path, prompt caching, per-call cost logging, small hand-checked eval set (budget is about $100 total)
- Score IronPay (`ironpay-ledgerapi`) transfers with a separate model trained on features IronPay actually has
- IEEE-CIS as a second dataset with per-card features
- Optuna tuning with time-aware validation; LightGBM comparison

## Working rules for Claude

- Build **one phase at a time**, in the order above. Each phase must run end to end before the next starts.
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

Only `README.md`, `INFO.md` and this file exist so far. Start at **Phase 1**.
