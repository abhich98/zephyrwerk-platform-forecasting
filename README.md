# Zephyrwerk Energy Analytics Platform

Production-grade data engineering platform ingesting live German electricity market data (SMARD / Bundesnetzagentur), transforming it with dbt, serving ML-powered predictions via a FastAPI REST API, and visualising results in a Streamlit dashboard — deployed on AWS.

Built as a portfolio project demonstrating end-to-end data platform engineering: from raw API ingestion to ML inference to cloud deployment.

---

## Architecture

**Stack:** Python · dbt Core · PostgreSQL · FastAPI · Streamlit · XGBoost · AWS (S3, RDS, ECS Fargate, Step Functions, EventBridge) · Docker · GitHub Actions

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│         SMARD API                    Open-Meteo API             │
│   (generation, consumption,       (wind, solar, temperature)    │
│    prices, neighbour prices)                                    │
└────────────────────┬────────────────────────┬───────────────────┘
                     │                        │
                     ▼                        ▼
              ingestion/smard_client.py   ingestion/weather_client.py
                     │                        │
                     └───────────┬────────────┘
                                 │ raw JSON → Parquet
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS S3 — RAW LAYER                           │
│  s3://zephyrwerk-data-lake/raw/smard/year=YYYY/month=MM/        │
│  s3://zephyrwerk-data-lake/raw/weather/year=YYYY/month=MM/      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ dbt Core (ECS Task)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              AWS RDS PostgreSQL — ANALYTICS LAYER               │
│  schema: staging   → stg_smard_generation, stg_smard_prices ... │
│  schema: analytics → fct_energy_generation, fct_market_prices   │
│                       fct_price_spreads, fct_ml_features ...    │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐     ┌────────────────────────────────────┐
│     ML MODELS        │     │     FastAPI — AWS ECS Fargate       │
│  XGBoost price +     │     │  GET  /energy/generation           │
│  generation forecast │     │  GET  /energy/prices               │
│  stored in S3        │     │  GET  /energy/summary              │
└──────────────────────┘     │  POST /predict/price               │
                             │  POST /predict/generation          │
                             └──────────────────┬─────────────────┘
                                                ▼
                             ┌──────────────────────────────────┐
                             │  Streamlit — AWS ECS Fargate     │
                             │  Historical Overview             │
                             │  Market Monitor                  │
                             │  Forecast Viewer                 │
                             └──────────────────────────────────┘

CI/CD:         GitHub Actions → Docker build → ECR push → ECS deploy
Orchestration: EventBridge → Step Functions → ECS Tasks (daily 06:00 UTC)
Monitoring:    AWS CloudWatch (logs + cost alerts)
```

---

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1 — Ingestion | SMARD + Open-Meteo clients, S3 raw layer, LocalStack | ✅ Complete · `v0.1.0` |
| 2 — EDA | Jupyter notebooks, energy mix analysis, findings | ✅ Complete |
| 3 — dbt | Staging + analytics models, dbt tests, first Dockerfile | 🔜 Not started |
| 4 — ML | XGBoost price + generation forecasting, model registry | 🔜 Not started |
| 5 — API | FastAPI service, all endpoints, pytest suite | 🔜 Not started |
| 6 — Dashboard | Streamlit multipage dashboard, Docker Compose | 🔜 Not started |
| 7 — AWS | Full cloud deployment, CI/CD, v1.0.0 release | 🔜 Not started |

---

## Key EDA Findings (Phase 2)

Seven years of German electricity data (2019–2025, ~2.6M hourly rows) across 22 SMARD signals and 4 weather locations.

### Energy Mix Evolution

- **Germany became a net electricity importer in 2023.** The nuclear phase-out removed ~8 GW of baseload; renewable additions (+4 GW) only partially offset it. Germany stabilised at ~3 GW continuous net import.
- **Renewable share grew from 43.6% to 60.2%**, but ~80% of the 2023 jump came from nuclear leaving the denominator, not new renewable capacity.
- **Solar is the only renewable source that grew consistently** (+76% over the window, 4.8 → 8.4 GW average). Wind onshore has declined two years running.
- **Coal's structural decline only began in 2023** — the 2022 gas crisis kept fossil generation elevated because coal was suddenly cheaper than gas.

### Price Dynamics

- **Three structurally distinct price regimes:** pre-crisis (mean €59/MWh), 2022 gas crisis (mean €216/MWh, peak €871), post-nuclear (mean €85/MWh with extremes of −€500 to +€936).
- **Negative prices grew from 1.6% (2021) to 6.5% (2025)** of hours — now a structural feature of the renewable-heavy grid, concentrated in spring/summer midday solar surplus hours.
- **Residual load is the primary price driver** (Pearson r = 0.831). The empirical supply curve shows three zones: negative prices at <15 GW, competitive pricing at 15–40 GW, and steep scarcity pricing above ~40 GW.
- **The duck curve is visible in German prices**: summer midday prices dip near zero or negative; winter morning prices peak as high consumption meets low solar and minimum wind.

### Consumption Patterns

- **Industrial demand destruction, not residential.** Total consumption fell ~5 GW from the 2021 peak (57.6 GW) to 2025 (53.0 GW). The weekday–Sunday load gap shrunk from 12.4 GW to 10.3 GW — a clear industrial signature.
- **Germany's highest-stress grid period is early January/February, not Christmas** — December consumption drops due to industrial shutdown.
- **Residual load volatility grew 56%** even as its mean shrank 22%, which is the structural cause of persistent European price volatility.

### Neighbour Price Spreads

- **Germany is geographically split:** structurally cheaper than Poland (+€29/MWh average spread) and France (+€19/MWh); more expensive than Alpine zones (Austria −€5.50, Switzerland −€5.23).
- **Switzerland and France carry the most predictive spread information** for next-hour DE/LU prices (lead-1h Pearson r = 0.538, 0.501), reflecting Alpine hydro storage and French nuclear as independent supply signals.
- **Danish spreads are weak predictors** despite high level-correlation — both markets are wind-coupled, so the spread collapses to noise.

> Full analysis, charts, and downstream recommendations: [`notebooks/eda/FINDINGS.md`](notebooks/eda/FINDINGS.md)

---

## Local Setup

### Prerequisites

- Python ≥ 3.11 managed via [`uv`](https://docs.astral.sh/uv/)
- Docker (for LocalStack S3 emulation)

### Install

```bash
git clone https://github.com/hasanerdin/zephyrwerk-platform.git
cd zephyrwerk-platform

uv sync --extra dev
```

### Configure environment

```bash
cp .env.example .env
# Edit .env — all required keys are documented in .env.example
```

Key variables:

| Variable | Local default | Purpose |
|---|---|---|
| `AWS_ENDPOINT_URL` | `http://localhost:4566` | Points boto3 at LocalStack; leave empty in production |
| `ZEPHYRWERK_AWS_BUCKET_NAME` | `zephyrwerk-data-lake` | S3 bucket |
| `ZEPHYRWERK_RDS_HOST` | `localhost` | PostgreSQL host |

### Run the ingestion pipeline

```bash
# Start LocalStack (S3 emulation)
docker run -d -p 4566:4566 localstack/localstack

# Historical backfill (one-time, from post-nuclear regime)
python orchestration/run_pipeline.py --start_date 2023-04-16 --end_date 2025-12-31

# Daily incremental run (yesterday only — omit dates for default)
python orchestration/run_pipeline.py
```

The pipeline is idempotent: re-running a date range skips files already present in S3.

### Run EDA notebooks

```bash
uv run jupyter lab notebooks/eda/
```

Notebooks in order: `00_data_audit` → `01_energy_mix_history` → `02_renewable_seasonality` → `03_consumption_patterns` → `04_price_dynamics` → `05_price_spreads`

### Run tests

```bash
uv run pytest
```

---

## Data Sources

- **SMARD** (Bundesnetzagentur) — 12 generation signals, total consumption, residual load, DE/LU day-ahead prices, and 8 neighbour-zone prices · CC BY 4.0
- **Open-Meteo** — Historical (ERA5 reanalysis) and forecast weather for 4 German regions co-located with Zephyrwerk's wind and solar assets · Free for non-commercial use

---

## Author

Hasan Erdin — Data Engineer & Applied Data Scientist, Munich
[GitHub](https://github.com/hasanerdin) · [LinkedIn](https://linkedin.com/in/hasanerdin)
