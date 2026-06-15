import streamlit as st

st.title("Workload Prediction and Scheduling Optimization for Energy-Efficient, Reliable, and Service-Aware Data Centers")
st.write("Welcome")

st.link_button(
    "Go to Dashboard",
    "https://dashboard-workload-optimization-and-scheduling.streamlit.app"
)

st.divider()

# ---------------- EXECUTIVE SUMMARY ----------------
st.header("Executive Summary")

st.markdown("""
Modern data centers must continuously balance three competing objectives:
**high performance**, **energy efficiency**, and **service reliability**.
Traditional scheduling approaches often react to workload demand as it arrives,
which can lead to inefficient resource utilization, higher operating costs,
and increased risk of service degradation.

This capstone project presents a **forecast-driven workload scheduling framework**
designed to support intelligent, proactive decision-making in data center operations.
Using **12,374 real production jobs** from Alibaba's GPU cluster trace, the system
combines machine learning forecasting with mathematical optimization to anticipate
future workload demand and allocate computing resources more efficiently.

The forecasting component predicts future job arrivals, resource requirements,
and execution durations. These forecasts are then used as inputs to a
**Mixed-Integer Linear Programming (MILP)** model that schedules workloads across
**42 servers** over a **24-hour planning horizon**.

The optimization framework simultaneously considers energy consumption,
cooling costs, maintenance requirements, reliability constraints, and
service-level performance. By evaluating these objectives together,
the model identifies scheduling decisions that are difficult to achieve
using conventional reactive approaches.

Results demonstrate that predictive analytics and mathematical optimization
can be combined to improve workload management, increase operational efficiency,
and support more sustainable and reliable data center operations.
""")

# ---------------- ARCHITECTURE ----------------
st.header("Architecture Diagram")
st.image("docs/architecture_diagram.png")

# ---------------- FORECASTING ----------------
st.header("Forecasting Module")
st.markdown("""
The forecasting module generates a synthetic 24-hour job workload using a three-stage generative
ensemble pipeline (`ensemble_model.py`). The cleaned Alibaba GPU cluster trace is split 80/20
temporally: the first 80% of rows by arrival time form the training set; the last 20% form the test set.

---

**Stage 0 — Synthetic Workload Generation**

Future jobs are unknown, so the pipeline generates synthetic job descriptors by non-parametric
empirical sampling from the most recent 5,000 historical rows:

- **Interarrival times** are drawn from the historical distribution, clipped to the 1st–99th
  percentile to suppress extremes.
- **Job type** (`interactive` / `batch`) is sampled from the recent empirical distribution.
- **Role** and **app_name** are sampled conditional on job type.
- **GPU request** is derived deterministically: `0` if `role == "cn"`, `1` otherwise.

Generation continues until 86,400 seconds (24 hours) of simulated time elapse or 10,000 jobs
are produced, whichever comes first.

---

**Stage 1 — CPU & Memory Estimation (Classification)**

Two independent classifiers predict discrete CPU and memory request classes for each synthetic job.

*Features used:*
- Categorical: `role`, `app_name`, `job_type` (one-hot encoded)
- Temporal: `time_of_day_seconds`, `hour`, `minute`, `second`, cyclical `hour_sin` / `hour_cos`
- Context: `interarrival_seconds`, `gpu_request`
- Lag features (lag-1, lag-2, lag-5) and rolling mean (window=5) for `cpu_request`,
  `memory_request`, `duration_seconds`, and `interarrival_seconds`

*Base models:*
- **XGBoost** (`n_estimators=150`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`)
- **Random Forest** (`n_estimators=200`, `max_depth=8`, `class_weight="balanced"`)

Both are trained with `TimeSeriesSplit(5)` to generate out-of-fold (OOF) predictions on the
training set without leakage. Three meta-learner strategies are then evaluated:

- **SoftVoting** — averages class probabilities from all base models
- **Stack-LR** — Logistic Regression (`C=1.0`) trained on TRAIN OOF predictions only
- **Stack-XGB** — XGBoost classifier (`n_estimators=50`, `max_depth=2`) trained on TRAIN OOF predictions only

Best strategy per target is selected by test macro-F1:
- CPU → **SoftVoting** (macro-F1: **0.842**, p = 4×10⁻⁶)
- Memory → **Stack-XGB** (macro-F1: **0.630**, p = 3×10⁻⁷)

---

**Stage 2 — Duration Estimation (Regression)**

Duration is predicted in seconds using Stage 1 **predicted** CPU and memory values — not actual
ones — so that training and inference see the same input distribution (train-serving consistency).
The same lag and rolling features are used, extended with predicted `cpu_request` and `memory_request`.

*Base models:*
- **XGBoost_log** — XGBoost Regressor on `log1p(duration)` (`n_estimators=300`, `max_depth=4`, `learning_rate=0.04`)
- **Random Forest** — Random Forest Regressor (`n_estimators=200`, `max_depth=10`) on raw duration
- **Ridge_log** — Ridge Regression (`alpha=1.0`) on `log1p(duration)`

Log-transformed models apply Duan (1983) smearing correction at evaluation. For OOF meta-features,
only `expm1` is applied (no Duan) to keep training and inference meta-feature scales consistent.

Three ensemble strategies, all trained **exclusively on TRAIN OOF predictions** (no test leakage):

- **WeightedAvg** — inverse-RMSLE weighted average; weights from TRAIN OOF RMSLE
- **Stack-Ridge** — Ridge meta-learner trained on TRAIN OOF predictions
- **Stack-XGB_dur** — XGBoost regressor (`n_estimators=50`, `max_depth=2`) trained on
  `log1p(TRAIN OOF)`, inference with `expm1`

Best strategy selected by test RMSLE:
- Duration → **Stack-XGB_dur** (RMSLE: **1.155**, p = 0.0069)

---

**Autoregressive Buffer**

After each synthetic job is generated, its predicted CPU, memory, duration, and interarrival values
are appended to the rolling history (clipped to the 1st–99th percentile) so that lag features remain
coherent across the 24-hour rollout and prediction errors do not compound unboundedly.

---

**Critical Job Classification (Optimizer Input)**

After forecasting, jobs are flagged as critical (`is_critical = 1`) if any of the following apply:
- `gpu_request == 1`
- `cpu_request` ≥ 90th percentile of the forecasted batch
- `memory_request` ≥ 90th percentile of the forecasted batch

Critical jobs receive `replica_count = 2`; all others receive `replica_count = 1`. This feeds
directly into the MILP optimizer's redundancy and rack-diversity constraints.

---

**Statistical Validation**

All ensemble strategies are compared against the single-model XGBoost baseline using the
Wilcoxon signed-rank test (two-sided, α = 0.05). All three targets yield p < 0.05, confirming
that the ensemble improves aggregate macro-F1 (via minority-class recall) and RMSLE beyond
what is expected by chance under class imbalance.
""")

# ---------------- OPTIMIZATION ----------------
st.header("Optimization Module")
st.markdown("""
The MILP model is the decision-making base of the project. It focuses on scheduling jobs across physical servers over 
96 time slots of 15 minutes each (one full day), and it handles five operational dimensions simultaneously:

The model simultaneously optimizes five operational dimensions:

- **Energy Efficiency:** Minimizes operational costs and Power Usage
  Effectiveness (PUE) by reducing energy waste from cooling and
  infrastructure overhead.

- **Reliability and Maintenance:** Schedules preventive and corrective
  maintenance activities while enforcing redundancy requirements for
  critical workloads.

- **Thermal Management:** Maintains server inlet temperatures within
  ASHRAE A1 operating limits to ensure safe and efficient operation.

- **Workload Prioritization:** Differentiates between batch and
  interactive jobs, reserving capacity for latency-sensitive workloads.

- **Server State Management:** Minimizes unnecessary server power
  transitions and supports affinity and anti-affinity scheduling
  requirements.

The model produces an optimized daily schedule that specifies workload
placement, server activation states, and maintenance windows while
satisfying capacity, reliability, and operational constraints.
""")

# ---------------- LINKS ----------------
st.link_button(
    "💻 View GitHub Repository",
    "https://github.com/GasparFG/Workload_Prediction_and_Scheduling_Optimization_for_Energy_Efficient_Reliable_and_Service-Aware_DC"
)

# ---------------- TEAM ----------------
st.header("Team Members")

st.markdown("""
### Ana Carolina Quiroz Vazquez
ana.quiroz2010@myunfc.ca

### Gaspar Franco Garcia
gaspar.franco6692@myunf.ca

### Maria Jose Agualimpia Martinez
maria.agualimpia7916@myunfc.ca

### Monica Giselle Mendoza Mendoza
monica.mendoza5170@myunfc.ca
""")
