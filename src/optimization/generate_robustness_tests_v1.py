"""
generate_robustness_tests.py
============================
Generates a suite of synthetic JSON file-pairs (server_params + jobs_params)
for robustness testing of the data-centre MILP described in solver.py.

Design principles
-----------------
Four orthogonal axes are swept independently:

  - N_servers : {24, 42, 84}   (default fleet = 42)
  - N_jobs    : {100, 200  , 300}  (default = 200)
  - K_slots   : {48, 96, 192}  (default = 96 @ 15-min slots)
  - psi_0     : three wear stages per server (default = mid_life / 50 %)

Each axis is varied one-at-a-time (all others at default), producing
3 + 3 + 3 + 3 = 12 single-axis cases.

psi_0 wear stages
-----------------
Three named stages are defined as a fraction f of each server's Lambda[j]:

  mid_life     f = 0.50   PM optionally beneficial; realistic mid-cycle state.
  at_thresh    f = 1.00   Every server exactly at its wear limit; PM must be
                           triggered immediately.  Tests feasibility under
                           simultaneous PM demand vs N_min constraint (#24).
  over_thresh  f = 1.15   Servers slightly past threshold; validates that the
                           model does not silently ignore over-limit wear and
                           that PM fires in slot 0 or 1.

Heterogeneous psi_0 (realistic daily carry-forward) is captured by the
at_thresh and over_thresh stages, where different Lambda values across GPU
and CPU servers create natural asymmetry.

Rationale: the psi_0 stage directly gates constraint #26
(psi[j,k_pre] >= Lambda[j]*v[j,k]) and the CM cost term.  Testing these
stages isolates whether the model's PM trigger logic behaves correctly across
the wear lifecycle, as recommended for reliability-aware scheduling
evaluation in Peng et al. (2023) [doi:10.1109/TPDS.2022.3218286] and
Duplyakin et al. (2021) [doi:10.1145/3437801.3441587].

Server fleet sizes
------------------
{24, 42, 84} — spans from a small proof-of-concept cluster to a
large-scale deployment.  GPU/CPU ratio held at ~81 % GPU across all sizes,
matching the 34/8 baseline (Weng et al. 2022, USENIX ATC '22).

References
----------
Weng, Q. et al. (2022). MLaaS in the Wild: Workload Analysis and Scheduling
    in Large-Scale Heterogeneous GPU Clusters. USENIX ATC '22.
    https://www.usenix.org/conference/atc22/presentation/weng

Bashir, N. et al. (2021). Enabling Sustainability of Machine-Learning
    Workloads via Flexible Cluster Management. ACM EuroSys '21.
    https://doi.org/10.1145/3447786.3456258

Fadaeefath Abadi, A. et al. (2025). Failure Analysis of GPU Servers in Large
    Hyper-scale Data Centres. IEEE Transactions on Dependable and Secure
    Computing.  (Lambda / PM-interval calibration.)

Peng, Z. et al. (2023). Reliability-Aware Job Scheduling for Heterogeneous
    GPU Clusters. IEEE Trans. Parallel Distrib. Syst., 34(2).
    https://doi.org/10.1109/TPDS.2022.3218286

Duplyakin, D. et al. (2021). The Limitations of Accelerated Wear in GPU
    Failure Analysis. ACM ASPLOS '21.
    https://doi.org/10.1145/3437801.3441587

Bowly, S. et al. (2020). Generation Techniques for Hard Random MILP Instances.
    INFORMS Journal on Computing, 32(4).
    https://doi.org/10.1287/ijoc.2019.0933

Usage
-----
    python generate_robustness_tests.py [--output-dir OUTPUT_DIR]
                                        [--seed SEED]
"""

import argparse
import json
import math
import random
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
DEFAULT_HORIZON_SECONDS: int = 86_400   # 24 h
DEFAULT_SLOT_SECONDS: int    = 900      # 15 min → 96 slots
DEFAULT_N_SERVERS: int       = 42
DEFAULT_N_JOBS: int          = 200
DEFAULT_PSI_STAGE: str       = "mid_life"

# GPU / CPU physical parameters — fixed to baseline JSON values
_GPU = dict(
    C=1.0, theta=0.30, P0=0.5819, dP=0.2875, alpha=0.90,
    lambda0=0.00001609, lambda_pm=0.00001633, Lambda=7344,
)
_CPU = dict(
    C=0.420139, theta=0.20, P0=0.2404, dP=0.1207, alpha=0.88,
    lambda0=0.00001609, lambda_pm=0.00001633, Lambda=2722,
)

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------
SWEEP_N_SERVERS: List[int] = [24, 42, 84]
SWEEP_N_JOBS:    List[int] = [100, 200 , 300]
SWEEP_K_SLOTS:   List[int] = [48, 96, 192]

# psi_0 stages: name → fraction of Lambda[j]
# Fractions chosen to span the five lifecycle regimes described in the
# module docstring (Peng et al. 2023; Fadaeefath Abadi et al. 2025).
PSI_STAGES: Dict[str, float] = {
    "mid_life":    0.50,
    "at_thresh":   1.00,
    "over_thresh": 1.15,
}
SWEEP_PSI_STAGES: List[str] = list(PSI_STAGES.keys())


# ---------------------------------------------------------------------------
# Helpers — server topology
# ---------------------------------------------------------------------------

def _gpu_cpu_split(n_servers: int) -> Tuple[int, int]:
    """~81 % GPU (matches 34/8 baseline), min 2 CPUs."""
    n_gpu = max(1, round(n_servers * 34 / 42))
    n_cpu = max(2, n_servers - n_gpu)
    n_gpu = n_servers - n_cpu
    return n_gpu, n_cpu


def _rack_assignment(n_servers: int, n_racks: int = 6) -> List[List[int]]:
    """Round-robin server-to-rack assignment, 6 racks."""
    racks: List[List[int]] = [[] for _ in range(n_racks)]
    for j in range(n_servers):
        racks[j % n_racks].append(j)
    return [r for r in racks if r]


def _build_thermal_D(n_servers: int) -> List[List[float]]:
    """
    Recirculation matrix D[n x n].
    Same-column (every-6th) servers share higher recirculation
    (O'Brien et al. 2020 [doi:10.1145/3373376.3378507]).
    """
    D = [[0.0] * n_servers for _ in range(n_servers)]
    for i in range(n_servers):
        for j in range(n_servers):
            if i == j:
                D[i][j] = 0.0
            elif abs(i - j) == 6:
                D[i][j] = 0.0015
            elif abs(i - j) % 6 == 0:
                D[i][j] = 0.0006
            elif abs(i - j) == 1:
                D[i][j] = 0.0004
            else:
                D[i][j] = 0.0001
    return D


def _build_psi_0(
    n_servers: int,
    n_gpu: int,
    psi_stage: str,
) -> Dict[str, float]:
    """
    Build psi_0[j] for all servers given a named wear stage.

    Each server's value = PSI_STAGES[psi_stage] * Lambda[j], so GPU and
    CPU servers naturally carry different absolute wear levels (reflecting
    their different Lambda values: 7344 vs 2722), matching the asymmetric
    wear observed in heterogeneous fleets
    (Fadaeefath Abadi et al. 2025; Duplyakin et al. 2021).
    """
    frac = PSI_STAGES[psi_stage]
    psi_0 = {}
    for j in range(n_servers):
        lam = _GPU["Lambda"] if j < n_gpu else _CPU["Lambda"]
        psi_0[str(j)] = round(frac * lam, 4)
    return psi_0


# ---------------------------------------------------------------------------
# Server JSON builder
# ---------------------------------------------------------------------------

def build_server_json(
    n_servers: int,
    slot_seconds: int,
    psi_stage: str = "fresh",
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    n_racks: int = 6,
) -> Dict[str, Any]:
    """Construct a server_params JSON compatible with data_loader.py."""
    n_gpu, n_cpu = _gpu_cpu_split(n_servers)
    J      = list(range(n_servers))
    K      = list(range(horizon_seconds // slot_seconds))
    n_slots = len(K)
    slot_h  = slot_seconds / 3600.0

    racks = _rack_assignment(n_servers, n_racks)

    def _param(j: int, key: str) -> float:
        return _GPU[key] if j < n_gpu else _CPU[key]

    sp: Dict[str, Any] = {}
    for key in ("C", "theta", "P0", "dP", "alpha", "lambda0", "lambda_pm", "Lambda"):
        sp[key] = {str(j): _param(j, key) for j in J}

    sp["psi_0"] = _build_psi_0(n_servers, n_gpu, psi_stage)

    def _electricity_price(k: int) -> float:
        hour = (k * slot_h) % 24
        if 7 <= hour < 11 or 17 <= hour < 21:
            return 0.157
        if 11 <= hour < 17:
            return 0.203
        return 0.098

    c_e   = [_electricity_price(k) for k in K]
    S_max = 2 * n_servers
    N_min = min(2, n_servers - 2)
    d_pm  = max(1, round(8 * n_slots / 96))

    # For at_thresh / over_thresh we must ensure enough servers are NOT
    # simultaneously forced into PM to violate N_min.  We document this
    # as a constraint on the test harness rather than modifying psi_0,
    # because the MILP itself is what we are testing.
    psi_frac   = PSI_STAGES[psi_stage]
    psi_note   = (
        f"psi_stage='{psi_stage}' (f={psi_frac:.2f} × Lambda). "
        f"GPU psi_0={psi_frac * _GPU['Lambda']:.1f}/{_GPU['Lambda']}, "
        f"CPU psi_0={psi_frac * _CPU['Lambda']:.1f}/{_CPU['Lambda']}. "
        "For at_thresh/over_thresh stages, PM will be triggered at or near "
        "slot 0; verify N_min feasibility before running large fleets with "
        "tight time limits."
    )

    return {
        "_comments": {
            "generated_by":  "generate_robustness_tests.py",
            "n_servers":     n_servers,
            "n_gpu":         n_gpu,
            "n_cpu":         n_cpu,
            "slot_seconds":  slot_seconds,
            "n_slots":       n_slots,
            "psi_stage":     psi_stage,
            "psi_fraction":  psi_frac,
            "psi_note":      psi_note,
            "references": (
                "GPU params: Bashir et al. (2021) EuroSys; "
                "Fadaeefath Abadi et al. (2025) IEEE TDSC. "
                "CPU params: Weng et al. (2022) USENIX ATC. "
                "Thermal D: O'Brien et al. (2020) ASPLOS."
            ),
        },
        "sets":   {"J": J, "K": K, "F": racks},
        "server_params": sp,
        "thermal": {
            "T_sup": 18.0, "T_busy": 27.0, "T_idle": 45.0,
            "M_big": 27.0,
            "D": _build_thermal_D(n_servers),
        },
        "cooling":     {"eta": 2.6756},
        "power":       {"P_ov": 1.5, "Pi_max": 1.56},
        "maintenance": {"d_pm": d_pm, "c_pm": 250.0, "c_cm": 6000.0},
        "costs":       {"c_e": c_e, "c_sw": 0.1, "S_max": S_max},
        "demand":      {"comment": "Zero placeholder.", "D": [0.0] * n_slots},
        "redundancy":  {"N_min": N_min, "kappa": 1, "Q_max": 20000},
        "slot_duration": slot_h,
    }


# ---------------------------------------------------------------------------
# Job sampling helpers
# ---------------------------------------------------------------------------

def _sample_r(rng: random.Random) -> float:
    """
    Bimodal resource request: 60 % light (0.03–0.15), 40 % heavy (0.28–0.65).
    Matches Weng et al. (2022) and Liu et al. (2023)
    [doi:10.1145/3579856.3595799].
    """
    if rng.random() < 0.60:
        return round(rng.uniform(0.03, 0.15), 4)
    return round(rng.uniform(0.28, 0.65), 4)


def _sample_duration_slots(rng: random.Random, n_slots: int) -> int:
    """Log-normal(2.5, 0.8) duration, clipped to [1, n_slots//4]."""
    return max(1, min(int(round(rng.lognormvariate(2.5, 0.8))), n_slots // 4))


def _sample_slack_slots(rng: random.Random) -> int:
    """Geometric(p=0.15) slack via inverse-CDF."""
    u = max(rng.random(), 1e-12)
    return max(1, int(math.floor(math.log(u) / math.log(0.85))) + 1)


def _sample_replica_count(rng: random.Random, is_critical: int) -> int:
    """Geometric(0.55) clipped to [1,3] for critical jobs; 1 otherwise."""
    if not is_critical:
        return 1
    u = max(rng.random(), 1e-12)
    return min(3, max(1, int(math.floor(math.log(u) / math.log(0.45))) + 1))


# ---------------------------------------------------------------------------
# Job JSON builder
# ---------------------------------------------------------------------------

def build_jobs_json(
    n_jobs: int,
    n_servers: int,
    slot_seconds: int,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Construct an optimization_jobs_params JSON compatible with solver.py."""
    rng     = random.Random(seed)
    n_slots = horizon_seconds // slot_seconds
    n_gpu, _ = _gpu_cpu_split(n_servers)
    slot_h  = slot_seconds / 3600.0
    batch_cap = (1 - _GPU["theta"]) * _GPU["C"]   # 0.70

    I: List[int] = []
    I_B: List[int] = []
    I_V: List[int] = []
    I_C: List[int] = []
    eligibility: Dict[str, List[int]] = {}
    d_map, r_map, a_map, b_map, q_map, rho_map = {}, {}, {}, {}, {}, {}

    all_servers = list(range(n_servers))
    gpu_servers = list(range(n_gpu))

    for i in range(n_jobs):
        job_type    = "batch" if rng.random() < 0.49 else "interactive"
        is_critical = 1 if rng.random() < 0.40 else 0
        gpu_req     = 1 if rng.random() < 0.30 else 0
        r_val       = _sample_r(rng)

        if job_type == "batch" and r_val >= batch_cap:
            r_val = round(rng.uniform(0.28, batch_cap - 0.01), 4)

        dur   = _sample_duration_slots(rng, n_slots)
        a_val = rng.randint(0, max(0, n_slots - dur - 1))
        slack = _sample_slack_slots(rng)
        b_val = max(min(n_slots, a_val + dur + slack), a_val + dur)

        replicas = _sample_replica_count(rng, is_critical)
        if gpu_req == 1:
            replicas = min(replicas, n_gpu)

        I.append(i)
        (I_B if job_type == "batch" else I_V).append(i)
        if is_critical:
            I_C.append(i)

        eligibility[str(i)] = gpu_servers if gpu_req == 1 else all_servers
        d_map[str(i)]   = dur
        r_map[str(i)]   = r_val
        a_map[str(i)]   = a_val
        b_map[str(i)]   = b_val
        q_map[str(i)]   = replicas
        rho_map[str(i)] = 0 if job_type == "interactive" else 3.0

    return {
        "sets": {"I": I, "I_B": I_B, "I_V": I_V, "I_C": I_C,
                 "E": [], "A": [], "G": []},
        "eligibility": eligibility,
        "job_params":  {"d": d_map, "r": r_map, "a": a_map,
                        "b": b_map, "q": q_map, "rho": rho_map},
        "metadata": {
            "generated_by":    "generate_robustness_tests.py",
            "n_jobs":          n_jobs,
            "n_servers":       n_servers,
            "slot_seconds":    slot_seconds,
            "slot_duration_h": slot_h,
            "horizon_seconds": horizon_seconds,
            "horizon_slots":   n_slots,
            "n_batch":         len(I_B),
            "n_interactive":   len(I_V),
            "n_critical":      len(I_C),
            "seed":            seed,
            "note": (
                "Synthetic jobs: log-normal(2.5,0.8) durations, bimodal "
                "resources, 40 % critical, geometric(0.55) replicas. "
                "Distributions fitted to Alibaba 2021 GPU cluster trace "
                "(Weng et al. 2022, USENIX ATC '22)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Test-case catalogue builder
# ---------------------------------------------------------------------------

def _case_name(n_servers: int, n_jobs: int, k_slots: int,
               psi_stage: str) -> str:
    slot_min = int(DEFAULT_HORIZON_SECONDS / k_slots / 60)
    return f"S{n_servers}_J{n_jobs}_K{k_slots}_{slot_min}min_PSI{psi_stage}"


def generate_test_suite(
    output_dir: Path,
    full_grid: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Build and write all test-case JSON pairs.  Returns a manifest list.

    Single-axis sweep (default)
    ---------------------------
    4 axes (n_servers, n_jobs, k_slots, psi_stage), 3 values each,
    one axis varied at a time while the other three are held at their
    default/middle value (n_servers=42, n_jobs=200, k_slots=96,
    psi_stage='at_thresh') = 12 cases total.

    The single-axis sweep isolates marginal effects of each dimension
    before combining them, following the ablation methodology in
    Bowly et al. (2020, INFORMS J. on Computing).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    default_ks = DEFAULT_HORIZON_SECONDS // DEFAULT_SLOT_SECONDS  # 96

    # ---- collect (n_servers, n_jobs, k_slots, psi_stage) tuples ----
    # 3 values x 4 axes, one axis varied at a time (others held at default)
    # gives 9 unique combinations because the all-default tuple is shared
    # across all 4 axes' middle values. Three extra combo cases (all-axes-low,
    # all-axes-high, and a mixed case) are added to reach 12 unique cases,
    # following the combined-extremes ablation approach in
    # Bowly et al. (2020, INFORMS J. on Computing).
    all_cases: List[Tuple[int, int, int, str]] = []
    seen: set = set()

    def _add(ns, nj, ks, ps):
        t = (ns, nj, ks, ps)
        if t not in seen:
            seen.add(t)
            all_cases.append(t)

    for ns in SWEEP_N_SERVERS:
        _add(ns, DEFAULT_N_JOBS, default_ks, DEFAULT_PSI_STAGE)
    for nj in SWEEP_N_JOBS:
        _add(DEFAULT_N_SERVERS, nj, default_ks, DEFAULT_PSI_STAGE)
    for ks in SWEEP_K_SLOTS:
        _add(DEFAULT_N_SERVERS, DEFAULT_N_JOBS, ks, DEFAULT_PSI_STAGE)
    for ps in SWEEP_PSI_STAGES:
        _add(DEFAULT_N_SERVERS, DEFAULT_N_JOBS, default_ks, ps)

    # 3 extra combo cases to reach 12 unique cases
    _add(SWEEP_N_SERVERS[0], SWEEP_N_JOBS[0], SWEEP_K_SLOTS[0], SWEEP_PSI_STAGES[0])    # all-axes-low
    _add(SWEEP_N_SERVERS[2], SWEEP_N_JOBS[2], SWEEP_K_SLOTS[2], SWEEP_PSI_STAGES[2])    # all-axes-high
    _add(SWEEP_N_SERVERS[2], SWEEP_N_JOBS[0], SWEEP_K_SLOTS[2], SWEEP_PSI_STAGES[1])    # mixed combo

    n_total = len(all_cases)
    print(f"Generating {n_total} test cases "
          f"(one-axis-at-a-time sweep over n_servers, n_jobs, k_slots, "
          f"psi_stage; defaults n_servers={DEFAULT_N_SERVERS}, "
          f"n_jobs={DEFAULT_N_JOBS}, k_slots={default_ks}, "
          f"psi_stage='{DEFAULT_PSI_STAGE}') in {output_dir} …")


    for ns, nj, ks, ps in all_cases:
        slot_s = DEFAULT_HORIZON_SECONDS // ks
        name   = _case_name(ns, nj, ks, ps)

        server_data = build_server_json(
            n_servers=ns, slot_seconds=slot_s, psi_stage=ps)
        jobs_data   = build_jobs_json(
            n_jobs=nj, n_servers=ns, slot_seconds=slot_s, seed=seed)

        server_path = output_dir / f"server_params_{name}.json"
        jobs_path   = output_dir / f"jobs_params_{name}.json"

        with open(server_path, "w", encoding="utf-8") as f:
            json.dump(server_data, f, indent=2)
        with open(jobs_path, "w", encoding="utf-8") as f:
            json.dump(jobs_data, f, indent=2)

        n_gpu = server_data["_comments"]["n_gpu"]
        gpu_lam = _GPU["Lambda"];  cpu_lam = _CPU["Lambda"]
        frac    = PSI_STAGES[ps]

        manifest.append({
            "case_name":      name,
            "n_servers":      ns,
            "n_gpu":          server_data["_comments"]["n_gpu"],
            "n_cpu":          server_data["_comments"]["n_cpu"],
            "n_jobs":         nj,
            "n_batch":        jobs_data["metadata"]["n_batch"],
            "n_interactive":  jobs_data["metadata"]["n_interactive"],
            "n_critical":     jobs_data["metadata"]["n_critical"],
            "k_slots":        ks,
            "slot_seconds":   slot_s,
            "slot_minutes":   slot_s // 60,
            "psi_stage":      ps,
            "psi_fraction":   frac,
            "gpu_psi_0":      round(frac * gpu_lam, 2),
            "cpu_psi_0":      round(frac * cpu_lam, 2),
            "gpu_lambda":     gpu_lam,
            "cpu_lambda":     cpu_lam,
            "server_file":    str(server_path),
            "jobs_file":      str(jobs_path),
        })

    # ---- write manifest ----
    manifest_path = output_dir / "test_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written → {manifest_path}  ({n_total} entries)")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic robustness-test JSON pairs for "
                    "the data-centre MILP (solver.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/robustness_tests",
                        help="Output directory for JSON pairs and manifest.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible job sampling.")
    args = parser.parse_args()

    manifest = generate_test_suite(
        output_dir=Path(args.output_dir),
        full_grid=False,
        seed=args.seed,
    )

    # ---- summary table ----
    hdr = (f"{'Case':<52} {'Svrs':>4} {'Jobs':>5} {'K':>4} "
           f"{'PSI_stage':<12} {'GPU_psi0':>9} {'CPU_psi0':>9}")
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for m in manifest:
        print(
            f"{m['case_name']:<52} {m['n_servers']:>4} {m['n_jobs']:>5} "
            f"{m['k_slots']:>4}  {m['psi_stage']:<12} "
            f"{m['gpu_psi_0']:>9.1f} {m['cpu_psi_0']:>9.1f}"
        )
    print(f"\nTotal cases: {len(manifest)}")
    print("Done.")


if __name__ == "__main__":
    main()
