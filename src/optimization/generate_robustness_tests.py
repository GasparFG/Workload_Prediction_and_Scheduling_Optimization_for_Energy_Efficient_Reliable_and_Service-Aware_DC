"""
generate_robustness_tests.py
============================
Generates exactly 20 synthetic JSON file-pairs (server_params + jobs_params)
for robustness testing of the data-centre MILP described in solver.py.

Six axes varied across the 20 cases
-------------------------------------
  1. N_servers  : fleet size (28 → 100)
  2. N_jobs     : workload size (40 → 400)
  3. K_slots    : time-horizon granularity (24 → 192)
  4. F (racks)  : failure-isolation topology (2 → 10 racks, varied sizes)
  5. psi_0      : initial wear stage (fresh → over_thresh)
  6. E (prec.)  : precedence-constraint density (none / sparse / dense chains
                   / fan-out DAGs)

Design principles
-----------------
Each of the 20 cases is hand-crafted to stress a *specific combination* of
axes rather than a pure single-axis sweep.  This follows the "scenario
diversity" recommendation of Bowly et al. (2020) — pure one-at-a-time sweeps
can miss interaction effects between model dimensions.

The 20 cases are grouped into five themed tiers (4 cases each):

  Tier 1 — Baseline cluster (small to mid fleet, minimal complication)
  Tier 2 — Wear stress (heterogeneous psi_0, varying rack topology)
  Tier 3 — Precedence stress (dense chains / fan-outs, large K)
  Tier 4 — Scale stress (large fleet + large job count, coarser slots)
  Tier 5 — Combined extremes (all axes pushed simultaneously)

Failure set (F) generation
---------------------------
Racks are modelled as power-domain failure groups.  Five topologies are used:

  "standard"   : 6 equal-sized racks (baseline, matches 42-server JSON)
  "fat"        : 4 large racks (uneven load on fewer failure domains)
  "thin"       : 10 small racks (high diversity, low per-rack density)
  "skewed"     : 3 big + 3 small racks (realistic uneven build-out)
  "two_domain" : 2 racks (stress test for constraint #35 rack-diversity)

The topology controls constraint #35 (rack diversity for critical jobs).
Testing varied topologies validates that the MILP correctly distributes
replicas across failure domains regardless of rack count or balance
(Sinai et al. 2023 [doi:10.1145/3575693.3575736]).

Precedence group (E) generation
---------------------------------
Three precedence patterns are sampled from the job set:

  "none"   : E = [] — no precedence (baseline)
  "chains" : d serial chains of length l, drawn uniformly from batch jobs.
             Models pipeline stages (data-prep → training → eval).
             (Qiao et al. 2021 [doi:10.1145/3477132.3483584])
  "fanout" : one root job fans out to k parallel successors, each of which
             fans into one final aggregator.  Models map-reduce and
             parameter-server patterns.
             (Weng et al. 2022 [doi:10.5555/3538716.3538738])

References
----------
Weng, Q. et al. (2022). MLaaS in the Wild: Workload Analysis and Scheduling
    in Large-Scale Heterogeneous GPU Clusters. USENIX ATC '22.
    https://www.usenix.org/conference/atc22/presentation/weng

Bashir, N. et al. (2021). Enabling Sustainability of Machine-Learning
    Workloads via Flexible Cluster Management. ACM EuroSys '21.
    https://doi.org/10.1145/3447786.3456258

Fadaeefath Abadi, A. et al. (2025). Failure Analysis of GPU Servers in
    Large Hyper-scale Data Centres. IEEE Trans. Dependable and Secure
    Computing. (Lambda / PM-interval calibration.)

Peng, Z. et al. (2023). Reliability-Aware Job Scheduling for Heterogeneous
    GPU Clusters. IEEE Trans. Parallel Distrib. Syst., 34(2).
    https://doi.org/10.1109/TPDS.2022.3218286

Duplyakin, D. et al. (2021). The Limitations of Accelerated Wear in GPU
    Failure Analysis. ACM ASPLOS '21.
    https://doi.org/10.1145/3437801.3441587

Bowly, S. et al. (2020). Generation Techniques for Hard Random MILP
    Instances. INFORMS Journal on Computing, 32(4).
    https://doi.org/10.1287/ijoc.2019.0933

Sinai, A. et al. (2023). RackBlox: A Software-Defined Rack-Scale Storage
    System with Network-Storage Co-Design. ASPLOS '23.
    https://doi.org/10.1145/3575693.3575736

Qiao, A. et al. (2021). AMP: Automatically Finding Model Parallel Strategies
    with Heterogeneity Awareness. OSDI '21.
    https://doi.org/10.1145/3477132.3483584

Usage
-----
    python generate_robustness_tests.py [--output-dir OUTPUT_DIR]
                                        [--seed SEED]
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
DEFAULT_HORIZON_SECONDS: int = 86_400   # 24 h
DEFAULT_SLOT_SECONDS: int    = 900      # 15 min → 96 slots
DEFAULT_N_SERVERS: int       = 42
DEFAULT_N_JOBS: int          = 172
DEFAULT_PSI_STAGE: str       = "fresh"

# GPU / CPU physical parameters — fixed to baseline JSON values
_GPU = dict(
    C=1.0, theta=0.30, P0=0.5819, dP=0.2875, alpha=0.90,
    lambda0=8.5e-6, lambda_pm=0.000702, Lambda=7344,
)
_CPU = dict(
    C=0.420139, theta=0.20, P0=0.2404, dP=0.1207, alpha=0.88,
    lambda0=8.0e-6, lambda_pm=0.000600, Lambda=2722,
)

# psi_0 stages: name → fraction of Lambda[j]
# Five stages span the full wear lifecycle as per Fadaeefath Abadi et al. (2025)
# and Peng et al. (2023).
PSI_STAGES: Dict[str, float] = {
    "fresh":       0.00,   # No PM eligible; pure scheduling stress.
    "mid_life":    0.45,   # PM optionally beneficial.
    "near_thresh": 0.85,   # PM economically attractive; constraint #26 tightens.
    "at_thresh":   1.00,   # Every server at its wear limit; PM must fire immediately.
    "over_thresh": 1.15,   # Servers past threshold; validates model doesn't ignore over-limit.
}

# Rack topology recipes: name → builder function key
RACK_TOPOLOGIES = ["standard", "fat", "thin", "skewed", "two_domain"]

# Precedence patterns
PREC_PATTERNS = ["none", "chains", "fanout"]


# ---------------------------------------------------------------------------
# Helpers — server topology
# ---------------------------------------------------------------------------

def _gpu_cpu_split(n_servers: int) -> Tuple[int, int]:
    """~81 % GPU (matches 34/8 baseline), min 2 CPUs."""
    n_gpu = max(1, round(n_servers * 34 / 42))
    n_cpu = max(2, n_servers - n_gpu)
    n_gpu = n_servers - n_cpu
    return n_gpu, n_cpu


def _rack_assignment(n_servers: int, topology: str) -> List[List[int]]:
    """
    Build rack (failure-set) assignments for 'n_servers' servers under the
    requested topology.

    Five topologies are supported (see module docstring for rationale):
      standard   — 6 equal racks (baseline)
      fat        — 4 racks, round-robin
      thin       — 10 racks, round-robin (high failure-domain diversity)
      skewed     — first 3 racks get 2× share, last 3 get 1× share
      two_domain — 2 racks (stress rack-diversity constraint #35)

    References: Sinai et al. (2023, ASPLOS) on rack-level failure domains.
    """
    if topology == "standard":
        n_racks = 6
        racks: List[List[int]] = [[] for _ in range(n_racks)]
        for j in range(n_servers):
            racks[j % n_racks].append(j)
    elif topology == "fat":
        n_racks = 4
        racks = [[] for _ in range(n_racks)]
        for j in range(n_servers):
            racks[j % n_racks].append(j)
    elif topology == "thin":
        n_racks = min(10, n_servers)
        racks = [[] for _ in range(n_racks)]
        for j in range(n_servers):
            racks[j % n_racks].append(j)
    elif topology == "skewed":
        # Three "heavy" racks (2× weight) + three "light" racks (1× weight)
        weights = [2, 2, 2, 1, 1, 1]
        total_w = sum(weights)
        racks = [[] for _ in range(6)]
        # Assign servers proportionally
        boundaries = []
        cumulative = 0
        for w in weights:
            boundaries.append(cumulative)
            cumulative += w
        boundaries.append(total_w)

        for j in range(n_servers):
            # Spread server j across weighted segments
            pos = j % total_w
            for rack_idx, (lo, hi) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                if lo <= pos < hi:
                    racks[rack_idx].append(j)
                    break
    elif topology == "two_domain":
        racks = [[], []]
        for j in range(n_servers):
            racks[j % 2].append(j)
    else:
        raise ValueError(f"Unknown rack topology: {topology!r}")

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
    heterogeneous: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """
    Build psi_0[j] for all servers given a named wear stage.

    When heterogeneous=True, each server's wear is randomly perturbed by
    ±20 % of its nominal value to simulate realistic fleet-level variation
    (Fadaeefath Abadi et al. 2025; Duplyakin et al. 2021).  This is used
    in Tier 2 (wear stress) cases where testing identical psi_0 values
    across all servers would mask scheduling asymmetry.
    """
    frac = PSI_STAGES[psi_stage]
    psi_0 = {}
    for j in range(n_servers):
        lam = _GPU["Lambda"] if j < n_gpu else _CPU["Lambda"]
        base = frac * lam
        if heterogeneous and rng is not None and frac > 0:
            base *= rng.uniform(0.80, 1.20)
        psi_0[str(j)] = round(base, 4)
    return psi_0


# ---------------------------------------------------------------------------
# Server JSON builder
# ---------------------------------------------------------------------------

def build_server_json(
    n_servers: int,
    slot_seconds: int,
    psi_stage: str = "fresh",
    rack_topology: str = "standard",
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    heterogeneous_wear: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Construct a server_params JSON compatible with data_loader.py."""
    n_gpu, n_cpu = _gpu_cpu_split(n_servers)
    J      = list(range(n_servers))
    K      = list(range(horizon_seconds // slot_seconds))
    n_slots = len(K)
    slot_h  = slot_seconds / 3600.0

    racks = _rack_assignment(n_servers, rack_topology)

    def _param(j: int, key: str) -> float:
        return _GPU[key] if j < n_gpu else _CPU[key]

    sp: Dict[str, Any] = {}
    for key in ("C", "theta", "P0", "dP", "alpha", "lambda0", "lambda_pm", "Lambda"):
        sp[key] = {str(j): _param(j, key) for j in J}

    sp["psi_0"] = _build_psi_0(
        n_servers, n_gpu, psi_stage,
        heterogeneous=heterogeneous_wear,
        rng=rng,
    )

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

    psi_frac = PSI_STAGES[psi_stage]
    psi_note = (
        f"psi_stage='{psi_stage}' (f={psi_frac:.2f} × Lambda), "
        f"heterogeneous_wear={heterogeneous_wear}. "
        f"GPU psi_0 ≈ {psi_frac * _GPU['Lambda']:.1f}/{_GPU['Lambda']}, "
        f"CPU psi_0 ≈ {psi_frac * _CPU['Lambda']:.1f}/{_CPU['Lambda']}. "
        "For at_thresh/over_thresh stages, PM fires at or near slot 0; "
        "verify N_min feasibility for large fleets with tight time limits."
    )

    return {
        "_comments": {
            "generated_by":     "generate_robustness_tests.py (v2 — 20 cases)",
            "n_servers":        n_servers,
            "n_gpu":            n_gpu,
            "n_cpu":            n_cpu,
            "slot_seconds":     slot_seconds,
            "n_slots":          n_slots,
            "psi_stage":        psi_stage,
            "psi_fraction":     psi_frac,
            "psi_note":         psi_note,
            "rack_topology":    rack_topology,
            "n_racks":          len(racks),
            "heterogeneous_wear": heterogeneous_wear,
            "references": (
                "GPU params: Bashir et al. (2021) EuroSys; "
                "Fadaeefath Abadi et al. (2025) IEEE TDSC. "
                "CPU params: Weng et al. (2022) USENIX ATC. "
                "Thermal D: O'Brien et al. (2020) ASPLOS. "
                "Rack topology: Sinai et al. (2023) ASPLOS."
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
    Fitted to Alibaba 2021 GPU cluster trace (Weng et al. 2022, USENIX ATC).
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
# Precedence group (E) builder
# ---------------------------------------------------------------------------

def build_precedence_edges(
    batch_job_ids: List[int],
    pattern: str,
    rng: random.Random,
    n_slots: int,
    d: Dict[int, int],
    b: Dict[int, int],
) -> List[Tuple[int, int]]:
    """
    Build a list of (predecessor, successor) precedence edges for the E set.

    Three patterns (Qiao et al. 2021, OSDI; Weng et al. 2022, USENIX ATC):

    none
        E = [].  No precedence dependencies.

    chains
        Randomly partition a fraction (~30 %) of batch jobs into serial
        chains of length 2–5.  Models data-prep → training → evaluation
        pipeline stages common in ML workloads.

    fanout
        Build one or more map-reduce subgraphs:
          root → [k parallel workers] → aggregator
        where k ∈ {2, 3, 4}.  Models parameter-server and scatter-gather
        patterns.  Precedence is: root < each worker, each worker < aggregator.

    Only adds edges where successor.b > predecessor.a + predecessor.d so the
    model remains feasible (Qiao et al. 2021).
    """
    if pattern == "none" or len(batch_job_ids) < 4:
        return []

    edges: List[Tuple[int, int]] = []

    if pattern == "chains":
        # Sample ~30 % of batch jobs for chaining
        pool = rng.sample(batch_job_ids, k=max(2, int(0.30 * len(batch_job_ids))))
        rng.shuffle(pool)
        chain_len_min, chain_len_max = 2, min(5, len(pool))
        i = 0
        while i + 1 < len(pool):
            chain_len = rng.randint(chain_len_min, chain_len_max)
            chain = pool[i : i + chain_len]
            i += chain_len
            for pred, succ in zip(chain[:-1], chain[1:]):
                # Feasibility guard: succ must have enough horizon after pred
                if b.get(succ, n_slots) > b.get(pred, 0) + d.get(pred, 1):
                    edges.append((pred, succ))

    elif pattern == "fanout":
        # Build map-reduce subgraphs using ~35 % of batch jobs
        pool = rng.sample(batch_job_ids, k=max(4, int(0.35 * len(batch_job_ids))))
        rng.shuffle(pool)
        i = 0
        while i + 3 < len(pool):
            k_workers = rng.randint(2, min(4, len(pool) - i - 2))
            root       = pool[i]
            workers    = pool[i + 1 : i + 1 + k_workers]
            aggregator = pool[i + 1 + k_workers] if i + 1 + k_workers < len(pool) else None
            i += 1 + k_workers + (1 if aggregator else 0)

            for w in workers:
                if b.get(w, n_slots) > b.get(root, 0) + d.get(root, 1):
                    edges.append((root, w))
            if aggregator:
                for w in workers:
                    if b.get(aggregator, n_slots) > b.get(w, 0) + d.get(w, 1):
                        edges.append((w, aggregator))
    else:
        raise ValueError(f"Unknown precedence pattern: {pattern!r}")

    # Deduplicate and drop self-loops
    seen = set()
    clean: List[Tuple[int, int]] = []
    for e in edges:
        if e[0] != e[1] and e not in seen:
            seen.add(e)
            clean.append(e)
    return clean


# ---------------------------------------------------------------------------
# Job JSON builder
# ---------------------------------------------------------------------------

def build_jobs_json(
    n_jobs: int,
    n_servers: int,
    slot_seconds: int,
    prec_pattern: str = "none",
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

    # Build precedence edges (E set)
    edges = build_precedence_edges(
        batch_job_ids=I_B,
        pattern=prec_pattern,
        rng=rng,
        n_slots=n_slots,
        d={int(k): v for k, v in d_map.items()},
        b={int(k): v for k, v in b_map.items()},
    )

    return {
        "sets": {
            "I":   I,
            "I_B": I_B,
            "I_V": I_V,
            "I_C": I_C,
            "E":   [[p, s] for p, s in edges],   # serialisable list-of-lists
            "A":   [],
            "G":   [],
        },
        "eligibility": eligibility,
        "job_params":  {
            "d":   d_map,
            "r":   r_map,
            "a":   a_map,
            "b":   b_map,
            "q":   q_map,
            "rho": rho_map,
        },
        "metadata": {
            "generated_by":    "generate_robustness_tests.py (v2 — 20 cases)",
            "n_jobs":          n_jobs,
            "n_servers":       n_servers,
            "slot_seconds":    slot_seconds,
            "slot_duration_h": slot_h,
            "horizon_seconds": horizon_seconds,
            "horizon_slots":   n_slots,
            "n_batch":         len(I_B),
            "n_interactive":   len(I_V),
            "n_critical":      len(I_C),
            "n_precedence_edges": len(edges),
            "prec_pattern":    prec_pattern,
            "seed":            seed,
            "note": (
                "Synthetic jobs: log-normal(2.5,0.8) durations, bimodal "
                "resources, 40 % critical, geometric(0.55) replicas. "
                "Distributions fitted to Alibaba 2021 GPU cluster trace "
                "(Weng et al. 2022, USENIX ATC '22). "
                "Precedence patterns: chains/fanout from "
                "Qiao et al. (2021, OSDI) and Weng et al. (2022)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Hand-crafted 20-case specification table
# ---------------------------------------------------------------------------

# Each entry is a dict describing one test case across all six axes.
# Fields:
#   name          — short identifier (appended to filenames)
#   tier          — thematic group (1–5)
#   tier_label    — human description
#   n_servers     — fleet size
#   n_jobs        — workload size
#   k_slots       — number of time slots (determines slot_seconds)
#   psi_stage     — one of PSI_STAGES keys
#   rack_topology — one of RACK_TOPOLOGIES
#   prec_pattern  — one of PREC_PATTERNS
#   heterogeneous_wear — bool (True = random ±20% variation on psi_0)
#   seed          — per-case RNG seed for reproducibility
TWENTY_CASES = [
    # ------------------------------------------------------------------ Tier 1 — Baseline cluster
    dict(
        name="T1_A_baseline_small",
        tier=1, tier_label="Baseline cluster",
        n_servers=28, n_jobs=80,  k_slots=96,
        psi_stage="fresh",     rack_topology="standard", prec_pattern="none",
        heterogeneous_wear=False, seed=101,
        rationale=(
            "Small fleet, fresh wear, standard racks, no precedence. "
            "Serves as the minimal-complexity anchor case (Bowly et al. 2020)."
        ),
    ),
    dict(
        name="T1_B_baseline_mid",
        tier=1, tier_label="Baseline cluster",
        n_servers=42, n_jobs=172, k_slots=96,
        psi_stage="fresh",     rack_topology="standard", prec_pattern="none",
        heterogeneous_wear=False, seed=102,
        rationale=(
            "Default 42-server fleet (34 GPU + 8 CPU). Reproduces the exact "
            "configuration of the primary server JSON (server_params_42servers_v6.json). "
            "Weng et al. (2022): 34/8 GPU/CPU split matches MLaaS production clusters."
        ),
    ),
    dict(
        name="T1_C_coarse_slots",
        tier=1, tier_label="Baseline cluster",
        n_servers=42, n_jobs=120, k_slots=24,
        psi_stage="fresh",     rack_topology="standard", prec_pattern="none",
        heterogeneous_wear=False, seed=103,
        rationale=(
            "Coarse 1-hour slots (24 per day). Tests whether the MILP scales "
            "gracefully when K is reduced 4×, shrinking variable count but "
            "coarsening scheduling resolution (Bowly et al. 2020, §4.2)."
        ),
    ),
    dict(
        name="T1_D_fine_slots",
        tier=1, tier_label="Baseline cluster",
        n_servers=42, n_jobs=172, k_slots=192,
        psi_stage="fresh",     rack_topology="thin",    prec_pattern="none",
        heterogeneous_wear=False, seed=104,
        rationale=(
            "Fine 7.5-minute slots (192 per day). Maximises time-granularity "
            "variable count. Paired with thin racks (10 domains) to stress "
            "constraint #35 with high slot resolution. "
            "Sinai et al. (2023): fine-grained scheduling under rack isolation."
        ),
    ),

    # ------------------------------------------------------------------ Tier 2 — Wear stress
    dict(
        name="T2_A_midlife_skewed_racks",
        tier=2, tier_label="Wear stress",
        n_servers=42, n_jobs=150, k_slots=96,
        psi_stage="mid_life",  rack_topology="skewed",  prec_pattern="none",
        heterogeneous_wear=True, seed=201,
        rationale=(
            "Mid-life wear (±20 % heterogeneous) with skewed 3+3 rack topology. "
            "Heterogeneous psi_0 creates asymmetric PM pressure across servers "
            "(Fadaeefath Abadi et al. 2025). Skewed racks stress affinity "
            "placement for critical replicas."
        ),
    ),
    dict(
        name="T2_B_nearthresh_fat_racks",
        tier=2, tier_label="Wear stress",
        n_servers=56, n_jobs=200, k_slots=96,
        psi_stage="near_thresh", rack_topology="fat",   prec_pattern="none",
        heterogeneous_wear=True, seed=202,
        rationale=(
            "Near-threshold wear on a larger 56-server fleet with fat (4) racks. "
            "Constraint #26 (PM trigger) becomes binding for most servers. "
            "Fat racks reduce diversity, making constraint #35 harder to satisfy "
            "(Peng et al. 2023, IEEE TPDS §V-B)."
        ),
    ),
    dict(
        name="T2_C_atthresh_two_domain",
        tier=2, tier_label="Wear stress",
        n_servers=42, n_jobs=172, k_slots=48,
        psi_stage="at_thresh", rack_topology="two_domain", prec_pattern="none",
        heterogeneous_wear=False, seed=203,
        rationale=(
            "All servers exactly at Lambda threshold; PM must fire immediately "
            "(slot 0 or 1). Two-domain (2 rack) topology maximally constrains "
            "replica placement for critical jobs. Tests feasibility under "
            "simultaneous PM demand vs N_min (Duplyakin et al. 2021, ASPLOS)."
        ),
    ),
    dict(
        name="T2_D_overthresh_hetero",
        tier=2, tier_label="Wear stress",
        n_servers=35, n_jobs=100, k_slots=96,
        psi_stage="over_thresh", rack_topology="standard", prec_pattern="chains",
        heterogeneous_wear=True, seed=204,
        rationale=(
            "Over-threshold wear (±20 % heterogeneous) on a small fleet. "
            "Validates that the model does not silently ignore over-limit wear "
            "and that PM fires in slot 0. Sparse chain precedence added to test "
            "interaction between wear-triggered PM and job sequencing "
            "(Fadaeefath Abadi et al. 2025; Qiao et al. 2021)."
        ),
    ),

    # ------------------------------------------------------------------ Tier 3 — Precedence stress
    dict(
        name="T3_A_sparse_chains",
        tier=3, tier_label="Precedence stress",
        n_servers=42, n_jobs=150, k_slots=96,
        psi_stage="fresh",    rack_topology="standard", prec_pattern="chains",
        heterogeneous_wear=False, seed=301,
        rationale=(
            "Sparse serial chains (~30 % of batch jobs) on the standard fleet. "
            "Models ML pipeline stages: data-prep → training → evaluation "
            "(Qiao et al. 2021, OSDI §3). Tests constraint #8 (precedence) "
            "without compounding other stress axes."
        ),
    ),
    dict(
        name="T3_B_dense_chains_fineslots",
        tier=3, tier_label="Precedence stress",
        n_servers=42, n_jobs=200, k_slots=192,
        psi_stage="mid_life", rack_topology="thin",    prec_pattern="chains",
        heterogeneous_wear=False, seed=302,
        rationale=(
            "Dense chains on fine-grained slots (192 K). High chain density "
            "combined with many slots creates long critical paths that compete "
            "with the scheduling horizon. Thin racks (10 domains) add placement "
            "diversity pressure. Peng et al. (2023): precedence + granularity "
            "interaction is a primary solver hardness driver."
        ),
    ),
    dict(
        name="T3_C_fanout_standard",
        tier=3, tier_label="Precedence stress",
        n_servers=56, n_jobs=180, k_slots=96,
        psi_stage="fresh",    rack_topology="standard", prec_pattern="fanout",
        heterogeneous_wear=False, seed=303,
        rationale=(
            "Fan-out (map-reduce) subgraphs on a 56-server fleet. Models "
            "parameter-server and scatter-gather communication patterns "
            "(Weng et al. 2022, USENIX ATC §5). Tests whether the MILP "
            "can schedule root → workers → aggregator DAGs without timeline "
            "infeasibility."
        ),
    ),
    dict(
        name="T3_D_fanout_wear_skewed",
        tier=3, tier_label="Precedence stress",
        n_servers=42, n_jobs=160, k_slots=96,
        psi_stage="near_thresh", rack_topology="skewed", prec_pattern="fanout",
        heterogeneous_wear=True, seed=304,
        rationale=(
            "Fan-out DAGs combined with near-threshold heterogeneous wear on "
            "skewed racks. Tests the joint feasibility of precedence scheduling, "
            "PM triggering (constraint #26), and asymmetric rack diversity "
            "(constraint #35). Interaction identified as high-difficulty by "
            "Bowly et al. (2020, §5.3)."
        ),
    ),

    # ------------------------------------------------------------------ Tier 4 — Scale stress
    dict(
        name="T4_A_large_fleet_coarse",
        tier=4, tier_label="Scale stress",
        n_servers=100, n_jobs=300, k_slots=48,
        psi_stage="fresh",    rack_topology="fat",     prec_pattern="none",
        heterogeneous_wear=False, seed=401,
        rationale=(
            "Large 100-server fleet with 300 jobs and coarse 30-min slots. "
            "Variable count ∝ |I|×|J|×|K|; coarser K keeps it tractable. "
            "Fat racks (4 domains) across 100 servers tests rack-diversity "
            "constraint under high replica demand. Weng et al. (2022): "
            "cluster sizes of 100+ GPUs are standard in production."
        ),
    ),
    dict(
        name="T4_B_large_jobs_fine",
        tier=4, tier_label="Scale stress",
        n_servers=56, n_jobs=400, k_slots=96,
        psi_stage="mid_life", rack_topology="standard", prec_pattern="none",
        heterogeneous_wear=False, seed=402,
        rationale=(
            "High job count (400) on a mid-size fleet. Tests solver scaling "
            "on dense assignment matrices (|I|×|J|×|K| = 400×56×96). "
            "Mid-life wear adds a non-trivial PM sub-problem. Bowly et al. "
            "(2020): job count is the dominant hardness driver in scheduling MILPs."
        ),
    ),
    dict(
        name="T4_C_huge_fleet_thin_racks",
        tier=4, tier_label="Scale stress",
        n_servers=100, n_jobs=250, k_slots=96,
        psi_stage="fresh",    rack_topology="thin",    prec_pattern="chains",
        heterogeneous_wear=False, seed=403,
        rationale=(
            "100-server fleet with thin (10-rack) topology and sparse chains. "
            "Tests constraint #35 at large scale: with 100 servers across 10 "
            "racks, critical-replica placement has many valid options but "
            "scheduling chains on GPU racks can bottleneck throughput "
            "(Sinai et al. 2023; Qiao et al. 2021)."
        ),
    ),
    dict(
        name="T4_D_large_fleet_wear",
        tier=4, tier_label="Scale stress",
        n_servers=70, n_jobs=300, k_slots=96,
        psi_stage="near_thresh", rack_topology="skewed", prec_pattern="none",
        heterogeneous_wear=True, seed=404,
        rationale=(
            "Large fleet (70 servers) at near-threshold wear with heterogeneous "
            "psi_0 and skewed racks. All but a few servers are approaching PM "
            "eligibility simultaneously, creating a combinatorial PM scheduling "
            "problem on top of job placement. Tests the worst-case N_min "
            "feasibility analysed by Fadaeefath Abadi et al. (2025)."
        ),
    ),

    # ------------------------------------------------------------------ Tier 5 — Combined extremes
    dict(
        name="T5_A_extreme_small",
        tier=5, tier_label="Combined extremes",
        n_servers=28, n_jobs=40,  k_slots=24,
        psi_stage="at_thresh", rack_topology="two_domain", prec_pattern="fanout",
        heterogeneous_wear=False, seed=501,
        rationale=(
            "Smallest possible stress: 28 servers, 40 jobs, 24 slots, all at "
            "wear threshold, 2-domain racks, fan-out DAGs. Designed to produce "
            "a near-infeasible instance (PM + replica diversity + precedence on "
            "a tiny fleet). Tests IIS diagnostics and constraint interaction "
            "(Duplyakin et al. 2021; Bowly et al. 2020, §5.4)."
        ),
    ),
    dict(
        name="T5_B_extreme_large",
        tier=5, tier_label="Combined extremes",
        n_servers=100, n_jobs=400, k_slots=192,
        psi_stage="near_thresh", rack_topology="thin",    prec_pattern="fanout",
        heterogeneous_wear=True, seed=502,
        rationale=(
            "Maximum-scale stress: 100 servers, 400 jobs, 192 fine slots, "
            "near-threshold heterogeneous wear, 10-domain thin racks, fan-out "
            "DAGs. Expected to hit the Gurobi time limit; validates that the "
            "solver returns a feasible incumbent within the time budget "
            "(Bowly et al. 2020; Peng et al. 2023)."
        ),
    ),
    dict(
        name="T5_C_combined_chains_wear",
        tier=5, tier_label="Combined extremes",
        n_servers=56, n_jobs=250, k_slots=96,
        psi_stage="over_thresh", rack_topology="fat",     prec_pattern="chains",
        heterogeneous_wear=True, seed=503,
        rationale=(
            "Over-threshold heterogeneous wear (±20 %) on a 56-server fleet "
            "with fat (4) racks and dense serial chains. PM must fire in slot 0 "
            "for many servers; chain precedence limits which jobs can be "
            "frontloaded. Interaction between early PM windows and job precedence "
            "is the key stress mechanism (Fadaeefath Abadi et al. 2025; "
            "Qiao et al. 2021)."
        ),
    ),
    dict(
        name="T5_D_combined_all_axes",
        tier=5, tier_label="Combined extremes",
        n_servers=70, n_jobs=300, k_slots=48,
        psi_stage="mid_life",  rack_topology="skewed",  prec_pattern="fanout",
        heterogeneous_wear=True, seed=504,
        rationale=(
            "All six axes simultaneously at non-trivial levels: 70 servers, "
            "300 jobs, 30-min slots, mid-life heterogeneous wear, skewed 3+3 "
            "racks, fan-out DAGs. Designed to expose solver sensitivity to "
            "multi-axis interactions that single-axis sweeps cannot detect "
            "(Bowly et al. 2020, ablation methodology §3). "
            "Represents a realistic future-cycle daily run scenario."
        ),
    ),
]

assert len(TWENTY_CASES) == 20, f"Expected 20 cases, got {len(TWENTY_CASES)}"


# ---------------------------------------------------------------------------
# Test-case catalogue builder
# ---------------------------------------------------------------------------

def generate_test_suite(
    output_dir: Path,
    seed_offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Build and write all 20 test-case JSON pairs.  Returns a manifest list.

    The per-case seed in TWENTY_CASES ensures reproducible job sampling;
    seed_offset can shift all seeds simultaneously for sensitivity analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    print(f"Generating {len(TWENTY_CASES)} test cases in {output_dir} …\n")

    for idx, case in enumerate(TWENTY_CASES):
        slot_s  = DEFAULT_HORIZON_SECONDS // case["k_slots"]
        n_slots = DEFAULT_HORIZON_SECONDS // slot_s
        name    = case["name"]
        seed    = case["seed"] + seed_offset

        rng_for_wear = random.Random(seed)

        server_data = build_server_json(
            n_servers=case["n_servers"],
            slot_seconds=slot_s,
            psi_stage=case["psi_stage"],
            rack_topology=case["rack_topology"],
            heterogeneous_wear=case["heterogeneous_wear"],
            rng=rng_for_wear,
        )
        jobs_data = build_jobs_json(
            n_jobs=case["n_jobs"],
            n_servers=case["n_servers"],
            slot_seconds=slot_s,
            prec_pattern=case["prec_pattern"],
            seed=seed,
        )

        server_path = output_dir / f"server_params_{name}.json"
        jobs_path   = output_dir / f"jobs_params_{name}.json"

        with open(server_path, "w", encoding="utf-8") as f:
            json.dump(server_data, f, indent=2)
        with open(jobs_path, "w", encoding="utf-8") as f:
            json.dump(jobs_data, f, indent=2)

        n_gpu = server_data["_comments"]["n_gpu"]
        frac  = PSI_STAGES[case["psi_stage"]]

        manifest.append({
            "case_name":          name,
            "tier":               case["tier"],
            "tier_label":         case["tier_label"],
            "n_servers":          case["n_servers"],
            "n_gpu":              n_gpu,
            "n_cpu":              server_data["_comments"]["n_cpu"],
            "n_jobs":             case["n_jobs"],
            "n_batch":            jobs_data["metadata"]["n_batch"],
            "n_interactive":      jobs_data["metadata"]["n_interactive"],
            "n_critical":         jobs_data["metadata"]["n_critical"],
            "k_slots":            n_slots,
            "slot_seconds":       slot_s,
            "slot_minutes":       slot_s // 60,
            "psi_stage":          case["psi_stage"],
            "psi_fraction":       frac,
            "gpu_psi_0":          round(frac * _GPU["Lambda"], 2),
            "cpu_psi_0":          round(frac * _CPU["Lambda"], 2),
            "gpu_lambda":         _GPU["Lambda"],
            "cpu_lambda":         _CPU["Lambda"],
            "heterogeneous_wear": case["heterogeneous_wear"],
            "rack_topology":      case["rack_topology"],
            "n_racks":            server_data["_comments"]["n_racks"],
            "prec_pattern":       case["prec_pattern"],
            "n_precedence_edges": jobs_data["metadata"]["n_precedence_edges"],
            "seed":               seed,
            "server_file":        str(server_path),
            "jobs_file":          str(jobs_path),
            "rationale":          case["rationale"],
        })

        slot_min_str = f"{slot_s // 60}min"
        print(
            f"  [{idx+1:>2}/20] {name}\n"
            f"         servers={case['n_servers']} jobs={case['n_jobs']} "
            f"K={n_slots} ({slot_min_str}) | "
            f"wear={case['psi_stage']}{'(het.)' if case['heterogeneous_wear'] else '':6} | "
            f"racks={case['rack_topology']:10} prec={case['prec_pattern']:6} | "
            f"edges={jobs_data['metadata']['n_precedence_edges']}"
        )

    # Write manifest
    manifest_path = output_dir / "test_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written → {manifest_path}  ({len(manifest)} entries)")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 20 synthetic robustness-test JSON pairs for "
                    "the data-centre MILP (solver.py) across 6 axes: "
                    "servers, jobs, slots, failure-sets (F), wear (psi_0), "
                    "and precedence groups (E).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="data/robustness_tests",
        help="Output directory for JSON pairs and manifest.",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Add this value to every case seed (sensitivity analysis).",
    )
    args = parser.parse_args()

    manifest = generate_test_suite(
        output_dir=Path(args.output_dir),
        seed_offset=args.seed_offset,
    )

    # Summary table
    hdr = (
        f"\n{'#':>2}  {'Case':<38} {'Svrs':>4} {'Jobs':>5} {'K':>4} "
        f"{'PSI_stage':<13} {'Racks':<11} {'Prec':>5} {'Edges':>5}"
    )
    print(hdr)
    print("-" * len(hdr.lstrip()))
    for i, m in enumerate(manifest, 1):
        tier_sep = "\n" if i > 1 and (i - 1) % 4 == 0 else ""
        print(
            f"{tier_sep}{i:>2}  {m['case_name']:<38} "
            f"{m['n_servers']:>4} {m['n_jobs']:>5} {m['k_slots']:>4} "
            f"{m['psi_stage']:<13} {m['rack_topology']:<11} "
            f"{m['prec_pattern']:>5} {m['n_precedence_edges']:>5}"
        )

    print(f"\nTotal cases: {len(manifest)}")
    print("\nTier breakdown:")
    for tier in range(1, 6):
        tier_cases = [m for m in manifest if m["tier"] == tier]
        print(f"  Tier {tier} ({tier_cases[0]['tier_label']}): "
              f"{len(tier_cases)} cases")
    print("\nDone.")


if __name__ == "__main__":
    main()
