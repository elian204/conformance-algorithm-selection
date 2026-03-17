"""
experiments/benchmark_loader.py
---------------------------------
Loads benchmark datasets and process models.

Supports:
  - XES event logs  (via PM4Py)
  - PNML process models (via PM4Py) -- "given" model option
  - Inductive Miner model discovery from a log (via PM4Py) -- "discover" option
  - Conversion of PM4Py objects to internal WorkflowNet representation
  - Model quality metrics (fitness, precision)
  - Built-in synthetic benchmarks for unit testing

File naming convention (from screenshot):
  pr-{P}-{T}-{places}-{model_id}_{model_size}_{log_level}.xes
  e.g.  pr-1-11-1244-A59_m17_l1.xes  ->  model at pr-1-11-1244-A59_m17.pnml
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.petri_net import PetriNet, WorkflowNet, Place, marking_from_dict, marking_from_places
from core.trace_model import build_trace_net  # noqa: imported for convenience

# ---------------------------------------------------------------------------
# PM4Py import guard
# ---------------------------------------------------------------------------
try:
    import pm4py

    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("[WARNING] pm4py not available. Install with: pip install pm4py")


# ---------------------------------------------------------------------------
# Internal conversion: PM4Py -> WorkflowNet
# ---------------------------------------------------------------------------

def convert_pm4py_net(pm_net, pm_initial, pm_final) -> WorkflowNet:
    """Convert a PM4Py PetriNet + markings to our internal WorkflowNet."""
    net = PetriNet(name=getattr(pm_net, 'name', 'model') or 'model')
    place_map: Dict = {}
    trans_map: Dict = {}

    for pm_p in pm_net.places:
        place_map[pm_p] = net.add_place(pm_p.name)

    for pm_t in pm_net.transitions:
        trans_map[pm_t] = net.add_transition(pm_t.name, pm_t.label)

    for arc in pm_net.arcs:
        src, tgt = arc.source, arc.target
        if hasattr(src, 'label'):  # Transition -> Place
            net.add_arc_transition_to_place(trans_map[src], place_map[tgt])
        else:  # Place -> Transition
            net.add_arc_place_to_transition(place_map[src], trans_map[tgt])

    init_m = marking_from_dict({place_map[p]: count for p, count in pm_initial.items()})
    final_m = marking_from_dict({place_map[p]: count for p, count in pm_final.items()})
    return WorkflowNet(net=net, initial_marking=init_m, final_marking=final_m)


# ---------------------------------------------------------------------------
# Model loading from PNML
# ---------------------------------------------------------------------------

def load_model_from_pnml(pnml_path: str) -> Tuple[WorkflowNet, object, object, object]:
    """
    Load a Petri net from a PNML file.

    Returns (WorkflowNet, pm_net, pm_initial, pm_final)
    The PM4Py objects are retained for quality-metric computation.
    """
    if not PM4PY_AVAILABLE:
        raise ImportError("pm4py is required to load PNML files.")
    pm_net, pm_initial, pm_final = pm4py.read_pnml(pnml_path)
    wf = convert_pm4py_net(pm_net, pm_initial, pm_final)
    print(f"[Loader] Model '{Path(pnml_path).name}': "
          f"|P|={len(wf.net.places)}, |T|={len(wf.net.transitions)}, "
          f"τ={sum(1 for t in wf.net.transitions if t.label is None)}")
    return wf, pm_net, pm_initial, pm_final


# ---------------------------------------------------------------------------
# Model loading from Pickle (.pkl)
# ---------------------------------------------------------------------------

def load_model_from_pickle(pkl_path: str) -> Tuple[WorkflowNet, object, object, object]:
    """
    Load a Petri net from a pickle file.

    Expected pickle structure (dict):
        {'net': PetriNet, 'im': Marking, 'fm': Marking}

    Alternative structure (tuple):
        (PetriNet, initial_marking, final_marking)

    Returns (WorkflowNet, pm_net, pm_initial, pm_final)
    """
    import pickle

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    # Handle dict format: {'net': ..., 'im': ..., 'fm': ...}
    if isinstance(obj, dict):
        pm_net = obj.get('net') or obj.get('petri_net') or obj.get('model')
        pm_initial = obj.get('im') or obj.get('initial_marking') or obj.get('init')
        pm_final = obj.get('fm') or obj.get('final_marking') or obj.get('final')
    # Handle tuple format: (net, im, fm)
    elif isinstance(obj, (tuple, list)) and len(obj) == 3:
        pm_net, pm_initial, pm_final = obj
    else:
        raise ValueError(
            f"Unrecognized pickle structure: {type(obj)}. "
            "Expected dict with keys 'net','im','fm' or a 3-tuple."
        )

    if pm_net is None or pm_initial is None or pm_final is None:
        raise ValueError(
            f"Could not extract (net, im, fm) from pickle. "
            f"Got: net={pm_net is not None}, im={pm_initial is not None}, fm={pm_final is not None}"
        )

    wf = convert_pm4py_net(pm_net, pm_initial, pm_final)
    print(f"[Loader] Model '{Path(pkl_path).name}': "
          f"|P|={len(wf.net.places)}, |T|={len(wf.net.transitions)}, "
          f"τ={sum(1 for t in wf.net.transitions if t.label is None)}")
    return wf, pm_net, pm_initial, pm_final


# ---------------------------------------------------------------------------
# Unified model loader (auto-detects format)
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> Tuple[WorkflowNet, object, object, object]:
    """
    Load a Petri net from file, auto-detecting format by extension.

    Supported formats:
      - .pnml  : standard PNML format (via PM4Py)
      - .pkl   : pickled PM4Py objects

    Returns (WorkflowNet, pm_net, pm_initial, pm_final)
    """
    path = Path(model_path)
    ext = path.suffix.lower()

    if ext == '.pnml':
        return load_model_from_pnml(model_path)
    elif ext == '.pkl':
        return load_model_from_pickle(model_path)
    else:
        raise ValueError(
            f"Unsupported model format: '{ext}'. "
            "Supported: .pnml, .pkl"
        )


# ---------------------------------------------------------------------------
# Model discovery via Inductive Miner
# ---------------------------------------------------------------------------

def discover_model_inductive_miner(
        log_path: str,
        noise_threshold: float = 0.0,
        save_to: Optional[str] = None,
) -> Tuple[WorkflowNet, object, object, object]:
    """
    Discover a process model from an XES log using PM4Py's Inductive Miner.

    Parameters
    ----------
    log_path        : path to the .xes event log
    noise_threshold : 0.0 = standard IM;  >0 = IMf (infrequent variant)
    save_to         : if given, save the discovered model as PNML here

    Returns
    -------
    (WorkflowNet, pm_net, pm_initial, pm_final)
    """
    if not PM4PY_AVAILABLE:
        raise ImportError("pm4py is required for Inductive Miner discovery.")

    print(f"[Discovery] Log: {Path(log_path).name}, noise_threshold={noise_threshold}")
    log = pm4py.read_xes(log_path)

    # Convert to EventLog if DataFrame (for discovery compatibility)
    if hasattr(log, 'iterrows'):
        log = pm4py.convert_to_event_log(log)

    if noise_threshold > 0.0:
        pm_net, pm_initial, pm_final = pm4py.discover_petri_net_inductive(
            log, noise_threshold=noise_threshold
        )
    else:
        pm_net, pm_initial, pm_final = pm4py.discover_petri_net_inductive(log)

    wf = convert_pm4py_net(pm_net, pm_initial, pm_final)
    n_silent = sum(1 for t in wf.net.transitions if t.label is None)
    print(f"[Discovery] Model discovered: "
          f"|P|={len(wf.net.places)}, |T|={len(wf.net.transitions)}, τ={n_silent}")

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        pm4py.write_pnml(pm_net, pm_initial, pm_final, str(save_to))
        print(f"[Discovery] Saved to: {save_to}")

    return wf, pm_net, pm_initial, pm_final


# ---------------------------------------------------------------------------
# Log (XES) loading
# ---------------------------------------------------------------------------

def load_traces_from_xes(
        xes_path: str,
        max_traces: Optional[int] = None,
        activity_key: str = "concept:name",
) -> Tuple[List[List[str]], List[str]]:
    """
    Load traces from an XES event log.

    Returns
    -------
    (traces, case_ids)
    """
    if not PM4PY_AVAILABLE:
        raise ImportError("pm4py is required to load XES files.")

    log = pm4py.read_xes(xes_path)
    traces: List[List[str]] = []
    case_ids: List[str] = []

    # Handle both old EventLog format and new pandas DataFrame format
    if hasattr(log, 'iterrows'):
        # Pandas DataFrame format (PM4Py >= 2.3)
        import pandas as pd
        case_id_key = "case:concept:name"
        if case_id_key not in log.columns:
            case_id_key = "case:id" if "case:id" in log.columns else None

        grouped = log.groupby(case_id_key) if case_id_key else [(str(i), g) for i, g in
                                                                enumerate(log.groupby(log.index))]

        for case_id, group in grouped:
            trace = group[activity_key].tolist()
            traces.append(trace)
            case_ids.append(str(case_id))
            if max_traces and len(traces) >= max_traces:
                break
    else:
        # Traditional EventLog format (PM4Py < 2.3)
        for case in log:
            trace = [event[activity_key] for event in case]
            case_id = case.attributes.get("concept:name", str(len(traces)))
            traces.append(trace)
            case_ids.append(str(case_id))
            if max_traces and len(traces) >= max_traces:
                break

    avg = sum(len(t) for t in traces) / max(1, len(traces))
    print(f"[Loader] Log '{Path(xes_path).name}': "
          f"{len(traces)} traces, avg_len={avg:.1f}")
    return traces, case_ids


# ---------------------------------------------------------------------------
# Model quality metrics
# ---------------------------------------------------------------------------

def compute_model_quality(
        pm_net, pm_initial, pm_final, xes_path: str
) -> Dict[str, float]:
    """
    Compute token-based replay fitness and precision.
    Returns {'fitness': float, 'precision': float}; NaN on failure.
    """
    fitness = float('nan')
    precision = float('nan')

    if not PM4PY_AVAILABLE:
        return {"fitness": fitness, "precision": precision}

    try:
        log = pm4py.read_xes(xes_path)
        # Convert to EventLog if DataFrame (for replay compatibility)
        if hasattr(log, 'iterrows'):
            log = pm4py.convert_to_event_log(log)
        res = pm4py.fitness_token_based_replay(log, pm_net, pm_initial, pm_final)
        fitness = float(res.get("average_trace_fitness", float('nan')))
    except Exception as e:
        print(f"[Warning] Fitness failed: {e}")

    try:
        log = pm4py.read_xes(xes_path)
        if hasattr(log, 'iterrows'):
            log = pm4py.convert_to_event_log(log)
        precision = float(
            pm4py.precision_token_based_replay(log, pm_net, pm_initial, pm_final)
        )
    except Exception as e:
        print(f"[Warning] Precision failed: {e}")

    return {"fitness": fitness, "precision": precision}


# ---------------------------------------------------------------------------
# Model structural characteristics
# ---------------------------------------------------------------------------

def model_characteristics(wf: WorkflowNet) -> Dict:
    """Extract structural properties of the workflow net."""
    net = wf.net
    n_places = len(net.places)
    n_trans = len(net.transitions)
    n_silent = sum(1 for t in net.transitions if t.label is None)
    n_labeled = n_trans - n_silent
    # AND-split proxy: transitions with |postset| > 1
    n_and_splits = sum(1 for t in net.transitions if len(net._postset[t]) > 1)
    # AND-join proxy: transitions with |preset| > 1
    n_and_joins = sum(1 for t in net.transitions if len(net._preset[t]) > 1)
    # XOR-split proxy: places that are input to >1 transition
    n_xor_splits = sum(
        1 for p in net.places
        if sum(1 for t in net.transitions if p in net._preset[t]) > 1
    )
    return {
        "model_places": n_places,
        "model_transitions": n_trans,
        "model_silent_tau": n_silent,
        "model_labeled": n_labeled,
        "model_and_splits": n_and_splits,
        "model_and_joins": n_and_joins,
        "model_xor_splits": n_xor_splits,
    }


# ---------------------------------------------------------------------------
# High-level dataset loader (single entry point)
# ---------------------------------------------------------------------------

def load_dataset(
        log_path: str,
        model_path: Optional[str] = None,
        discover_model: bool = False,
        noise_threshold: float = 0.0,
        save_discovered_model: bool = True,
        max_traces: Optional[int] = None,
        activity_key: str = "concept:name",
        compute_quality: bool = True,
) -> Tuple[WorkflowNet, List[List[str]], List[str], Dict]:
    """
    Master loader: load log + model (given or discovered via Inductive Miner).

    Parameters
    ----------
    log_path              : path to .xes event log
    model_path            : path to .pnml file  (use this OR discover_model)
    discover_model        : run Inductive Miner on the log to find a model
    noise_threshold       : IM noise threshold (only when discover_model=True)
    save_discovered_model : persist discovered model as <stem>_discovered.pnml
    max_traces            : cap on number of traces loaded
    activity_key          : XES attribute name for activity label
    compute_quality       : whether to compute fitness/precision (can be slow)

    Returns
    -------
    (workflow_net, traces, case_ids, metadata_dict)
    """
    if not PM4PY_AVAILABLE:
        raise ImportError("pm4py is required. pip install pm4py")

    # --- Traces ---
    traces, case_ids = load_traces_from_xes(log_path, max_traces, activity_key)

    # --- Model ---
    pm_net = pm_initial = pm_final = None
    if model_path:
        wf, pm_net, pm_initial, pm_final = load_model(model_path)
        model_source = f"given:{Path(model_path).name}"
    elif discover_model:
        save_to = None
        if save_discovered_model:
            stem = Path(log_path).stem
            save_to = str(Path(log_path).parent / f"{stem}_discovered.pnml")
        wf, pm_net, pm_initial, pm_final = discover_model_inductive_miner(
            log_path, noise_threshold=noise_threshold, save_to=save_to
        )
        model_source = f"discovered:IM(noise={noise_threshold})"
    else:
        raise ValueError("Provide model_path= or set discover_model=True")

    # --- Quality metrics ---
    quality = {"fitness": float("nan"), "precision": float("nan")}
    if compute_quality and pm_net is not None:
        quality = compute_model_quality(pm_net, pm_initial, pm_final, log_path)
        print(f"[Quality] fitness={quality['fitness']:.4f}, "
              f"precision={quality['precision']:.4f}")

    # --- Metadata ---
    char = model_characteristics(wf)
    avg_len = sum(len(t) for t in traces) / max(1, len(traces))
    metadata = {
        "dataset": Path(log_path).name,
        "log_path": str(log_path),
        "model_source": model_source,
        "n_traces": len(traces),
        "avg_trace_length": round(avg_len, 4),
        **quality,
        **char,
    }
    return wf, traces, case_ids, metadata


# ---------------------------------------------------------------------------
# Built-in synthetic benchmarks (for unit tests — no files needed)
# ---------------------------------------------------------------------------

def build_paper_example() -> Tuple[WorkflowNet, List[str]]:
    """Section 4 paper example: AND-split/join model, σ=<a,e>, C*=2."""
    net = PetriNet("paper_example")
    pi = net.add_place("pi");
    p1 = net.add_place("p1");
    p2 = net.add_place("p2")
    p3 = net.add_place("p3");
    p4 = net.add_place("p4");
    pf = net.add_place("pf")
    ta = net.add_transition("ta", "a");
    tb = net.add_transition("tb", "b")
    tc = net.add_transition("tc", "c");
    td = net.add_transition("td", "d")
    te = net.add_transition("te", "e")
    net.add_arc_place_to_transition(pi, ta)
    net.add_arc_transition_to_place(ta, p1);
    net.add_arc_transition_to_place(ta, p2)
    net.add_arc_place_to_transition(p1, tb);
    net.add_arc_transition_to_place(tb, p3)
    net.add_arc_place_to_transition(p2, tc);
    net.add_arc_transition_to_place(tc, p4)
    net.add_arc_place_to_transition(p2, td);
    net.add_arc_transition_to_place(td, p4)
    net.add_arc_place_to_transition(p3, te);
    net.add_arc_place_to_transition(p4, te)
    net.add_arc_transition_to_place(te, pf)
    wf = WorkflowNet(net=net,
                     initial_marking=marking_from_places(pi),
                     final_marking=marking_from_places(pf))
    return wf, ["a", "e"]


def build_simple_sequence() -> Tuple[WorkflowNet, List[str]]:
    """Simple sequence a->b->c, trace <a,c>, C*=1."""
    net = PetriNet("simple_sequence")
    p0 = net.add_place("p0");
    p1 = net.add_place("p1")
    p2 = net.add_place("p2");
    p3 = net.add_place("p3")
    ta = net.add_transition("ta", "a");
    tb = net.add_transition("tb", "b")
    tc = net.add_transition("tc", "c")
    net.add_arc_place_to_transition(p0, ta);
    net.add_arc_transition_to_place(ta, p1)
    net.add_arc_place_to_transition(p1, tb);
    net.add_arc_transition_to_place(tb, p2)
    net.add_arc_place_to_transition(p2, tc);
    net.add_arc_transition_to_place(tc, p3)
    wf = WorkflowNet(net=net,
                     initial_marking=marking_from_places(p0),
                     final_marking=marking_from_places(p3))
    return wf, ["a", "c"]
