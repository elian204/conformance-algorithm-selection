"""
core/trace_model.py
-------------------
Constructs the event-net representation TN of a trace σ = <a1, ..., an>.

  TN = (P_TN, T_TN, F_TN, λ_TN, m_i^TN, m_f^TN)
  P_TN  = {q0, ..., qn}
  T_TN  = {t'1, ..., t'n}
  F_TN  = {(q_{i-1}, t'_i), (t'_i, q_i) | i=1..n}
  λ_TN(t'_i) = a_i
"""
from __future__ import annotations
from typing import List, Optional

from core.petri_net import PetriNet, WorkflowNet, Place, Transition, marking_from_places


def build_trace_net(trace: List[str], trace_id: str = "trace") -> WorkflowNet:
    """
    Build the event net for a trace.

    Parameters
    ----------
    trace : list of activity labels, e.g. ['a', 'e']
    trace_id : identifier prefix for place/transition names

    Returns
    -------
    WorkflowNet with the event net and its initial/final markings.
    """
    net = PetriNet(name=f"TN_{trace_id}")
    n = len(trace)

    # Places q0 ... qn
    places: List[Place] = []
    for i in range(n + 1):
        p = net.add_place(f"q{i}_{trace_id}")
        places.append(p)

    # Transitions t'1 ... t'n
    for i, activity in enumerate(trace):
        t = net.add_transition(
            name=f"t'_{i+1}_{trace_id}",
            label=activity
        )
        net.add_arc_place_to_transition(places[i], t)
        net.add_arc_transition_to_place(t, places[i + 1])

    initial_marking = marking_from_places(places[0])
    final_marking   = marking_from_places(places[n])

    return WorkflowNet(net=net,
                       initial_marking=initial_marking,
                       final_marking=final_marking)
