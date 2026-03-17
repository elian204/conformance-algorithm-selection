"""
Utilities for exporting Petri net visualizations.
"""
from __future__ import annotations

import os
from contextlib import redirect_stderr
from pathlib import Path
from typing import Optional, Tuple

from graphviz import Digraph

from core.petri_net import Marking, PetriNet, Place, Transition, WorkflowNet


def _normalize_output_path(output_path: str) -> Tuple[Path, Path]:
    """Return the Graphviz render stem and final PDF path."""
    path = Path(output_path)
    if path.suffix.lower() == ".pdf":
        return path.with_suffix(""), path
    return path, path.with_suffix(".pdf")


def _node_id(prefix: str, name: str) -> str:
    return f"{prefix}__{name}"


def _resolve_net_and_markings(
    net: PetriNet | WorkflowNet,
    marking: Optional[Marking],
) -> Tuple[PetriNet, Optional[Marking], Optional[Marking], Optional[Marking]]:
    if isinstance(net, WorkflowNet):
        petri_net = net.net
        initial_marking = net.initial_marking
        final_marking = net.final_marking
    elif isinstance(net, PetriNet):
        petri_net = net
        initial_marking = None
        final_marking = None
    else:
        raise TypeError(
            "net must be a core.petri_net.PetriNet or core.petri_net.WorkflowNet"
        )

    if marking is not None and not isinstance(marking, Marking):
        raise TypeError("marking must be a core.petri_net.Marking or None")

    return petri_net, marking, initial_marking, final_marking


def build_petri_net_digraph(
    net: PetriNet | WorkflowNet,
    marking: Optional[Marking] = None,
) -> Digraph:
    """
    Build a Graphviz graph for an internal Petri net or workflow net.
    """
    petri_net, marking, initial_marking, final_marking = _resolve_net_and_markings(
        net, marking
    )

    viz = Digraph(engine="dot")
    viz.attr(rankdir="TB")

    for place in sorted(petri_net.places, key=lambda item: item.name):
        label = place.name
        tokens = marking.count(place) if marking is not None else 0
        if tokens == 1:
            label += "\n<FONT POINT-SIZE='30'>●</FONT>"
        elif tokens > 1:
            label += f"\n<FONT POINT-SIZE='18'>{tokens}</FONT>"

        viz.node(
            _node_id("place", place.name),
            label=f"<{label}>",
            shape="circle",
            style="filled",
            fillcolor="white",
            fixedsize="true",
            width="0.75",
            height="0.75",
        )

    for transition in sorted(petri_net.transitions, key=lambda item: item.name):
        label = transition.label if transition.label is not None else "tau"
        viz.node(_node_id("transition", transition.name), label=label, shape="box")

        for place in sorted(petri_net.preset(transition), key=lambda item: item.name):
            viz.edge(
                _node_id("place", place.name),
                _node_id("transition", transition.name),
            )
        for place in sorted(petri_net.postset(transition), key=lambda item: item.name):
            viz.edge(
                _node_id("transition", transition.name),
                _node_id("place", place.name),
            )

    if initial_marking is not None:
        with viz.subgraph() as subgraph:
            subgraph.attr(rank="source")
            for place, _count in sorted(initial_marking.items(), key=lambda item: item[0].name):
                subgraph.node(_node_id("place", place.name))

    if final_marking is not None:
        with viz.subgraph() as subgraph:
            subgraph.attr(rank="sink")
            for place, _count in sorted(final_marking.items(), key=lambda item: item[0].name):
                subgraph.node(_node_id("place", place.name))

    return viz


def visualize_petri_net(
    net: PetriNet | WorkflowNet,
    marking: Optional[Marking] = None,
    output_path: str = "./model",
) -> Path:
    """
    Render a Petri net or workflow net to a PDF file.

    Parameters
    ----------
    net:
        Internal ``PetriNet`` or ``WorkflowNet`` object.
    marking:
        Optional ``Marking`` to display as tokens on the places.
    output_path:
        Output path with or without the ``.pdf`` suffix.
    """
    viz = build_petri_net_digraph(net, marking=marking)
    render_stem, final_path = _normalize_output_path(output_path)
    render_stem.parent.mkdir(parents=True, exist_ok=True)

    with open(os.devnull, "w") as stderr_handle, redirect_stderr(stderr_handle):
        viz.render(str(render_stem), format="pdf", cleanup=True)

    print(f"Visualization saved to: {final_path}")
    return final_path
