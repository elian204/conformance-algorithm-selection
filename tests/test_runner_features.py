"""Tests for runner feature extraction utilities."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.petri_net import PetriNet, WorkflowNet, marking_from_places
from experiments.runner import _compute_model_features, _count_trace_impossible_activities


def test_model_feature_arc_and_degree_stats():
    net = PetriNet("n")
    p0 = net.add_place("p0")
    p1 = net.add_place("p1")
    t = net.add_transition("t", "A")

    net.add_arc_place_to_transition(p0, t)
    net.add_arc_transition_to_place(t, p1)

    wf = WorkflowNet(net=net, initial_marking=marking_from_places(p0), final_marking=marking_from_places(p1))
    f = _compute_model_features(wf)

    assert f["model_places"] == 2
    assert f["model_transitions"] == 1
    assert f["model_arcs"] == 2
    assert f["model_silent_transitions"] == 0
    assert f["model_visible_transitions"] == 1

    assert f["model_place_in_degree_avg"] == 0.5
    assert f["model_place_out_degree_avg"] == 0.5
    assert f["model_place_in_degree_max"] == 1
    assert f["model_place_out_degree_max"] == 1

    assert f["model_transition_in_degree_avg"] == 1.0
    assert f["model_transition_out_degree_avg"] == 1.0
    assert f["model_transition_in_degree_max"] == 1
    assert f["model_transition_out_degree_max"] == 1


def test_trace_impossible_activities_count():
    trace = ["A", "B", "A", "Z"]
    visible = {"A", "B", "C"}
    assert _count_trace_impossible_activities(trace, visible) == 1
