from core.petri_net import PetriNet, WorkflowNet, marking_from_places
from utils.visualization import build_petri_net_digraph


def test_build_petri_net_digraph_contains_nodes_edges_and_marking():
    net = PetriNet("viz_test")
    p_start = net.add_place("start")
    p_end = net.add_place("end")
    t_a = net.add_transition("t_a", "A")

    net.add_arc_place_to_transition(p_start, t_a)
    net.add_arc_transition_to_place(t_a, p_end)

    workflow = WorkflowNet(
        net=net,
        initial_marking=marking_from_places(p_start),
        final_marking=marking_from_places(p_end),
    )

    dot = build_petri_net_digraph(workflow, marking=workflow.initial_marking)
    source = dot.source

    assert "place__start" in source
    assert "transition__t_a" in source
    assert "place__start -> transition__t_a" in source
    assert "transition__t_a -> place__end" in source
    assert "POINT-SIZE='30'" in source
    assert "rank=source" in source
    assert "rank=sink" in source
