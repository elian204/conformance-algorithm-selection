from core.petri_net import PetriNet, WorkflowNet, marking_from_places
from core.synchronous_product import MoveType, SynchronousProduct
from core.trace_model import build_trace_net


def _build_single_move_sp():
    net = PetriNet("single_move")
    p0 = net.add_place("p0")
    p1 = net.add_place("p1")
    t = net.add_transition("t_model", label="a")
    net.add_arc_place_to_transition(p0, t)
    net.add_arc_transition_to_place(t, p1)

    wf = WorkflowNet(
        net=net,
        initial_marking=marking_from_places(p0),
        final_marking=marking_from_places(p1),
    )
    tn = build_trace_net([])
    sp = SynchronousProduct(wf, tn)
    model_move = next(
        tr for tr in sp.sp_transitions_list
        if tr.move_type == MoveType.MODEL_ONLY and tr.sn_transition == t
    )
    trace_place = next(iter(tn.initial_marking))
    return sp, model_move, p0, p1, trace_place


def test_predecessors_reject_missing_postset_tokens():
    sp, model_move, p0, p1, trace_place = _build_single_move_sp()
    impossible_successor = marking_from_places(trace_place)

    preds = [
        pred for tr, pred, _ in sp.predecessors(impossible_successor)
        if tr == model_move
    ]

    assert preds == []


def test_predecessors_enumerate_unique_multiset_predecessor():
    sp, model_move, p0, p1, trace_place = _build_single_move_sp()
    successor = marking_from_places(p1, trace_place)

    preds = {
        pred for tr, pred, _ in sp.predecessors(successor)
        if tr == model_move
    }

    expected = {marking_from_places(p0, trace_place)}

    assert preds == expected
    for pred in preds:
        assert sp.fire(pred, model_move) == successor


def test_fire_preserves_multiple_tokens_in_same_place():
    net = PetriNet("multiset")
    p0 = net.add_place("p0")
    p1 = net.add_place("p1")
    p2 = net.add_place("p2")

    t_split = net.add_transition("t_split", label="split")
    net.add_arc_place_to_transition(p0, t_split)
    net.add_arc_transition_to_place(t_split, p1)
    net.add_arc_transition_to_place(t_split, p2)

    t_add = net.add_transition("t_add", label="add")
    net.add_arc_place_to_transition(p1, t_add)
    net.add_arc_transition_to_place(t_add, p2)

    marking = marking_from_places(p0)
    marking = net.fire(marking, t_split)
    marking = net.fire(marking, t_add)

    assert marking.count(p2) == 2
    assert marking.total_tokens == 2
