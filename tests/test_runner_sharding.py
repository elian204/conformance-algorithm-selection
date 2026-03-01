"""Tests for deterministic unique-trace sharding."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.runner import _select_shard_trace_hashes


TRACE_FIXTURE = [
    ("t0", ["A", "B"]),
    ("t1", ["A", "B"]),  # duplicate
    ("t2", ["A", "C"]),
    ("t3", ["B", "C"]),
    ("t4", ["D"]),
]


def test_shard_assignment_is_deterministic_and_complete():
    shard_sets = []
    all_freq = None
    total_unique = None

    for shard_index in range(3):
        selected, freq, unique = _select_shard_trace_hashes(
            TRACE_FIXTURE,
            trace_shard_count=3,
            trace_shard_index=shard_index,
        )
        shard_sets.append(selected)
        if all_freq is None:
            all_freq = freq
        if total_unique is None:
            total_unique = unique

    assert total_unique == 4
    assert all_freq is not None
    assert all_freq.most_common(1)[0][1] == 2  # duplicate variant frequency retained

    union = set().union(*shard_sets)
    assert len(union) == 4
    for i, shard_a in enumerate(shard_sets):
        for j, shard_b in enumerate(shard_sets):
            if i >= j:
                continue
            assert shard_a.isdisjoint(shard_b)


def test_invalid_shard_arguments_rejected():
    with pytest.raises(ValueError):
        _select_shard_trace_hashes(TRACE_FIXTURE, trace_shard_count=0, trace_shard_index=0)

    with pytest.raises(ValueError):
        _select_shard_trace_hashes(TRACE_FIXTURE, trace_shard_count=2, trace_shard_index=2)
