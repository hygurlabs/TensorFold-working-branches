"""With no spare slot, a fork takes the idle slot that holds the fewest kept tokens instead of cutting the shared chain."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


class Slot:
    def __init__(self):
        self.copied = None

    def copy_prefix(self, source, pos, mtp_len):
        self.copied = (source, pos, mtp_len)


def decoder(other_tokens, source_tokens=600, busy_source=False):
    dec = object.__new__(MultiDecoder)
    dec.solo, dec.planning = None, False
    dec.w = SimpleNamespace(comm=None)
    source, other = Slot(), Slot()
    shared = list(range(300))
    snap = {"pos": 300, "mtp_len": 299}
    dec.kept = [(shared, source, snap, "tail"), (shared + list(range(1000, 1000 + source_tokens - 300)), source, {"pos": 1}, "a"),
                (list(range(5000, 5000 + other_tokens)), other, {"pos": 2}, "b")]
    dec.free, dec.filling, dec.fills = [], [], {}
    dec.streams = {1: SimpleNamespace(st=source, waiting=False)} if busy_source else {}
    dec.depth, dec.capacity = 3, 4096
    dec._grow = lambda st, rows, **kwargs: True
    return dec, source, other, shared, snap


def test_the_idle_slot_that_costs_less_takes_the_fork_and_the_shared_chain_stays():
    dec, source, other, shared, snap = decoder(other_tokens=100)
    st, resume, cached = dec._slot_for(shared + [7, 8, 9], True)
    assert st is other and cached == len(shared) and resume == {"state": snap, "tail": "tail"}
    assert other.copied == (source, len(shared), 299)
    assert [k[1] for k in dec.kept] == [source, source] and dec.free == []


def test_resuming_in_place_when_the_idle_slot_would_cost_more():
    dec, source, other, shared, _ = decoder(other_tokens=500)
    st, _, cached = dec._slot_for(shared + [7, 8, 9], True)
    assert st is source and cached == len(shared) and source.copied is None
    assert [k[1] for k in dec.kept] == [source, other] and len(dec.kept[0][0]) == len(shared)


def test_a_busy_shared_slot_is_copied_into_the_idle_slot_that_costs_least():
    dec, source, other, shared, _ = decoder(other_tokens=100, busy_source=True)
    st, _, cached = dec._slot_for(shared + [7, 8, 9], True)
    assert st is other and cached == len(shared) and other.copied == (source, len(shared), 299)
