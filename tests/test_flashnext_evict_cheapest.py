"""Kept prompt ends: a burst of short requests does not push out a long conversation, and a short request waits for a
live stream rather than dropping a long kept end."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.families.qwen4_exp.cuda import prefixes


class Slot:
    pass


def owner(entries, *, keep=8, live=0, free=()):
    """``entries``: (token count, slot) in kept order, oldest first; every end was last used in that order."""

    own = SimpleNamespace(kept=[(list(range(n)), st, {"pos": n}, None) for n, st in entries], free=list(free), keep=keep,
                          _busy=lambda: set(), live=lambda: live)
    own._drop_kept = lambda st: setattr(own, "kept", [k for k in own.kept if k[1] is not st])
    for k in own.kept:
        prefixes._touch(own, k)
    return own


def test_a_burst_of_short_requests_does_not_evict_the_long_conversation():
    long, a, b, c = Slot(), Slot(), Slot(), Slot()
    own = owner([(8000, long), (100, a), (100, b)], keep=3)
    prefixes.remember(own, list(range(5000, 5100)), c, {"pos": 100}, None)
    assert long in [k[1] for k in own.kept] and a not in [k[1] for k in own.kept]    # the oldest short end went
    assert own.free == [a]


def test_among_ends_of_similar_length_the_least_recently_used_goes():
    x, y, z, new = Slot(), Slot(), Slot(), Slot()
    own = owner([(900, x), (1000, y), (950, z)], keep=3)
    prefixes._touch(own, own.kept[0])                      # x is resumed from again
    prefixes.remember(own, list(range(7000, 7900)), new, {"pos": 900}, None)
    assert [k[1] for k in own.kept] == [x, z, new]


def test_a_short_request_waits_for_a_live_stream_instead_of_dropping_a_long_kept_end():
    long = Slot()
    own = owner([(8000, long)], live=1)
    with pytest.raises(NoRoom, match="waits for a stream"):
        prefixes.slot_for(own, list(range(5000, 5300)), False)
    assert [k[1] for k in own.kept] == [long] and own.free == []


def test_with_no_live_stream_the_slot_is_dropped_as_before():
    long = Slot()
    own = owner([(8000, long)], live=0)
    st, resume, cached = prefixes.slot_for(own, list(range(5000, 5300)), False)
    assert st is long and resume is None and cached == 0 and own.kept == []


def test_a_prompt_comparable_to_the_kept_end_still_takes_its_slot():
    other = Slot()
    own = owner([(400, other)], live=1)
    st, _, cached = prefixes.slot_for(own, list(range(5000, 5300)), False)
    assert st is other and cached == 0


def test_a_short_kept_end_is_dropped_at_once_even_while_streams_are_live():
    other = Slot()
    own = owner([(500, other)], live=2)
    st, _, cached = prefixes.slot_for(own, list(range(5000, 5020)), False)      # 500 >= 4 x 20, but only 500 tokens
    assert st is other and cached == 0 and own.kept == []
