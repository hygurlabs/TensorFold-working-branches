"""Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy."""

from tensorfold.cuda.memory_gate import NoRoom


def _best(kept, prompt, busy=()):
    return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
                and prompt[:len(k[0])] == k[0]), key=lambda k: len(k[0]), default=None)


def _longer(kept, entry):
    return any(k[1] is entry[1] and len(k[0]) > len(entry[0]) for k in kept)


def slot_for(owner, prompt: list[int], reuse: bool):
    """Copy a fork into spare capacity; otherwise retain the released idle-slot and memory-pressure behavior."""

    busy = owner._busy()
    best = _best(owner.kept, prompt) if reuse else None
    fork = best is not None and (id(best[1]) in busy or _longer(owner.kept, best))
    if fork:
        if owner.free:
            spare = owner.free.pop()
            try:
                if owner._grow(spare, len(prompt) + owner.depth + 2, protect=best[1]):
                    spare.copy_prefix(best[1], len(best[0]), best[2]["mtp_len"])
                    return spare, {"state": best[2], "tail": best[3]}, len(best[0])
            except Exception:
                owner.free.append(spare)
                owner._shrink(spare, force=True)
                raise
            owner.free.append(spare)
        best = _best(owner.kept, prompt, busy) if reuse else None
        if best is not None and owner.free and _longer(owner.kept, best):
            best = None
    if best is not None:
        n = len(best[0])
        for k in owner.kept:
            if k[1] is best[1]:
                _touch(owner, k)                         # resumed from: recently used again
        owner.kept = [k for k in owner.kept if k[1] is not best[1]
                      or len(k[0]) <= n and best[0][:len(k[0])] == k[0]]
        return best[1], {"state": best[2], "tail": best[3]}, n
    if not owner.free:
        idle = cheapest_slot(owner, [k for k in owner.kept if id(k[1]) not in busy])
        if idle is None:
            raise RuntimeError("no free stream slot")
        cost = max(len(k[0]) for k in owner.kept if k[1] is idle)
        if cost >= PROTECT * len(prompt) and owner.live():
            # Waiting for a live stream to end costs this prompt a few rounds; dropping the slot would make that
            # conversation's next turn prefill ``cost`` tokens again (a 32k-token chain: 15-20 s on one GB10).
            raise NoRoom(f"a {len(prompt)}-token prompt waits for a stream to end rather than drop a "
                         f"{cost}-token kept prompt")
        owner._drop_kept(idle)
        owner.free.append(idle)
    return owner.free.pop(), None, 0


# A slot whose kept chain costs PROTECT times the incoming prompt or more is not dropped to admit it while a live
# stream will free a slot: short requests (routing calls, a few hundred tokens) wait instead of evicting conversations.
PROTECT = 4


# Which kept prompt end goes: the least recently used among those PROTECT times shorter than the longest kept, else
# the least recently used. Oldest-first let a burst of short requests push out a long conversation; fewest-tokens-first
# never let an abandoned long one go; GreedyDual-Size (clock + tokens) still dropped the current conversation, its
# clock jumping by a whole stale chain at each eviction.
def _touch(owner, entry) -> None:
    if not hasattr(owner, "kept_used"):
        owner.kept_used, owner.kept_tick = {}, 0
    owner.kept_tick += 1
    owner.kept_used[(id(entry[1]), len(entry[0]))] = owner.kept_tick


def _used(owner, entry) -> int:
    return getattr(owner, "kept_used", {}).get((id(entry[1]), len(entry[0])), 0)


def _pick(candidates: list):
    """``candidates``: (tokens, last use, item); the item to evict, None when there is none."""

    if not candidates:
        return None
    cutoff = max(c[0] for c in candidates) / PROTECT
    short = [c for c in candidates if c[0] < cutoff] or candidates
    return min(short, key=lambda c: c[1])[2]             # min() keeps the first, i.e. the oldest kept, of equal uses


def cheapest(owner, kept: list, entry: bool = False):
    """The kept prompt end to evict (see above); its slot unless ``entry``."""

    k = _pick([(len(k[0]), _used(owner, k), k) for k in kept])
    return k if entry or k is None else k[1]


def cheapest_slot(owner, kept: list):
    """The idle slot to evict, a slot weighing its longest kept end and its latest use."""

    slots: dict[int, list] = {}
    for k in kept:
        c = slots.setdefault(id(k[1]), [0, 0, k[1]])
        c[0], c[1] = max(c[0], len(k[0])), max(c[1], _used(owner, k))
    return _pick(list(slots.values()))


def remember(owner, ids, st, snap, tail) -> None:
    """Keep each slot's prefix chain, returning displaced idle slots to the free list."""

    gone = [k[1] for k in owner.kept if k[0] == ids]
    owner.kept = [k for k in owner.kept if k[0] != ids] + [(ids, st, snap, tail)]
    _touch(owner, owner.kept[-1])
    while len(owner.kept) > owner.keep:
        k = cheapest(owner, owner.kept[:-1], entry=True)
        owner.kept = [e for e in owner.kept if e is not k]     # by identity: entries hold tensors
        gone.append(k[1])
    live = {(id(k[1]), len(k[0])) for k in owner.kept}
    for key in [key for key in owner.kept_used if key not in live]:
        del owner.kept_used[key]
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)
