from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from collections.abc import Iterator
from typing import TYPE_CHECKING
import copy
import sys
import weakref

if TYPE_CHECKING:
    from .history import Tomb


class Collection:
    """A list of objects that can hold dead slots: every public view skips them, in slot order."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, items: Iterable[Any] | None = None):
        """Construct a collection, every entry of items live."""

        self._items: list = []  # Raw slots in canonical order, dead ones included.
        self._dead: list[bool] = []  # One flag per slot.
        self._slots: dict[str, int] = {}  # Live guid -> slot.
        self._tombs: dict[int, list[weakref.ref]] = {}  # Weak pins, newest last.
        self._live = 0  # Live count.
        self._count = 0  # Dead slots not yet purged.
        self._low = 0  # Lowest dead slot, where compaction starts.
        self._cursor: tuple[int, int] | None = None  # (read, write) mid-compaction.
        self._positions: list[int] | None = None  # Live slot positions, built lazily.

        if items is not None:
            self.extend(items)

    def __deepcopy__(self, memo) -> Collection:
        """Copy the live entries, compacted and without pins."""

        result = Collection()
        memo[id(self)] = result

        for item in self:
            result.append(copy.deepcopy(item, memo))

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def __len__(self) -> int:
        """Return the number of live entries."""
        return self._live

    def __bool__(self) -> bool:
        """Return whether an entry is live."""
        return self._live > 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate the live entries in slot order."""

        if self._count == 0 and self._cursor is None:
            return iter(self._items)

        return self._iter_live()

    def __getitem__(self, i: int | slice) -> Any:
        """Return the live entry at position i, or a list for a slice; O(n) once after an edit while dead slots exist, then O(1)."""

        if self._count == 0 and self._cursor is None:
            return self._items[i]

        if self._positions is None:
            self._positions = []

            for slot in range(len(self._dead)):
                if not self._dead[slot]:
                    self._positions.append(slot)

        if isinstance(i, slice):
            return [self._items[slot] for slot in self._positions[i]]

        return self._items[self._positions[i]]

    def __contains__(self, item: Any) -> bool:
        """Return whether a live entry equals item."""

        for entry in self:
            if entry == item:
                return True

        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Kernel slots: raw access for Session, History and tests
    # ═══════════════════════════════════════════════════════════════════════════
    def get_slot(self, guid: str) -> int | None:
        """Return the slot of a live guid."""
        return self._slots.get(guid)

    def get_item(self, slot: int) -> Any:
        """Return the entry in a slot, dead or alive."""
        return self._items[slot]

    def set_item(self, slot: int, item: Any) -> None:
        """Put an entry in a slot; a live slot re-indexes its guid."""

        if not self._dead[slot]:
            old = self._items[slot].guid

            if self._slots.get(old) == slot:
                del self._slots[old]

            self._slots[item.guid] = slot

        self._items[slot] = item

    def is_dead(self, slot: int) -> bool:
        """Return whether a slot is dead."""
        return self._dead[slot]

    def set_dead(self, slot: int, dead: bool) -> None:
        """Kill or revive a slot in O(1); a kill unindexes the guid only when it points at this slot."""

        if self._dead[slot] == dead:
            return

        if self._cursor is not None and self._cursor[1] <= slot < self._cursor[0]:
            return

        key = self._items[slot].guid
        self._dead[slot] = dead
        self._positions = None

        if dead:
            if self._slots.get(key) == slot:
                del self._slots[key]

            if self._count == 0 and self._cursor is None:
                self._low = slot

            self._live -= 1
            self._count += 1
            self._low = min(self._low, slot)
        else:
            self._slots[key] = slot
            self._live += 1
            self._count -= 1

    def get_tomb(self, slot: int) -> Tomb | None:
        """Return the newest tomb pinning a slot while a record still holds it."""

        held = _held(self._tombs.get(slot))

        return held[-1] if held else None

    def set_tomb(self, slot: int, tomb: Tomb) -> None:
        """Pin a slot weakly to a tomb and point the tomb at the slot; older pins a record still holds stay."""

        tomb.slot = slot
        pins = []

        for held in _held(self._tombs.get(slot)):
            if held is not tomb:
                pins.append(weakref.ref(held))

        pins.append(weakref.ref(tomb))
        self._tombs[slot] = pins

    def number_of_dead(self) -> int:
        """Return the number of dead slots not yet purged."""
        return self._count

    def number_of_slots(self) -> int:
        """Return the number of raw slots, dead ones included."""
        return len(self._items)

    def is_compacting(self) -> bool:
        """Return whether a compaction is part way."""
        return self._cursor is not None

    # ═══════════════════════════════════════════════════════════════════════════
    # Mutators
    # ═══════════════════════════════════════════════════════════════════════════
    def append(self, item: Any) -> None:
        """Append a live entry and index its guid, O(1) amortised."""

        slot = len(self._items)
        self._slots[item.guid] = slot
        self._items.append(item)
        self._dead.append(False)
        self._live += 1

        if self._positions is not None:
            self._positions.append(slot)

    def extend(self, items: Iterable[Any]) -> None:
        """Append every entry."""

        for item in items:
            self.append(item)

    def clear(self) -> None:
        """Drop every entry and pin."""

        self._items.clear()
        self._dead.clear()
        self._slots.clear()
        self._tombs.clear()
        self._live = 0
        self._count = 0
        self._low = 0
        self._cursor = None
        self._positions = None

    def compact_step(self, work: int) -> int:
        """Purge unpinned dead slots for at most work slots, resuming where the last call stopped; returns the slots examined."""

        if work == 0 or (self._cursor is None and self._count == 0):
            return 0

        if self._cursor is None:
            start = min(self._low, len(self._items))
            self._low = sys.maxsize
            self._cursor = (start, start)

        r, w = self._cursor
        examined = 0
        self._positions = None

        while examined < work and r < len(self._items):
            held = _held(self._tombs.get(r))

            if self._dead[r] and not held:
                self._tombs.pop(r, None)
                self._count -= 1
            else:
                if w != r:
                    self._move(r, w, held)

                if self._dead[w]:
                    self._low = min(self._low, w)

                w += 1

            r += 1
            examined += 1

        if r < len(self._items):
            self._cursor = (r, w)

            return examined

        del self._items[w:]
        del self._dead[w:]
        self._cursor = None
        self._low = min(self._low, w)

        return examined

    def compact(self) -> None:
        """Finish a running compaction, then purge every unpinned dead slot."""

        if self._cursor is not None:
            self.compact_step(sys.maxsize)

        self.compact_step(sys.maxsize)

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other) -> bool:
        """Compare the live entries in order against a list or a Collection."""

        if not isinstance(other, (list, Collection)):
            return NotImplemented

        return len(self) == len(other) and list(self) == list(other)

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> list:
        """Serialize the live entries as a plain list, the JSON shape of a list."""

        out = []

        for item in self:
            out.append(item.__jsondump__())

        return out

    @classmethod
    def __jsonload__(cls, data: list) -> Collection:
        """Deserialize a list, every entry live."""

        from .file_encoders import file_decode_node

        return cls(file_decode_node(data))

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return "Collection(3 live, 0 dead)"."""
        return f"Collection({self._live} live, {self._count} dead)"

    def __repr__(self) -> str:
        """Return "Collection(3 live, 0 dead)"."""
        return str(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # Details
    # ═══════════════════════════════════════════════════════════════════════════
    def _iter_live(self) -> Iterator[Any]:
        """Yield the live entries in slot order."""

        for slot in range(len(self._items)):
            if not self._dead[slot]:
                yield self._items[slot]

    def _move(self, r: int, w: int, held: list[Tomb]) -> None:
        """Move the entry of slot r down to slot w with its dead flag and the pins a record holds."""

        self._items[w], self._items[r] = self._items[r], self._items[w]
        self._dead[w], self._dead[r] = self._dead[r], self._dead[w]
        self._tombs.pop(r, None)

        for tomb in held:
            tomb.slot = w

        if held:
            self._tombs[w] = [weakref.ref(tomb) for tomb in held]

        if not self._dead[w]:
            self._slots[self._items[w].guid] = w


def _held(pins: list[weakref.ref] | None) -> list[Tomb]:
    """The tombs of a slot's pins that a record still holds, oldest first."""

    held = []

    for pin in pins or ():
        tomb = pin()

        if tomb is not None:
            held.append(tomb)

    return held
