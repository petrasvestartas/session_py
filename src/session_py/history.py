from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from .session import Session
    from .tree import TreeNode
    from .xform import Xform

CAPACITY = 64


def clone(obj: Any) -> Any:
    """A deep copy that KEEPS the guid: a snapshot must still name the object it stands for.
    `duplicate()` mints a fresh guid on purpose, and so does a bare deepcopy of Point and
    Element, which is why the guid is put back after the copy."""
    snapshot = copy.deepcopy(obj)
    if snapshot.guid != obj.guid:
        snapshot.guid = obj.guid
    return snapshot


# ═══════════════════════════════════════════════════════════════════════════
# Records
# ═══════════════════════════════════════════════════════════════════════════

class Tombstone:
    """Everything needed to put ONE object back into every live table of a session.

    Attributes
    ----------
    guid : str
        The object's guid; the clone carries the same one.
    obj : Any
        A `clone()` of the object, never the live instance.
    collection : str
        The `Objects` list it lives in: "points", "lines", ... "components".
    obj_index : int
        Its position in that list, so the `order()` sequence survives a round trip.
    xform : Xform | None
        Its local transform, None when none was set.
    parent_guid : str | None
        Name of its tree parent, None when it was added without one.
    index : int
        Its position among the parent's children.
    node : TreeNode | None
        The detached tree node with its whole subtree, None for an add.
    attribute : str
        Its graph node attribute.
    edges : list[tuple[str, str, bool]]
        Incident graph edges as (other_guid, attribute, forward), forward when the
        object was the edge's v0.
    """

    kind = ""

    def __init__(
        self,
        guid: str,
        obj: Any,
        collection: str,
        obj_index: int,
        xform: "Xform | None",
        parent_guid: str | None,
        index: int,
        node: "TreeNode | None",
        attribute: str,
        edges: list[tuple[str, str, bool]],
    ):
        self.guid = guid
        self.obj = obj
        self.collection = collection
        self.obj_index = obj_index
        self.xform = xform
        self.parent_guid = parent_guid
        self.index = index
        self.node = node
        self.attribute = attribute
        self.edges = edges

    def __str__(self) -> str:
        return f"{self.kind}({self.guid})"

    def __repr__(self) -> str:
        return f"{self.kind}({self.guid}, {self.collection}[{self.obj_index}])"


class AddOp(Tombstone):
    """An object entered the session; undo detaches it, redo attaches the kit again."""

    kind = "add"


class RemoveOp(Tombstone):
    """An object left the session; the kit is what brings it back on undo."""

    kind = "remove"


class ReplaceOp:
    """The object under `guid` was swapped: absolute before/after snapshots, never deltas."""

    kind = "replace"

    def __init__(self, guid: str, before: Any, after: Any):
        self.guid = guid
        self.before = before
        self.after = after

    def __str__(self) -> str:
        return f"replace({self.guid})"

    def __repr__(self) -> str:
        return f"replace({self.guid})"


class XformOp:
    """The local transform under `guid` changed; None on either side means "none set"."""

    kind = "xform"

    def __init__(self, guid: str, before: "Xform | None", after: "Xform | None"):
        self.guid = guid
        self.before = before
        self.after = after

    def __str__(self) -> str:
        return f"xform({self.guid})"

    def __repr__(self) -> str:
        return f"xform({self.guid})"


class Transaction:
    """One undoable step: a label and the ops it made, in the order they happened."""

    def __init__(self, label: str = "my_transaction"):
        self.label = label
        self.ops: list = []

    def __str__(self) -> str:
        return f"Transaction({self.label}, {len(self.ops)} ops)"

    def __repr__(self) -> str:
        return f"Transaction({self.label}, {len(self.ops)} ops)"


# ═══════════════════════════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════════════════════════

class History:
    """CAD-style undo/redo over a Session, in memory only.

    A removed object leaves every live table at once; its RemoveOp is the tombstone that
    carries the resurrection kit. Records are only written while a transaction is open
    (`begin` ... `commit`), and every save purges the buffer, as Rhino does: history never
    crosses pb or JSON, and a loaded session starts with an empty one.

    Attributes
    ----------
    undo_stack : list[Transaction]
        Committed transactions, oldest first; capped at CAPACITY, the oldest dropped.
    redo_stack : list[Transaction]
        Undone transactions, cleared the moment a new transaction commits.
    current : Transaction | None
        The open transaction, None between `commit` and the next `begin`.
    """

    def __init__(self):
        self.undo_stack: list[Transaction] = []
        self.redo_stack: list[Transaction] = []
        self.current: Transaction | None = None

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def depth(self) -> int:
        return len(self.undo_stack)

    def begin(self, label: str) -> None:
        """Open a transaction; an already open one is committed first so no op is lost."""
        self.commit()
        self.current = Transaction(label)

    def commit(self) -> None:
        """Close the open transaction. An empty one is dropped; a real one clears redo."""
        transaction = self.current
        self.current = None
        if transaction is None or len(transaction.ops) == 0:
            return
        self.undo_stack.append(transaction)
        self.redo_stack.clear()
        if len(self.undo_stack) > CAPACITY:
            self.undo_stack.pop(0)

    def record(self, op: Any) -> None:
        """Append an op to the open transaction; a no-op when none is open."""
        if self.current is None:
            return
        self.current.ops.append(op)

    def undo(self, session: "Session") -> bool:
        """Revert the newest transaction, ops in reverse order, and park it for redo."""
        self.commit()
        if len(self.undo_stack) == 0:
            return False
        transaction = self.undo_stack.pop()
        for i in range(len(transaction.ops) - 1, -1, -1):
            self._revert(transaction.ops[i], session)
        self.redo_stack.append(transaction)
        return True

    def redo(self, session: "Session") -> bool:
        """Re-apply the newest undone transaction, ops in their original order."""
        self.commit()
        if len(self.redo_stack) == 0:
            return False
        transaction = self.redo_stack.pop()
        for i in range(len(transaction.ops)):
            self._apply(transaction.ops[i], session)
        self.undo_stack.append(transaction)
        return True

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current = None

    def _revert(self, op: Any, session: "Session") -> None:
        if op.kind == "add":
            session._detach(op.guid)
        elif op.kind == "remove":
            session._attach(op)
        elif op.kind == "replace":
            session._swap(op.guid, clone(op.before))
        elif op.kind == "xform":
            session._place(op.guid, op.before)

    def _apply(self, op: Any, session: "Session") -> None:
        if op.kind == "add":
            session._attach(op)
        elif op.kind == "remove":
            session._detach(op.guid)
        elif op.kind == "replace":
            session._swap(op.guid, clone(op.after))
        elif op.kind == "xform":
            session._place(op.guid, op.after)

    def __str__(self) -> str:
        return f"History({len(self.undo_stack)} undo, {len(self.redo_stack)} redo)"

    def __repr__(self) -> str:
        return f"History({len(self.undo_stack)} undo, {len(self.redo_stack)} redo)"
