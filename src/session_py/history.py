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
    """Return a deep copy that keeps the guid, which duplicate() and a bare deepcopy would mint anew."""

    snapshot = copy.deepcopy(obj)

    if snapshot.guid != obj.guid:
        snapshot.guid = obj.guid

    return snapshot


# ═══════════════════════════════════════════════════════════════════════════
# Records
# ═══════════════════════════════════════════════════════════════════════════


class Tombstone:
    """Everything needed to put one object back into every live table of a session."""

    kind = ""  # "add" or "remove".

    def __init__(
        self,
        guid: str,
        obj: Any,
        collection: str,
        obj_index: int,
        xform: Xform | None,
        parent_guid: str | None,
        index: int,
        node: TreeNode | None,
        attribute: str,
        edges: list[tuple[str, str, bool]],
    ):
        """Construct from every field of the kit."""

        self.guid = guid  # The object's guid; the clone carries the same one.
        self.obj = obj  # A clone() of the object, never the live instance.
        self.collection = collection  # The Objects list it lives in, e.g. "points".
        self.obj_index = obj_index  # Its position in that list, kept for order().
        self.xform = xform  # Its local transform, None when none was set.
        self.parent_guid = parent_guid  # Tree parent name, None when added without one.
        self.index = index  # Its position among the parent's children.
        self.node = node  # Detached tree node with its subtree, None for an add.
        self.attribute = attribute  # Its graph node attribute.
        self.edges = edges  # Incident edges as (guid, attribute, forward).

    def __str__(self) -> str:
        """Return a string representation of the record."""
        return f"{self.kind}({self.guid})"

    def __repr__(self) -> str:
        """Return a string representation of the record for debugging."""
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
        """Construct from the guid and the before and after snapshots."""
        self.guid = guid
        self.before = before
        self.after = after

    def __str__(self) -> str:
        """Return a string representation of the record."""
        return f"replace({self.guid})"

    def __repr__(self) -> str:
        """Return a string representation of the record for debugging."""
        return f"replace({self.guid})"


class XformOp:
    """The local transform under `guid` changed; nullopt on either side means "none set"."""

    kind = "xform"

    def __init__(self, guid: str, before: Xform | None, after: Xform | None):
        """Construct from the guid and the before and after transforms."""
        self.guid = guid
        self.before = before
        self.after = after

    def __str__(self) -> str:
        """Return a string representation of the record."""
        return f"xform({self.guid})"

    def __repr__(self) -> str:
        """Return a string representation of the record for debugging."""
        return f"xform({self.guid})"


class Transaction:
    """One undoable step: a label and the ops it made, in the order they happened."""

    def __init__(self, label: str = "my_transaction"):
        """Construct an empty transaction with a label."""
        self.label = label
        self.ops: list = []

    def __str__(self) -> str:
        """Return a string representation of the transaction."""
        return f"Transaction({self.label}, {len(self.ops)} ops)"

    def __repr__(self) -> str:
        """Return a string representation of the transaction for debugging."""
        return f"Transaction({self.label}, {len(self.ops)} ops)"


# ═══════════════════════════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════════════════════════


class History:
    """CAD-style undo/redo over a Session, in memory only: records exist between `begin` and `commit`, every save purges them."""

    def __init__(self):
        """Construct an empty history."""
        self.undo_stack: list[Transaction] = []  # Committed, oldest first; capped.
        self.redo_stack: list[Transaction] = []  # Undone, cleared on the next commit.
        self.current: Transaction | None = None  # Open transaction, None when closed.

    def can_undo(self) -> bool:
        """Return whether a committed transaction can be undone."""
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """Return whether an undone transaction can be redone."""
        return len(self.redo_stack) > 0

    def depth(self) -> int:
        """Return the number of committed transactions."""
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

    def undo(self, session: Session) -> bool:
        """Revert the newest transaction, ops in reverse order, and park it for redo."""

        self.commit()

        if len(self.undo_stack) == 0:
            return False

        transaction = self.undo_stack.pop()

        for i in range(len(transaction.ops) - 1, -1, -1):
            self._revert(transaction.ops[i], session)

        self.redo_stack.append(transaction)

        return True

    def redo(self, session: Session) -> bool:
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
        """Drop every transaction, open or committed."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current = None

    def _revert(self, op: Any, session: Session) -> None:
        """Undo one op against the session."""

        if op.kind == "add":
            session._detach(op.guid)
        elif op.kind == "remove":
            session._attach(op)
        elif op.kind == "replace":
            session._swap(op.guid, clone(op.before))
        elif op.kind == "xform":
            session._place(op.guid, op.before)

    def _apply(self, op: Any, session: Session) -> None:
        """Redo one op against the session."""

        if op.kind == "add":
            session._attach(op)
        elif op.kind == "remove":
            session._detach(op.guid)
        elif op.kind == "replace":
            session._swap(op.guid, clone(op.after))
        elif op.kind == "xform":
            session._place(op.guid, op.after)

    def __str__(self) -> str:
        """Return a string representation of the history."""
        return f"History({len(self.undo_stack)} undo, {len(self.redo_stack)} redo)"

    def __repr__(self) -> str:
        """Return a string representation of the history for debugging."""
        return f"History({len(self.undo_stack)} undo, {len(self.redo_stack)} redo)"
