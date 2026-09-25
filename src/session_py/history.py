from __future__ import annotations
from typing import Any
from typing import TYPE_CHECKING
import copy
from .brep import BRep
from .element import Element
from .instance_ref import InstanceRef
from .line import Line
from .mesh import Mesh
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .obb import OBB
from .plane import Plane
from .point import Point
from .pointcloud import PointCloud
from .polyline import Polyline

if TYPE_CHECKING:
    from .color import Color
    from .session import Session
    from .tree import TreeNode
    from .xform import Xform

CAPACITY = 64  # Committed transactions kept; past it the oldest is dropped.
BUDGET = 256 << 20  # Bytes the stacks may pin; past it the oldest is dropped.
RECORD = 256  # Bytes one record costs on top of what it pins.


def clone(obj: Any) -> Any:
    """A deep copy that keeps the guid, feature guids included, which `duplicate()` and a bare deepcopy would mint anew."""

    snapshot = copy.deepcopy(obj)

    if isinstance(obj, list):
        for i in range(len(obj)):
            if obj[i].has_guid():
                snapshot[i].guid = obj[i].guid

        return snapshot

    if snapshot.guid != obj.guid:
        snapshot.guid = obj.guid

    if isinstance(obj, Element):
        snapshot.set_features(clone(obj.features))

    if isinstance(obj, InstanceRef):
        snapshot.features = clone(obj.features)

    return snapshot


def _mesh_weight(mesh: Mesh) -> int:
    """Bytes a mesh pins, from its counts."""
    return 128 + 64 * mesh.number_of_vertices() + 48 * mesh.number_of_faces()


def _brep_weight(brep: BRep) -> int:
    """Bytes a brep pins, from its table lengths."""

    return (
        512
        + 256 * len(brep.m_surfaces)
        + 128 * (len(brep.m_curves_3d) + len(brep.m_curves_2d))
        + 24 * len(brep.m_vertices)
        + 64 * (len(brep.m_edges) + len(brep.m_faces))
    )


def weight(item: Any) -> int:
    """An estimate of the bytes an item pins while a record holds it, O(1) from its container lengths."""

    if isinstance(item, Point):
        return 64

    if isinstance(item, Line):
        return 96

    if isinstance(item, Plane):
        return 160

    if isinstance(item, OBB):
        return 192

    if isinstance(item, Polyline):
        return 64 + 24 * item.point_count()

    if isinstance(item, PointCloud):
        return (
            64
            + 24 * item.point_count()
            + 24 * item.normal_count()
            + 16 * item.color_count()
        )

    if isinstance(item, Mesh):
        return _mesh_weight(item)

    if isinstance(item, NurbsCurve):
        return 96 + 32 * item.cv_count() + 8 * len(item.m_nurbsknot)

    if isinstance(item, NurbsSurface):
        return (
            128
            + 32 * item.cv_count()
            + 8 * (len(item.m_nurbsknot[0]) + len(item.m_nurbsknot[1]))
        )

    if isinstance(item, BRep):
        return _brep_weight(item)

    if isinstance(item, Element):
        geometry = item.geometry
        bytes = 0

        if isinstance(geometry, Mesh):
            bytes = _mesh_weight(geometry)
        elif isinstance(geometry, BRep):
            bytes = _brep_weight(geometry)

        return 256 + bytes + 128 * len(item.features)

    if isinstance(item, InstanceRef):
        return 256 + 128 * len(item.features)

    return 128


# ═══════════════════════════════════════════════════════════════════════════
# Records
# ═══════════════════════════════════════════════════════════════════════════
class Tomb:
    """One dead or revivable entity: where it lives and what it parked while dead; slots and nodes pin it weakly, records strongly."""

    __slots__ = (
        "collection",
        "definition",
        "slot",
        "node",
        "vertex",
        "edges",
        "xform",
        "interactions",
        "__weakref__",
    )

    def __init__(
        self, collection: str, definition: bool, slot: int, node: TreeNode | None
    ):
        """Construct a tomb with nothing parked."""

        self.collection = collection  # Its Objects list, "" for a node-only tomb.
        self.definition = definition  # Whether the slot is in Session.definitions.
        self.slot = slot  # Its raw slot, moved by compaction.
        self.node = node  # Its tree node, None for a slot-only tomb.
        self.vertex = None  # Its graph vertex while dead.
        self.edges = []  # Its incident edges while dead.
        self.xform = None  # Its local transform while dead.
        self.interactions = {}  # Its edges' interactions while dead, by edge guid.


class Tombstone:
    """An object added or removed: the tomb that flips it and where its node sits."""

    kind = ""  # "add" or "remove".

    def __init__(
        self,
        guid: str,
        collection: str,
        parent_guid: str | None,
        index: int,
        node: TreeNode | None,
        tomb: Tomb,
    ):
        """Construct from every field of the record."""

        self.guid = guid  # The object's guid.
        self.collection = collection  # The Objects list it lives in, or "definitions".
        self.parent_guid = parent_guid  # Name of its tree parent, None without a node.
        self.index = index  # Its raw index among the parent's children, a hint.
        self.node = node  # Its tree node, for adds too; None when it has none.
        self.tomb = tomb  # The tomb undo and redo flip.

    def __str__(self) -> str:
        """Return a string representation of the record."""
        return f"{self.kind}({self.guid})"

    def __repr__(self) -> str:
        """Return a string representation of the record for debugging."""
        return f"{self.kind}({self.guid}, {self.collection})"


class AddOp(Tombstone):
    """An object entered the session; undo kills its tomb, redo revives it."""

    kind = "add"  # Always "add".


class RemoveOp(Tombstone):
    """An object left the session; undo revives its tomb, redo kills it."""

    kind = "remove"  # Always "remove".


class Entry:
    """The entry a replace was taken on: an object by its tree node at record time (None outside the tree) or a definition by the tomb pinning its slot."""

    def __init__(self, definition: bool, node: TreeNode | None, tomb: Tomb | None):
        """Construct an object entry from its node or a definition entry from its slot's tomb."""

        self.definition = definition  # Whether the entry is a definition.
        self.node = node  # An object's tree node at record time; None outside the tree or for a definition.
        self.tomb = tomb  # A definition's slot-only tomb, moved with its slot by compaction; None for an object.


class ReplaceOp:
    """The object or definition under `guid` was swapped: the stored objects before and after, never copies."""

    kind = "replace"  # Always "replace".

    def __init__(self, guid: str, before: Any, after: Any, entry: Entry):
        """Construct from the guid, the before and after items and the entry."""

        self.guid = guid  # The entry's guid.
        self.before = before  # The entry before the swap.
        self.after = after  # The entry after the swap.
        self.entry = entry  # The entry the swap was taken on.

    def __str__(self) -> str:
        """Return a string representation of the record."""
        return f"replace({self.guid})"

    def __repr__(self) -> str:
        """Return a string representation of the record for debugging."""
        return f"replace({self.guid})"


class XformOp:
    """The local transform under `guid` changed; None on either side means "none set"."""

    kind = "xform"  # Always "xform".

    def __init__(
        self,
        guid: str,
        before: Xform | None,
        after: Xform | None,
        node: TreeNode | None,
    ):
        """Construct from the guid, the before and after transforms and the entry's node."""

        self.guid = guid  # The object's guid.
        self.before = before  # Transform before the change.
        self.after = after  # Transform after the change.
        self.node = node  # The entry's tree node at record time; None for a group or an object outside the tree.

    def __str__(self) -> str:
        """Return a string representation of the record."""
        return f"xform({self.guid})"

    def __repr__(self) -> str:
        """Return a string representation of the record for debugging."""
        return f"xform({self.guid})"


class TreeOp:
    """A tree node added, removed, moved, renamed or recoloured: its state before and after."""

    kind = "tree"  # Always "tree".

    def __init__(
        self,
        guid: str,
        node: TreeNode,
        tomb: Tomb,
        ghost: TreeNode | None,
        name_before: str,
        name_after: str,
        color_before: Color | None,
        color_after: Color | None,
        dead_before: bool,
        dead_after: bool,
    ):
        """Construct from every field of the record."""

        self.guid = guid  # The node name at record time.
        self.node = node  # The node itself.
        self.tomb = tomb  # Node-only; pins the ghost of a move, else the node.
        self.ghost = ghost  # The dead ghost a move left in the old slot.
        self.name_before = name_before  # Name before.
        self.name_after = name_after  # Name after.
        self.color_before = color_before  # Colour before.
        self.color_after = color_after  # Colour after.
        self.dead_before = dead_before  # Whether it was dead or absent before.
        self.dead_after = dead_after  # Whether it is dead after.

    def __str__(self) -> str:
        """Return a string representation of the record."""
        return f"tree({self.guid})"

    def __repr__(self) -> str:
        """Return a string representation of the record for debugging."""
        return f"tree({self.guid})"


class Transaction:
    """One undoable step: a label, the ops it made in the order they happened, and the bytes they pin."""

    def __init__(self, label: str = "my_transaction"):
        """Construct an empty transaction with a label."""

        self.label = label  # What the step did.
        self.ops: list = []  # Ops in the order they happened.
        self.bytes = 0  # Bytes its records pin.

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
    """CAD-style undo/redo over a Session, in memory only: records flip tombs in place, every save purges them."""

    def __init__(self):
        """Construct an empty history with the default budget."""

        self.undo_stack: list[Transaction] = []  # Committed, oldest first, capped.
        self.redo_stack: list[Transaction] = []  # Undone, cleared on the next commit.
        self.current: Transaction | None = None  # Open transaction, None when closed.
        self.bytes = 0  # Bytes pinned by both stacks and the open transaction.
        self.budget = BUDGET  # Bytes the stacks may pin before the oldest is dropped.
        self.dropped = 0  # Ops dropped since the last purge cycle began, unrecorded kills included.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def can_undo(self) -> bool:
        """Return whether a committed transaction can be undone."""
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        """Return whether an undone transaction can be redone."""
        return len(self.redo_stack) > 0

    def depth(self) -> int:
        """Return the number of committed transactions."""
        return len(self.undo_stack)

    # ═══════════════════════════════════════════════════════════════════════════
    # Transactions
    # ═══════════════════════════════════════════════════════════════════════════
    def begin(self, label: str) -> None:
        """Open a transaction; an already open one is committed first so no op is lost."""

        self.commit()
        self.current = Transaction(label)

    def commit(self) -> None:
        """Close the open transaction. An empty one is dropped; a real one clears redo and trims the oldest past the caps."""

        transaction = self.current
        self.current = None

        if transaction is None or len(transaction.ops) == 0:
            return

        self.undo_stack.append(transaction)

        for undone in self.redo_stack:
            self.dropped += len(undone.ops)

        self.redo_stack.clear()
        self.bytes = self._pinned()

        while len(self.undo_stack) > 1 and (
            len(self.undo_stack) > CAPACITY or self.bytes > self.budget
        ):
            oldest = self.undo_stack.pop(0)
            self.dropped += len(oldest.ops)
            self.bytes -= oldest.bytes

    def record(self, op: Any, bytes: int) -> None:
        """Append an op pinning `bytes` to the open transaction; a no-op when none is open."""

        if self.current is None:
            return

        self.current.ops.append(op)
        self.current.bytes += bytes
        self.bytes += bytes

    def abort(self, session: Session) -> bool:
        """Revert the open transaction's ops in reverse and drop it, leaving both stacks as they are; False when none is open."""

        transaction = self.current
        self.current = None

        if transaction is None:
            return False

        for i in range(len(transaction.ops) - 1, -1, -1):
            self._revert(transaction.ops[i], session)

        self.dropped += len(transaction.ops)
        self.bytes = self._pinned()

        return True

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
        """Drop every transaction, open or committed; what they pinned is purgeable now."""

        for transaction in self.undo_stack:
            self.dropped += len(transaction.ops)

        for transaction in self.redo_stack:
            self.dropped += len(transaction.ops)

        if self.current is not None:
            self.dropped += len(self.current.ops)

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current = None
        self.bytes = 0

    def _pinned(self) -> int:
        """Bytes pinned by both stacks."""

        pinned = 0

        for transaction in self.undo_stack:
            pinned += transaction.bytes

        for transaction in self.redo_stack:
            pinned += transaction.bytes

        return pinned

    def _revert(self, op: Any, session: Session) -> None:
        """Undo one op against the session."""

        if op.kind == "add":
            session._kill(op.tomb)
        elif op.kind == "remove":
            session._revive(op.tomb)
        elif op.kind == "replace":
            session._swap(op.guid, op.before, op.entry)
        elif op.kind == "xform":
            session._place(op.guid, op.before, op.node)
        elif op.kind == "tree":
            session._tree(op, True)

    def _apply(self, op: Any, session: Session) -> None:
        """Redo one op against the session."""

        if op.kind == "add":
            session._revive(op.tomb)
        elif op.kind == "remove":
            session._kill(op.tomb)
        elif op.kind == "replace":
            session._swap(op.guid, op.after, op.entry)
        elif op.kind == "xform":
            session._place(op.guid, op.after, op.node)
        elif op.kind == "tree":
            session._tree(op, False)

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return a string representation of the history."""
        return f"History({len(self.undo_stack)} undo, {len(self.redo_stack)} redo)"

    def __repr__(self) -> str:
        """Return a string representation of the history for debugging."""
        return f"History({len(self.undo_stack)} undo, {len(self.redo_stack)} redo)"
