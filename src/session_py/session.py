from __future__ import annotations
from typing import Any
from typing import Iterable
from typing import NamedTuple
from typing import TYPE_CHECKING
import bisect
import copy
import heapq
import itertools
import json
import sys
import uuid
import weakref
from .collection import Collection
from .objects import Objects
from .objects import Component
from .point import Point
from .vector import Vector
from .line import Line
from .plane import Plane
from .obb import OBB
from .polyline import Polyline
from .pointcloud import PointCloud
from .mesh import Mesh
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .brep import BRep
from .element import Element
from .instance_ref import InstanceRef
from .interaction import Interaction
from .tree import Tree
from .tree import TreeNode
from .tree import node_head
from .tree import node_tail
from .graph import Graph
from .graph import edge_to_proto
from .graph import vertex_to_proto
from .history import History
from .history import AddOp
from .history import Entry
from .history import RemoveOp
from .history import ReplaceOp
from .history import XformOp
from .history import TreeOp
from .history import Tomb
from .history import RECORD
from .history import clone
from .history import weight
from .spatial_bvh import SpatialBVH
from .tolerance import Tolerance
from .xform import Xform
from .intersection import line_line
from .intersection import line_plane
from .intersection import ray_box
from .intersection import ray_mesh_bvh

if TYPE_CHECKING:
    from pathlib import Path
    from .color import Color
    from .proto import session_pb2


COLLECTIONS = [
    ("points", "point"),
    ("lines", "line"),
    ("planes", "plane"),
    ("bboxes", "bbox"),
    ("polylines", "polyline"),
    ("pointclouds", "pointcloud"),
    ("meshes", "mesh"),
    ("nurbscurves", "nurbscurve"),
    ("nurbssurfaces", "nurbssurface"),
    ("breps", "brep"),
    ("elements", "element"),
    ("components", "component"),
    ("instances", "instance"),
]

PURGE_WORK = 1024  # Work units of one idle purge or checkpoint step, about 2 ms: one raw slot, child, vertex or entry each.

_HEAD = 0  # Checkpoint phase: the session name and guid.
_OBJECTS = 1  # Checkpoint phases 1..13: the objects lists.
_TREE = 14  # Checkpoint phase: the tree, depth first.
_VERTICES = 15  # Checkpoint phase: the graph vertices.
_EDGES = 16  # Checkpoint phase: the graph edges.
_ORDERED = 17  # Checkpoint phases 17..27: xforms in order() sequence.
_REST = 28  # Checkpoint phase: xforms outside order(), by guid.
_DEFINITIONS = 29  # Checkpoint phases 29..41: the definitions lists.
_INTERACTIONS = 42  # Checkpoint phase: the interactions, by edge guid.
_ASSEMBLY = 43  # Checkpoint phase: the sections joined into one message.
_CHUNK = 64 << 10  # Bytes past which a finished node's chunks move whole.
_LENGTH_DELIMITED = 2  # Protobuf wire type of strings, bytes and messages.
_SECTIONS = (  # Checkpoint buffers in Session field order.
    "head",
    "objects",
    "tree",
    "graph",
    "xforms",
    "definitions",
    "interactions",
)
_FRAMED = ("objects", "tree", "graph", "definitions")  # Framed as one field.


class RayHit(NamedTuple):
    """One object a ray touched: which one, where, and how far from the ray origin."""

    guid: str
    hit_point: Point
    distance: float


def _clone_objects(objects: Objects) -> Objects:
    """A deep copy of every list and every object in it, guids included."""

    out = copy.deepcopy(objects)

    for collection, _ in COLLECTIONS:
        for src, dst in zip(getattr(objects, collection), getattr(out, collection)):
            dst.guid = src.guid

    return out


def _collection_of(geometry: Any) -> tuple[str, str]:
    """The COLLECTIONS entry whose list holds the type of geometry."""

    types = (
        Point,
        Line,
        Plane,
        OBB,
        Polyline,
        PointCloud,
        Mesh,
        NurbsCurve,
        NurbsSurface,
        BRep,
        Element,
    )

    for entry, cls in zip(COLLECTIONS, types):
        if isinstance(geometry, cls):
            return entry

    return "", ""


def _collection_for(item: Any) -> tuple[str, str]:
    """The COLLECTIONS entry of any stored item: geometry, instance or component."""

    if isinstance(item, InstanceRef):
        return COLLECTIONS[12]

    if isinstance(item, Component):
        return COLLECTIONS[11]

    return _collection_of(item)


def _prefix_of(collection: str) -> str:
    """The vertex label prefix of a list name, "" for an unknown one."""

    for name, prefix in COLLECTIONS:
        if name == collection:
            return prefix

    return ""


def _repoint(items: Collection, lookup: dict[str, Any]) -> None:
    """Point every live slot whose guid lookup holds with another value at the lookup value, and index every live slot lookup lacks."""

    for slot in range(items.number_of_slots()):
        if items.is_dead(slot):
            continue

        item = items.get_item(slot)
        held = lookup.get(item.guid)

        if held is None:
            lookup[item.guid] = item
        elif held is not item:
            items.set_item(slot, held)


def _adopt(objects: Objects, lookup: dict[str, Any], collection: str) -> None:
    """Append every value only lookup holds, in guid order, to collection or, when "", to the list of its type."""

    orphans = []

    for guid, item in lookup.items():
        name = collection if collection else _collection_of(item)[0]

        if getattr(objects, name).get_slot(guid) is None:
            orphans.append(guid)

    for guid in sorted(orphans):
        name = collection if collection else _collection_of(lookup[guid])[0]
        getattr(objects, name).append(lookup[guid])


def _place(geometry: Any, xform: Xform) -> None:
    """Move geometry in place: an element is placed, anything else transformed; identity leaves it untouched."""

    if xform.is_identity():
        return

    if isinstance(geometry, Element):
        geometry.place(xform)
    else:
        geometry.transform(xform)


def _resolve(instance: InstanceRef, definition: Any, xform: Xform) -> Any:
    """The definition copied as the instance, its guid and name and on an element its features, then moved by xform."""

    result = clone(definition)
    result.guid = instance.guid
    result.name = instance.name

    if isinstance(result, Element):
        for feature in clone(instance.features):
            result.add_feature(feature)

    _place(result, xform)

    return result


def _bake(items: list, world: dict[str, Xform]) -> None:
    """Transforms every object of a list by its world placement, identity entries untouched."""

    for item in items:
        xform = world.get(item.guid)

        if xform is None or xform.is_identity():
            continue

        item.transform(xform)


def _placed_box(points: list[Point], xform: Xform, inflate: float) -> OBB:
    """Inflated box around the placed points, around the origin when there are none."""

    if len(points) == 0:
        return OBB.from_point(Point(0, 0, 0), inflate)

    placed = []

    for point in points:
        placed.append(xform.transform_point(point))

    return OBB.from_points(placed, inflate)


def _ray_point(ray: Line, point: Point, tolerance: float) -> Point | None:
    """The point on the ray closest to point when it lies ahead and within tolerance."""

    ray_dir = ray.end() - ray.start()
    to_point = point - ray.start()
    t = to_point.dot(ray_dir) / ray_dir.dot(ray_dir)

    if t < 0:
        return None

    closest = ray.start() + ray_dir * t

    if point.distance(closest) > tolerance:
        return None

    return closest


def _ray_polyline(ray: Line, polyline: Polyline, tolerance: float) -> Point | None:
    """The segment hit closest to the ray start."""

    closest = None
    min_dist = float("inf")

    for i in range(polyline.segment_count()):
        segment = Line.from_points(polyline.get_point(i), polyline.get_point(i + 1))
        hit = line_line(ray, segment, tolerance)

        if hit is None:
            continue

        dist = ray.start().distance(hit)

        if dist < min_dist:
            min_dist = dist
            closest = hit

    return closest


def _ray_pointcloud(
    ray: Line, pointcloud: PointCloud, tolerance: float
) -> Point | None:
    """The ray point closest to a cloud point within tolerance."""

    closest = None
    min_dist = float("inf")

    for point in pointcloud.get_points():
        hit = _ray_point(ray, point, tolerance)

        if hit is None:
            continue

        dist = point.distance(hit)

        if dist < min_dist:
            min_dist = dist
            closest = hit

    return closest


def _ray_mesh(
    ray: Line, mesh: Mesh, tolerance: float, placement: Xform
) -> Point | None:
    """The first hit of the ray on the placed mesh, tested in the mesh frame."""

    inverse = placement.inverse()

    if inverse is None:
        return None

    local_ray = Line.from_points(
        inverse.transform_point(ray.start()), inverse.transform_point(ray.end())
    )
    hits = ray_mesh_bvh(local_ray, mesh, tolerance, True)

    if not hits:
        return None

    return placement.transform_point(hits[0])


def _box_points(geometry: Any) -> list[Point]:
    """The points whose box bounds a geometry: vertices, control points or surface samples."""

    points = []

    if isinstance(geometry, Line):
        points.append(geometry.start())
        points.append(geometry.end())
    elif isinstance(geometry, (Polyline, PointCloud)):
        points = geometry.get_points()
    elif isinstance(geometry, Mesh):
        for vertex in geometry.vertex.values():
            points.append(vertex.position())
    elif isinstance(geometry, BRep):
        for vertex in geometry.m_vertices:
            points.append(vertex.point)

        for surface in geometry.m_surfaces:
            u0, u1 = surface.domain(0)
            v0, v1 = surface.domain(1)

            for i in range(3):
                for j in range(3):
                    points.append(
                        surface.point_at(
                            u0 + (u1 - u0) * i / 2.0, v0 + (v1 - v0) * j / 2.0
                        )
                    )
    elif isinstance(geometry, NurbsCurve):
        for i in range(geometry.cv_count()):
            points.append(geometry.get_cv(i))
    elif isinstance(geometry, NurbsSurface):
        for i in range(geometry.cv_count(0)):
            for j in range(geometry.cv_count(1)):
                points.append(geometry.get_cv(i, j))

    return points


def _varint(value: int) -> bytearray:
    """The protobuf varint encoding of a non-negative integer."""

    out = bytearray()

    while value >= 0x80:
        out.append(value & 0x7F | 0x80)
        value >>= 7

    out.append(value)

    return out


def _prefix(field: int, length: int) -> bytearray:
    """The key and length of a length-delimited protobuf field."""
    return _varint(field << 3 | _LENGTH_DELIMITED) + _varint(length)


def _append(chunks: list[bytearray], piece: bytearray) -> None:
    """Append bytes to a chunked buffer: a small piece is copied into the last chunk, a large one moves whole."""

    if chunks and len(piece) <= _CHUNK and len(chunks[-1]) < _CHUNK:
        chunks[-1] += piece
    else:
        chunks.append(piece)


def _objects_head(objects: Objects) -> bytearray:
    """The name and guid fields of an Objects message."""

    from .proto import objects_pb2

    proto = objects_pb2.Objects()
    proto.name = objects.name

    if objects.has_guid():
        proto.guid = objects.guid

    return bytearray(proto.SerializeToString(deterministic=True))


def _registered(session: Session, guid: str) -> bool:
    """Whether guid is a graph node held by an object, instance or component."""

    return session.graph.has_node(guid) and (
        guid in session.lookup
        or guid in session.instance_lookup
        or guid in session.component_lookup
    )


class _Checkpoint:
    """A resumable protobuf writer over a session: live entries only, the layout to_proto encodes."""

    def __init__(self, revision: int):
        """Construct a writer at the first phase."""

        self.revision = revision  # The revision it writes.
        self.phase = _HEAD  # The section being written.
        self.cursor = 0  # Slot or key position in the phase.
        self.unsorted = None  # Keys yet to sort, in runs of at most work.
        self.runs = []  # The sorted runs.
        self.keys = None  # The phase's keys in order, merged lazily.
        self.hits = (
            0  # Xforms entries order() reached, every one once the rest scan is done.
        )
        self.rest = []  # Xforms guids outside order(), in guid order.
        self.stack = []  # Tree frames being written, root first: node, next raw child, bytes in chunks.
        self.tree = []  # The framed root, in chunks after the Tree head.
        self.sections = {name: bytearray() for name in _SECTIONS}  # One per field.
        self.pieces = []  # The framed sections in message order.
        self.out = bytearray()  # The joined message.

    def sort_step(self, keys: Iterable[str], work: int) -> int:
        """Sort a snapshot of keys in runs of at most work per call, then merge the runs lazily into self.keys; returns the keys sorted."""

        if self.unsorted is None:
            self.unsorted = list(keys)
            self.runs = []
            self.cursor = 0

        start = self.cursor
        end = min(len(self.unsorted), start + work)
        self.runs.append(sorted(self.unsorted[start:end]))
        self.cursor = end

        if end == len(self.unsorted):
            self.keys = heapq.merge(*self.runs)
            self.unsorted = None
            self.runs = []
            self.cursor = 0

        return end - start


class Session:
    """A session containing geometry objects."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, name: str = "my_session"):
        """Construct an empty session whose tree root carries the session name."""

        self._guid = None
        self.name = name  # The name of the session.
        self.objects = Objects()  # Collection of geometry objects.
        self.lookup: dict[str, Any] = {}  # Fast lookup table for geometry by GUID.
        self.tree = Tree(name=f"{name}_tree")  # Tree structure for hierarchy.
        self.graph = Graph(name=f"{name}_graph")  # Graph structure for relationships.
        self.component_lookup: dict[str, Component] = {}  # Components by GUID.
        self.xforms: dict[str, Xform] = {}  # Guid -> LOCAL transform.
        self.definitions = Objects()  # Shared geometry instances place, each in its own frame; never in order(), the tree, the graph or xforms.
        self.definition_lookup: dict[str, Any] = {}  # Definitions by guid.
        self.instance_lookup: dict[str, InstanceRef] = {}  # Instances by guid.
        self.interactions: dict = {}  # Interactions per graph edge, by the edge's guid; a subclass keeps its type.
        self.history = History()  # Undo/redo buffer, purged by every save.
        self.bvh = SpatialBVH()  # Bounding volume hierarchy for collision detection.
        self.cached_ray_bvh = SpatialBVH()  # Cached SpatialBVH for ray casting.
        self.cached_guids: list[str] = []  # GUID per leaf of cached_ray_bvh.
        self.cached_boxes: list[OBB] = []  # Box per leaf of cached_ray_bvh.
        self.bvh_cache_dirty = True  # Flag to rebuild cached_ray_bvh.
        self.node_lookup: dict[str, TreeNode] = {}  # Tree node per live object guid.
        self.revision = 0  # Bumped by every Session mutation.
        self._sweep = []  # Parents whose children died, for the purge to compact.
        self._pinned = []  # Parents the purge left while a record pinned a child.
        self._purging = None  # The purge phase: 0..12 objects lists, 13..25 definitions lists, 26 the tree; None between cycles.
        self._writer = None  # The checkpoint being written, stale once revision moves.

        self.tree.add(TreeNode(name=self.name))
        self._indexed = weakref.ref(self.tree.root)  # Root at the last reindex.

    def __deepcopy__(self, memo):
        """Copy every table and object, guids included; caches are rebuilt on demand and history starts empty."""

        result = Session(self.name)

        if self.has_guid():
            result.guid = self.guid

        result.objects = _clone_objects(self.objects)
        result.tree = copy.deepcopy(self.tree, memo)
        result.graph = copy.deepcopy(self.graph, memo)
        result.graph.renumber()
        result.xforms = copy.deepcopy(self.xforms, memo)
        result.definitions = _clone_objects(self.definitions)

        for edge, interactions in self.interactions.items():
            for interaction in interactions:
                result.interactions.setdefault(edge, []).append(interaction.clone())

        result.reindex()
        memo[id(self)] = result

        return result

    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""
        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def get_object(self, guid: str) -> Any | None:
        """Get a geometry object by GUID, None when there is none."""
        return self.lookup.get(guid)

    def get_node(self, guid: str) -> TreeNode | None:
        """The tree node of a live object in O(1) through node_lookup; a tree search when the index is stale, None for a guid that is no live object."""

        if not self._is_live(guid):
            return None

        indexed = None if self._indexed is None else self._indexed()
        node = self.node_lookup.get(guid)

        if indexed is not None and indexed is self.tree.root and node is not None:
            if node.name == guid and node.parent is not None:
                return node

        return self.tree.get_node_by_name(guid)

    def select_by_type(self, cls: type) -> list[list]:
        """Select objects of one type, grouped by the top-level nodes of the tree."""

        groups = []
        root = self.tree.root

        if root is None:
            return groups

        for group in root.children:
            items = []

            for node in group.descendants():
                obj = self.get_object(node.name)

                if isinstance(obj, cls):
                    items.append(obj)

            if items:
                groups.append(items)

        return groups

    def find_group(self, group_name: str) -> TreeNode:
        """Find an existing group by name; raises ValueError when there is none."""

        root = self.tree.root

        if root is not None:
            for child in root.children:
                if child is not None and child.name == group_name:
                    return child

        raise ValueError(f"Group '{group_name}' not found")

    def order(self) -> list[str]:
        """Canonical object order: the objects lists walked in one fixed type sequence; instances are not in it."""

        order = []

        for point in self.objects.points:
            order.append(point.guid)

        for line in self.objects.lines:
            order.append(line.guid)

        for plane in self.objects.planes:
            order.append(plane.guid)

        for bbox in self.objects.bboxes:
            order.append(bbox.guid)

        for polyline in self.objects.polylines:
            order.append(polyline.guid)

        for pointcloud in self.objects.pointclouds:
            order.append(pointcloud.guid)

        for mesh in self.objects.meshes:
            order.append(mesh.guid)

        for nurbscurve in self.objects.nurbscurves:
            order.append(nurbscurve.guid)

        for nurbssurface in self.objects.nurbssurfaces:
            order.append(nurbssurface.guid)

        for brep in self.objects.breps:
            order.append(brep.guid)

        for element in self.objects.elements:
            order.append(element.guid)

        return order

    def xform(self, guid: str) -> Xform:
        """The LOCAL transform of an object, identity when none was set."""
        return self.xforms.get(guid, Xform.identity())

    def world_xform(self, guid: str) -> Xform:
        """The CUMULATIVE placement of an object: every ancestor's transform multiplied down the tree onto its own."""

        acc = self.xform(guid)
        node = self.get_node(guid)

        if node is None:
            return acc

        for ancestor in node.ancestors:
            xform = self.xforms.get(ancestor.name)

            if xform is not None:
                acc = xform * acc

        return acc

    def world_xforms(self) -> dict[str, Xform]:
        """Every object's cumulative placement, computed in one downward pass."""

        out: dict[str, Xform] = {}

        if not self.xforms:
            return out

        stack = []

        if self.tree.root is not None:
            stack.append((self.tree.root, Xform.identity()))

        while stack:
            node, parent_xform = stack.pop()
            local = self.xforms.get(node.name)
            current = parent_xform if local is None else parent_xform * local
            out[node.name] = current

            for child in node.children:
                stack.append((child, current))

        for obj_guid, obj_xform in self.xforms.items():
            out.setdefault(obj_guid, obj_xform)

        return out

    def get_children(self, obj_guid: str) -> list[str]:
        """Get the children of a parent GUID."""
        return self.tree.get_children_guids(obj_guid)

    def get_neighbours(self, obj_guid: str) -> list[str]:
        """Get the neighbours of a GUID."""
        return self.graph.neighbors(obj_guid)

    def get_geometry(self) -> Objects:
        """All geometry with its hierarchical placement BAKED into the coordinates; each instance becomes its definition placed, in the definition's list."""

        out = _clone_objects(self.objects)
        world = self.world_xforms()
        _bake(out.points, world)
        _bake(out.lines, world)
        _bake(out.planes, world)
        _bake(out.bboxes, world)
        _bake(out.polylines, world)
        _bake(out.pointclouds, world)
        _bake(out.meshes, world)
        _bake(out.nurbscurves, world)
        _bake(out.nurbssurfaces, world)
        _bake(out.breps, world)

        for element in out.elements:
            xform = world.get(element.guid)

            if xform is None or xform.is_identity():
                continue

            element.place(xform)

        for instance in self.objects.instances:
            definition = self.definition_lookup.get(instance.definition_guid)

            if definition is None:
                continue

            resolved = _resolve(
                instance, definition, world.get(instance.guid, Xform.identity())
            )
            getattr(out, _collection_of(resolved)[0]).append(resolved)

        out.instances.clear()

        return out

    def definition_of(self, instance_guid: str) -> Any | None:
        """The definition an instance places; None when guid is no instance or its definition is missing."""

        instance = self.instance_lookup.get(instance_guid)

        if instance is None:
            return None

        return self.definition_lookup.get(instance.definition_guid)

    def instances_of(self, definition_guid: str) -> list[str]:
        """Guids of every instance of a definition, in objects.instances order."""

        guids = []

        for instance in self.objects.instances:
            if instance.definition_guid == definition_guid:
                guids.append(instance.guid)

        return guids

    def world_geometry(self, guid: str) -> Any | None:
        """One object in world placement, as a copy: an instance becomes its definition moved by the world transform, carrying the instance's guid, name and features."""

        world = self.world_xform(guid)
        geometry = self.lookup.get(guid)

        if geometry is not None:
            result = clone(geometry)
            _place(result, world)

            return result

        definition = self.definition_of(guid)

        if definition is None:
            return None

        return _resolve(self.instance_lookup[guid], definition, world)

    # ═══════════════════════════════════════════════════════════════════════════
    # Geometry management
    # ═══════════════════════════════════════════════════════════════════════════
    def add_point(
        self, point: Point | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a point; None adds nothing and returns None, a guid already live adds nothing and returns the node that guid has."""

        if point is None:
            return None

        node = self._add_object("points", point, "point", parent)

        return node if node is not None else self._node_of(point.guid)

    def add_line(
        self, line: Line | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a line; None adds nothing and returns None, a guid already live adds nothing and returns the node that guid has."""

        if line is None:
            return None

        node = self._add_object("lines", line, "line", parent)

        return node if node is not None else self._node_of(line.guid)

    def add_plane(
        self, plane: Plane | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a plane; None adds nothing and returns None, a guid already live adds nothing and returns the node that guid has."""

        if plane is None:
            return None

        node = self._add_object("planes", plane, "plane", parent)

        return node if node is not None else self._node_of(plane.guid)

    def add_obb(self, bbox: OBB | None) -> TreeNode | None:
        """Add a bounding box; None adds nothing and returns None, a guid already live adds nothing and returns the node that guid has."""

        if bbox is None:
            return None

        node = self._add_object("bboxes", bbox, "bbox", None)

        return node if node is not None else self._node_of(bbox.guid)

    def add_polyline(
        self, polyline: Polyline | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a polyline; None, or fewer than two points, or a guid already live, adds nothing and returns None."""

        if polyline is None or polyline.point_count() < 2:
            return None

        return self._add_object("polylines", polyline, "polyline", parent)

    def add_pointcloud(
        self, pointcloud: PointCloud | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a point cloud; None, or no points, or a guid already live, adds nothing and returns None."""

        if pointcloud is None or pointcloud.is_empty():
            return None

        return self._add_object("pointclouds", pointcloud, "pointcloud", parent)

    def add_mesh(
        self, mesh: Mesh | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a mesh; None, or no faces, or a guid already live, adds nothing and returns None."""

        if mesh is None or mesh.is_empty() or mesh.number_of_faces() == 0:
            return None

        return self._add_object("meshes", mesh, "mesh", parent)

    def add_nurbscurve(
        self, nurbscurve: NurbsCurve | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a curve; None, or fewer than two control vertices, or a guid already live, adds nothing and returns None."""

        if nurbscurve is None or nurbscurve.cv_count() < 2:
            return None

        return self._add_object("nurbscurves", nurbscurve, "nurbscurve", parent)

    def add_nurbssurface(
        self, nurbssurface: NurbsSurface | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a surface; None, or no control vertices, or a guid already live, adds nothing and returns None."""

        if nurbssurface is None or nurbssurface.cv_count() == 0:
            return None

        return self._add_object("nurbssurfaces", nurbssurface, "nurbssurface", parent)

    def add_brep(
        self, brep: BRep | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a brep; None, or no faces and no vertices, or a guid already live, adds nothing and returns None."""

        if brep is None or (brep.face_count() == 0 and brep.vertex_count() == 0):
            return None

        return self._add_object("breps", brep, "brep", parent)

    def add_element(
        self, element: Element | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add an element, a data record kept even without geometry; None adds nothing and returns None, a guid already live adds nothing and returns the node that guid has."""

        if element is None:
            return None

        node = self._add_object("elements", element, "element", parent)

        return node if node is not None else self._node_of(element.guid)

    def add_component(
        self, component: Component, parent: TreeNode | None = None
    ) -> TreeNode:
        """Add a custom component (any object with type_name/guid/name/extra); a guid already live adds nothing and returns the node that guid has."""

        node = self._add_object("components", component, "component", parent)

        return node if node is not None else self._node_of(component.guid)

    def add_definition(self, definition: Any) -> str:
        """Add a definition, geometry in its own frame that instances share; returns its guid, also when that guid is already defined, and "" for None or a guid an object, instance or component holds."""

        if definition is None:
            return ""

        guid = definition.guid

        if guid in self.definition_lookup:
            return guid

        if self._is_live(guid):
            return ""

        collection, _ = _collection_of(definition)
        items = getattr(self.definitions, collection)
        items.append(definition)
        slot = items.number_of_slots() - 1
        self.definition_lookup[guid] = definition
        self.bvh_cache_dirty = True
        self.revision += 1

        if self.history.current is not None:
            tomb = Tomb(collection, True, slot, None)
            items.set_tomb(slot, tomb)
            self.history.record(AddOp(guid, "definitions", None, 0, None, tomb), RECORD)

        return guid

    def add_instance(
        self,
        instance: InstanceRef,
        xform: Xform | None = None,
        parent: TreeNode | None = None,
    ) -> TreeNode | None:
        """Add an instance under parent, placed by xform relative to the parent with its own xform folded in; None when None, its definition_guid names no definition or its guid is already live."""

        if instance is None or instance.definition_guid not in self.definition_lookup:
            return None

        placement = (xform if xform is not None else Xform.identity()) * instance.xform
        node = self._add_object("instances", instance, "instance", parent)

        if node is None:
            return None

        instance.xform = Xform.identity()

        if not placement.is_identity():
            self.set_xform(instance.guid, placement)

        return node

    def add(self, node: TreeNode | None, parent: TreeNode | None = None) -> None:
        """Put a TreeNode under a parent, the root when none is given: a placed node moves and leaves a ghost, one already there is left alone; None is ignored."""

        if node is None:
            return

        parent = parent if parent is not None else self.tree.root

        if parent is None or node is parent or node.parent is parent:
            return

        name = node.name
        was_dead = node.is_dead()
        old = node.parent
        ghost = parent.add(node)

        if not parent.has_child(node):
            return

        node.set_dead(False)
        self.revision += 1

        if old is not None and ghost is not None:
            self._queue(old)

        tomb = node.get_tomb()

        if was_dead and tomb is not None and tomb.collection == "":
            if tomb.xform is not None:
                self.xforms[name] = tomb.xform
                tomb.xform = None

        if self._is_live(name):
            self.node_lookup[name] = node

        if self.history.current is None:
            self.history.dropped += 1 if ghost is not None else 0

            return

        tomb = self._node_tomb(ghost if ghost is not None else node)
        color = node.color
        dead_before = was_dead or ghost is None
        self.history.record(
            TreeOp(
                name, node, tomb, ghost, name, name, color, color, dead_before, False
            ),
            RECORD,
        )

    def add_group(self, group_name: str) -> TreeNode:
        """Create a named group (TreeNode) and add it to the root of the tree."""

        node = TreeNode(name=group_name)
        self.add(node)

        return node

    def rename_node(self, node: TreeNode, name: str) -> bool:
        """Rename a group node; False for an object node, a dead node or the same name."""

        before = node.name

        if self._is_live(before) or node.is_dead() or before == name:
            return False

        node.name = name
        self.revision += 1

        if self.history.current is not None:
            tomb = self._node_tomb(node)
            color = node.color
            self.history.record(
                TreeOp(
                    before, node, tomb, None, before, name, color, color, False, False
                ),
                RECORD,
            )

        return True

    def set_node_color(self, node: TreeNode, color: Color | None) -> bool:
        """Set or clear (None) the display colour of a node; False for a dead node."""

        if node.is_dead():
            return False

        before = node.color
        node.color = color
        self.revision += 1

        if self.history.current is not None:
            name = node.name
            tomb = self._node_tomb(node)
            self.history.record(
                TreeOp(name, node, tomb, None, name, name, before, color, False, False),
                RECORD,
            )

        return True

    def remove_group(self, node: TreeNode) -> bool:
        """Kill a group node with everything below it, parking its transform; False for an object node, the root or a dead node."""

        name = node.name
        parent = node.parent

        if parent is None or self._is_live(name):
            return False

        tomb = self._node_tomb(node)
        node.set_dead(True)
        tomb.xform = self.xforms.pop(name, None)
        self._queue(parent)
        self.revision += 1

        if self.history.current is None:
            self.history.dropped += 1

            return True

        color = node.color
        self.history.record(
            TreeOp(name, node, tomb, None, name, name, color, color, False, True),
            RECORD,
        )

        return True

    def add_edge(self, guid1: str, guid2: str, attribute: str = "") -> None:
        """Add an edge between two geometry objects in the graph."""

        self.revision += 1
        self.graph.add_edge(guid1, guid2, attribute)

    def add_hierarchy(self, parent_guid: str, child_guid: str) -> bool:
        """Add a parent-child relationship in the tree."""

        self.revision += 1

        return self.tree.add_child_by_guid(parent_guid, child_guid)

    def add_relationship(
        self, from_guid: str, to_guid: str, relationship_type: str = "default"
    ) -> None:
        """Add a relationship edge in the graph."""

        self.revision += 1
        self.graph.add_edge(from_guid, to_guid, relationship_type)

    def remove_object(self, obj_guid: str) -> bool:
        """Kill an object in place: its slot, node, transform, vertex, edges and interactions flip dead until undo revives them; O(1 + d log V)."""

        obj = self._item(obj_guid)

        if obj is None:
            return False

        tomb = self._tomb(obj_guid)
        degree = len(self.graph.edges.get(obj_guid, {}))
        node = tomb.node if tomb.node.parent is not None else None
        index = node.at() if node is not None else 0
        parent_guid = node.parent.name if node is not None else None
        self._kill(tomb)

        if self.history.current is None:
            self.history.dropped += 1

            return True

        self.history.record(
            RemoveOp(obj_guid, tomb.collection, parent_guid, index, node, tomb),
            RECORD + weight(obj) + 128 * degree,
        )

        return True

    def replace(self, guid: str, obj: Any) -> bool:
        """Swap the object stored under guid for obj, which takes over that guid; a different type moves the guid to that type's list, the node and edges staying."""

        before = self.lookup.get(guid)

        if before is None:
            return False

        obj.guid = guid
        old, _ = _collection_of(before)
        new, prefix = _collection_of(obj)
        bytes = RECORD + weight(before)

        if old == new:
            entry = Entry(False, self.get_node(guid), None)

            if self.history.current is not None:
                self.history.record(ReplaceOp(guid, before, obj, entry), bytes)

            self._swap(guid, obj, entry)

            return True

        node = self.get_node(guid)
        removed = self._half(False, old, guid)
        self._kill(removed)
        items = getattr(self.objects, new)
        items.append(obj)
        added = Tomb(new, False, items.number_of_slots() - 1, None)
        items.set_tomb(added.slot, added)
        self.lookup[guid] = obj
        self._label(guid, f"{prefix}_{obj.name}")
        self._pair(guid, old, new, node, removed, added, bytes)

        return True

    def replace_definition(self, guid: str, definition: Any) -> bool:
        """Swap the geometry of a definition, which keeps its guid, so every instance of it changes at once; False when guid is no definition."""

        before = self.definition_lookup.get(guid)

        if before is None:
            return False

        definition.guid = guid
        old, _ = _collection_of(before)
        new, _ = _collection_of(definition)
        bytes = RECORD + weight(before)

        if old == new:
            entry = Entry(True, None, self._half(True, old, guid))

            if self.history.current is not None:
                self.history.record(ReplaceOp(guid, before, definition, entry), bytes)

            self._swap(guid, definition, entry)

            return True

        removed = self._half(True, old, guid)
        self._kill(removed)
        items = getattr(self.definitions, new)
        items.append(definition)
        added = Tomb(new, True, items.number_of_slots() - 1, None)
        items.set_tomb(added.slot, added)
        self.definition_lookup[guid] = definition
        self._pair(guid, "definitions", "definitions", None, removed, added, bytes)

        return True

    def remove_definition(self, guid: str) -> bool:
        """Remove a definition; False when guid is no definition or an instance still names it."""

        before = self.definition_lookup.get(guid)

        if before is None or len(self.instances_of(guid)) > 0:
            return False

        collection, _ = _collection_of(before)
        tomb = self._half(True, collection, guid)
        self._kill(tomb)

        if self.history.current is None:
            self.history.dropped += 1

            return True

        self.history.record(
            RemoveOp(guid, "definitions", None, 0, None, tomb),
            RECORD + weight(before),
        )

        return True

    def to_instance(self, guid: str, definition_guid: str, frame: Xform) -> bool:
        """Turn an object into an instance of a definition, keeping its guid, name, tree node and edges; frame maps the definition onto the object and is folded into its local transform."""

        obj = self.lookup.get(guid)

        if obj is None or definition_guid not in self.definition_lookup:
            return False

        instance = InstanceRef(definition_guid, Xform.identity())
        instance.guid = guid
        instance.name = obj.name
        placement = self.xform(guid) * frame
        old, _ = _collection_of(obj)
        node = self.get_node(guid)
        removed = self._half(False, old, guid)
        self._kill(removed)
        items = self.objects.instances
        items.append(instance)
        added = Tomb("instances", False, items.number_of_slots() - 1, None)
        items.set_tomb(added.slot, added)
        self.instance_lookup[guid] = instance
        self._label(guid, f"instance_{instance.name}")
        bytes = RECORD + weight(obj)
        self._pair(guid, old, "instances", node, removed, added, bytes)

        if placement.is_identity():
            self.remove_xform(guid)
        else:
            self.set_xform(guid, placement)

        return True

    def explode(self, instance_guid: str) -> bool:
        """Turn an instance into a standalone copy of its definition in the definition frame, keeping its guid, name, transform, tree node and edges, and on an element its features."""

        definition = self.definition_of(instance_guid)

        if definition is None:
            return False

        instance = self.instance_lookup[instance_guid]
        result = _resolve(instance, definition, Xform.identity())
        collection, prefix = _collection_of(result)
        node = self.get_node(instance_guid)
        removed = self._half(False, "instances", instance_guid)
        self._kill(removed)
        items = getattr(self.objects, collection)
        items.append(result)
        added = Tomb(collection, False, items.number_of_slots() - 1, None)
        items.set_tomb(added.slot, added)
        self.lookup[instance_guid] = result
        self._label(instance_guid, f"{prefix}_{instance.name}")
        bytes = RECORD + weight(instance)
        self._pair(instance_guid, "instances", collection, node, removed, added, bytes)

        return True

    def set_xform(self, guid: str, xform: Xform) -> None:
        """Sets the LOCAL transform of an object, relative to its tree parent; a guid that names only a definition is ignored."""

        if (
            guid in self.definition_lookup
            and guid not in self.lookup
            and guid not in self.instance_lookup
        ):
            return

        if self.history.current is not None:
            before = self.xforms.get(guid)
            before = None if before is None else before.duplicate()
            node = self.get_node(guid)
            self.history.record(XformOp(guid, before, xform.duplicate(), node), RECORD)

        self.xforms[guid] = xform
        self.bvh_cache_dirty = True
        self.revision += 1

    def remove_xform(self, guid: str) -> bool:
        """Removes an object's local transform, returning whether one was present."""

        before = self.xforms.get(guid)

        if before is None:
            return False

        if self.history.current is not None:
            node = self.get_node(guid)
            self.history.record(XformOp(guid, before.duplicate(), None, node), RECORD)

        del self.xforms[guid]
        self.bvh_cache_dirty = True
        self.revision += 1

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # Session - Interactions
    # ═══════════════════════════════════════════════════════════════════════════
    def add_interaction(
        self, a: Element, b: Element, interaction: Interaction
    ) -> Interaction:
        """Make or reuse the pair's undirected edge, an existing edge keeping its attributes, and append interaction to its list; returns the stored interaction. Raises ValueError unless both elements are in the session and distinct."""

        first = a.guid
        second = b.guid

        if (
            first == second
            or not _registered(self, first)
            or not _registered(self, second)
        ):
            raise ValueError(
                "Session.add_interaction: add two distinct elements to the session first"
            )

        if not self.graph.has_edge((first, second)):
            self.graph.add_edge(first, second)

        self.revision += 1
        id = self.graph.edges[first][second].guid
        self.interactions.setdefault(id, []).append(interaction)

        return interaction

    def get_interaction(self, a: Element, b: Element) -> list[Interaction]:
        """The pair's interactions in either order, empty when there are none."""

        if not self.graph.has_edge((a.guid, b.guid)):
            return []

        id = self.graph.edges[a.guid][b.guid].guid

        return list(self.interactions.get(id, []))

    def has_interaction(self, a: Element, b: Element) -> bool:
        """True when the pair has an edge in either order."""
        return self.graph.has_edge((a.guid, b.guid))

    def remove_interaction(self, a: Element, b: Element) -> None:
        """Remove the pair's edge and all of its interactions in either order; a missing pair is a no-op."""

        if not self.graph.has_edge((a.guid, b.guid)):
            return

        id = self.graph.edges[a.guid][b.guid].guid
        self.revision += 1
        self.interactions.pop(id, None)
        self.graph.remove_edge((a.guid, b.guid))

    # ═══════════════════════════════════════════════════════════════════════════
    # History
    # ═══════════════════════════════════════════════════════════════════════════
    def begin(self, label: str) -> None:
        """Open a history transaction: every add, remove, replace and xform change until commit becomes one undo step."""
        self.history.begin(label)

    def commit(self) -> None:
        """Close the open transaction as one undo step."""
        self.history.commit()

    def undo(self) -> bool:
        """Revert the latest committed transaction, returning whether there was one."""

        self.revision += 1

        return self.history.undo(self)

    def redo(self) -> bool:
        """Reapply the latest undone transaction, returning whether there was one."""

        self.revision += 1

        return self.history.redo(self)

    def abort(self) -> bool:
        """Revert and drop the open transaction, leaving the stacks as they are; False when none is open."""

        self.revision += 1

        return self.history.abort(self)

    # ═══════════════════════════════════════════════════════════════════════════
    # Purge
    # ═══════════════════════════════════════════════════════════════════════════
    def number_of_dead(self) -> int:
        """Return the dead slots not yet purged, over the objects and the definitions lists."""

        count = 0

        for collection, _ in COLLECTIONS:
            count += getattr(self.objects, collection).number_of_dead()
            count += getattr(self.definitions, collection).number_of_dead()

        return count

    def purge_due(self) -> bool:
        """Return whether dropped records left dead entries or swept parents a purge cycle can free."""
        return self.history.dropped > 0 and (
            self.number_of_dead() > 0 or bool(self._sweep)
        )

    def is_purging(self) -> bool:
        """Return whether a purge cycle is part way."""
        return self._purging is not None

    def purge_step(self, work: int) -> bool:
        """Purge what no record reaches for at most work slots or children, resuming the running cycle; True while it is unfinished."""

        fresh = self._writer is not None and self._writer.revision == self.revision

        if fresh or (self._purging is None and not self.purge_due()):
            return False

        self._purge(work)

        return self._purging is not None

    def purge(self) -> None:
        """Drop the history and purge everything it pinned in one call: the running and a whole cycle, every live tree node, dense graph indices; O(n + N + V log V + E log E)."""

        self.history.clear()
        self._writer = None

        if self._purging is not None:
            self._purge(sys.maxsize)

        self._purge(sys.maxsize)

        for node in self.tree.nodes:
            node.compact()

        self.graph.renumber()
        self.revision += 1

    def checkpoint(self, work: int) -> bytes | None:
        """Write the live session as protobuf bytes for at most work units, purging first when due; bytes once done, history kept, restarted by any edit."""

        if self._writer is not None and self._writer.revision != self.revision:
            self._writer = None

        if self._writer is None and (self._purging is not None or self.purge_due()):
            work = self._purge(work)

        if self._purging is not None or work == 0:
            return None

        writer = self._writer
        self._writer = None

        if writer is None:
            writer = _Checkpoint(self.revision)

        if self._write(writer, work):
            return bytes(writer.out)

        self._writer = writer

        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Collision detection and ray casting
    # ═══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_bounding_box(geometry: Any, xform: Xform) -> OBB:
        """Bounding box of an object in WORLD placement, inflated by tolerance."""

        inflate = Tolerance.APPROXIMATION

        if isinstance(geometry, Point):
            return OBB.from_point(xform.transform_point(geometry), inflate)

        if isinstance(geometry, Plane):
            return OBB.from_point(
                xform.transform_point(geometry.origin), inflate * 10.0
            )

        if isinstance(geometry, OBB):
            inflated = copy.deepcopy(geometry)
            inflated.half_size = inflated.half_size + Vector(inflate, inflate, inflate)
            inflated.transform(xform)

            return inflated

        if isinstance(geometry, Element):
            box = copy.deepcopy(geometry.aabb)
            box.transform(xform)

            return box

        return _placed_box(_box_points(geometry), xform, inflate)

    def get_collisions(self) -> list[tuple[str, str]]:
        """Get all collision pairs using SpatialBVH and add them as graph edges."""

        guids: list[str] = []
        boxes = self._compute_boxes(guids)

        if len(boxes) == 0:
            return []

        self.bvh = SpatialBVH.from_boxes(boxes, SpatialBVH.compute_world_size(boxes))
        pairs, _colliding, _checks = self.bvh.check_all_collisions(boxes)
        guid_pairs = []

        for i, j in pairs:
            if i < 0 or j < 0 or i >= len(guids) or j >= len(guids):
                continue

            guid_pairs.append((guids[i], guids[j]))
            self.graph.add_edge(guids[i], guids[j], "bvh_collision")

        return guid_pairs

    def ray_cast(
        self, origin: Point, direction: Vector, tolerance: float = 1e-3
    ) -> list[RayHit]:
        """Cast a ray through the scene, returning the hits within tolerance of the closest one."""

        if self.bvh_cache_dirty:
            self._rebuild_ray_bvh_cache()
            self.bvh_cache_dirty = False

        if len(self.cached_guids) == 0:
            return []

        candidates: list[int] = []
        self.cached_ray_bvh.ray_cast(origin, direction, candidates, True)
        ray = Line.from_points(origin, origin + direction * 10000.0)
        world = self.world_xforms()
        hits: list[RayHit] = []
        closest = float("inf")

        for index in candidates:
            guid = self.cached_guids[index]
            geometry = self.lookup.get(guid)

            if geometry is None:
                geometry = self.definition_of(guid)

            if geometry is None:
                continue

            placement = world.get(guid, Xform.identity())
            hit = self._ray_intersect_geometry(ray, geometry, tolerance, placement)

            if hit is None:
                continue

            distance = origin.distance(hit)

            if distance >= closest:
                continue

            if distance < closest - tolerance:
                hits.clear()

            hits.append(RayHit(guid, hit, distance))
            closest = distance

        return hits

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

        xforms = []

        for obj_guid, obj_xform in self._xforms_ordered():
            xforms.append({"guid": obj_guid, "xform": obj_xform.__jsondump__()})

        interactions = []

        for edge in sorted(self.interactions):
            items = []

            for interaction in self.interactions[edge]:
                items.append(interaction.__jsondump__())

            interactions.append({"guid": edge, "interactions": items})

        data = {}

        if self.definition_lookup:
            data["definitions"] = self.definitions.__jsondump__()

        data["graph"] = self.graph.__jsondump__()
        data["guid"] = self.guid
        data["interactions"] = interactions
        data["name"] = self.name
        data["objects"] = self.objects.__jsondump__()
        data["tree"] = self.tree.__jsondump__()
        data["type"] = "Session"
        data["xforms"] = xforms

        return data

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Session:
        """Deserialize from a JSON object."""

        from .file_encoders import file_decode_node

        session = cls(name=data.get("name", "my_session"))
        session.guid = guid if guid is not None else data.get("guid", session.guid)

        if data.get("objects"):
            session.objects = file_decode_node(data["objects"])

        if data.get("tree"):
            session.tree = file_decode_node(data["tree"])

        if data.get("graph"):
            session.graph = file_decode_node(data["graph"])

        if data.get("definitions"):
            session.definitions = file_decode_node(data["definitions"])

        for entry in data.get("xforms", []):
            session.xforms[entry["guid"]] = Xform.__jsonload__(entry["xform"])

        for entry in data.get("interactions", []):
            for item in entry["interactions"]:
                session.interactions.setdefault(entry["guid"], []).append(
                    Interaction.__jsonload__(item)
                )

        session.reindex()

        return session

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string, dropping the history and purging what it pinned."""

        self.purge()

        return json.dumps(self.__jsondump__(), separators=(",", ":"))

    @classmethod
    def file_json_loads(cls, json_string: str) -> Session:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file, dropping the history and purging what it pinned."""

        self.purge()

        with open(filename, "w") as f:
            json.dump(self.__jsondump__(), f, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Session:
        """Read from a JSON file."""

        with open(filename) as f:
            return cls.__jsonload__(json.load(f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def to_proto(self) -> session_pb2.Session:
        """Convert to the protobuf message."""

        from .proto import session_pb2

        proto = session_pb2.Session()
        proto.name = self.name

        if self.has_guid():
            proto.guid = self.guid

        proto.objects.CopyFrom(self.objects.to_proto())
        proto.tree.ParseFromString(self.tree.pb_dumps())
        proto.graph.CopyFrom(self.graph.to_proto())

        for obj_guid, obj_xform in self._xforms_ordered():
            item = proto.xforms.add()
            item.guid = obj_guid
            item.xform.CopyFrom(obj_xform.to_proto())

        if self.definition_lookup:
            proto.definitions.CopyFrom(self.definitions.to_proto())

        for edge in sorted(self.interactions):
            item = proto.interactions.add()
            item.guid = edge

            for interaction in self.interactions[edge]:
                item.interactions.add().CopyFrom(interaction.to_proto())

        return proto

    @classmethod
    def from_proto(cls, proto: session_pb2.Session) -> Session:
        """Construct from the protobuf message."""

        session = cls(name=proto.name)

        if proto.guid:
            session.guid = proto.guid

        if proto.HasField("objects"):
            session.objects = Objects.from_proto(proto.objects)

        if proto.HasField("tree"):
            session.tree = Tree.pb_loads(proto.tree.SerializeToString())

        if proto.HasField("graph"):
            session.graph = Graph.from_proto(proto.graph)

        if proto.HasField("definitions"):
            session.definitions = Objects.from_proto(proto.definitions)

        for entry in proto.xforms:
            session.xforms[entry.guid] = Xform.from_proto(entry.xform)

        for entry in proto.interactions:
            for item in entry.interactions:
                session.interactions.setdefault(entry.guid, []).append(
                    Interaction.from_proto(item)
                )

        session.reindex()

        return session

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes with map entries sorted, dropping the history and purging what it pinned."""

        self.purge()

        return self.to_proto().SerializeToString(deterministic=True)

    @classmethod
    def pb_loads(cls, data: bytes) -> Session:
        """Deserialize from protobuf bytes."""

        from .proto import session_pb2

        proto = session_pb2.Session()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file, dropping the history and purging what it pinned."""

        with open(filename, "wb") as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Session:
        """Read from a protobuf file."""

        with open(filename, "rb") as f:
            return cls.pb_loads(f.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return the spatial hierarchy and the element interactions as a banner block."""

        bar = "=" * 80

        return f"{bar}\nSpatial Hierarchy\n{bar}\n{self.tree}{bar}\nElement Interactions\n{bar}\n{self.graph}\n{bar}\n"

    def __repr__(self) -> str:
        """Return "Session(name=..., objects=..., tree=..., graph=...)"."""
        return f"Session(name={self.name}, objects={self.objects}, tree={self.tree!r}, graph={self.graph!r})"

    # ═══════════════════════════════════════════════════════════════════════════
    # Details
    # ═══════════════════════════════════════════════════════════════════════════
    def _add_object(
        self, collection: str, obj: Any, type_prefix: str, parent: TreeNode | None
    ) -> TreeNode | None:
        """Store an object in its list, lookup, graph and tree, recording an add when a transaction is open; None for a guid that is already live, as an object or a definition, which adds nothing."""

        guid = obj.guid

        if self._is_live(guid) or guid in self.definition_lookup:
            return None

        items = getattr(self.objects, collection)
        items.append(obj)
        slot = items.number_of_slots() - 1
        self._table(collection)[guid] = obj
        attribute = f"{type_prefix}_{obj.name}"
        self.graph.add_node(guid, attribute)
        self.bvh_cache_dirty = True
        node = TreeNode(name=guid)
        self.node_lookup[guid] = node
        self.revision += 1
        host = parent if parent is not None else self.tree.root
        parent_guid = None

        if host is not None:
            self.tree.add(node, host)
            parent_guid = host.name

        if self.history.current is not None:
            tomb = Tomb(collection, False, slot, node)
            items.set_tomb(slot, tomb)
            node.set_tomb(tomb)
            self.history.record(
                AddOp(guid, collection, parent_guid, node.at(), node, tomb), RECORD
            )

        return node

    def _node_of(self, guid: str) -> TreeNode:
        """The node of a live guid, a detached one named guid when the object is outside the tree."""

        node = self.get_node(guid)

        return node if node is not None else TreeNode(name=guid)

    def _twin(self, definition: bool, collection: str, slot: int, guid: str) -> bool:
        """Whether the live entry under guid, if any, is another entry than the one in a slot of the list of that name: one on the other side of the object/definition divide, of another type, or of the same type at another slot."""

        if definition:
            other = self._is_live(guid)
            held = self.definition_lookup.get(guid)
            items = getattr(self.definitions, collection)
        else:
            other = guid in self.definition_lookup
            held = self._item(guid)
            items = getattr(self.objects, collection)

        if other:
            return True

        if held is None:
            return False

        return _collection_for(held)[0] != collection or items.get_slot(guid) != slot

    def _owns(self, guid: str, node: TreeNode | None) -> bool:
        """Whether the entry under guid is the one a record was taken on: any entry when no node was recorded, else the live entry whose node it is."""
        return node is None or self.get_node(guid) is node

    def _is_live(self, guid: str) -> bool:
        """Whether guid names a live object, component or instance."""

        return (
            guid in self.lookup
            or guid in self.component_lookup
            or guid in self.instance_lookup
        )

    def _table(self, collection: str) -> dict[str, Any]:
        """The lookup map of a list name."""

        if collection == "components":
            return self.component_lookup

        if collection == "instances":
            return self.instance_lookup

        return self.lookup

    def _item(self, guid: str) -> Any | None:
        """The stored object, component or instance under guid, the object itself."""

        item = self.lookup.get(guid)

        if item is None:
            item = self.component_lookup.get(guid)

        if item is None:
            item = self.instance_lookup.get(guid)

        return item

    def _tomb(self, guid: str) -> Tomb | None:
        """The object tomb of a live guid, reused while a record still holds it, else made and pinned on its slot and node; O(1)."""

        item = self._item(guid)

        if item is None:
            return None

        collection, _ = _collection_for(item)
        items = getattr(self.objects, collection)
        slot = items.get_slot(guid)

        if slot is None:
            items.append(item)
            slot = items.number_of_slots() - 1

        tomb = items.get_tomb(slot)

        if tomb is not None and tomb.node is not None:
            return tomb

        node = self.get_node(guid)

        if node is not None:
            self.node_lookup[guid] = node
        else:
            node = TreeNode(name=guid)

        tomb = Tomb(collection, False, slot, node)
        items.set_tomb(slot, tomb)
        node.set_tomb(tomb)

        return tomb

    def _node_tomb(self, node: TreeNode) -> Tomb:
        """The node-only tomb pinned on a node, reused while a record still holds it."""

        tomb = node.get_tomb()

        if tomb is not None and tomb.collection == "":
            return tomb

        tomb = Tomb("", False, 0, node)
        node.set_tomb(tomb)

        return tomb

    def _half(self, definition: bool, collection: str, guid: str) -> Tomb:
        """A slot-only tomb on the live slot of guid in the list of that name, reused while a record still holds it; a map-only entry is appended first."""

        item = self.definition_lookup[guid] if definition else self._item(guid)
        objects = self.definitions if definition else self.objects
        items = getattr(objects, collection)
        slot = items.get_slot(guid)

        if slot is None:
            items.append(item)
            slot = items.number_of_slots() - 1

        tomb = items.get_tomb(slot)

        if tomb is not None and tomb.node is None and tomb.definition == definition:
            return tomb

        tomb = Tomb(collection, definition, slot, None)
        items.set_tomb(slot, tomb)

        return tomb

    def _pair(
        self,
        guid: str,
        old: str,
        new: str,
        node: TreeNode | None,
        removed: Tomb,
        added: Tomb,
        bytes: int,
    ) -> None:
        """Record the halves of a type change under one guid, or drop them when no transaction is open."""

        self.revision += 1
        self.bvh_cache_dirty = True

        if self.history.current is None:
            self.history.dropped += 1

            return

        index = node.at() if node is not None else 0
        parent = node.parent if node is not None else None
        parent_guid = parent.name if parent is not None else None
        self.history.record(
            RemoveOp(guid, old, parent_guid, index, node, removed), bytes
        )
        self.history.record(AddOp(guid, new, parent_guid, index, node, added), RECORD)

    def _label(self, guid: str, label: str) -> None:
        """Relabel the graph vertex of guid, when it has one."""

        if self.graph.has_node(guid):
            self.graph.node_label(guid, label)

    def _queue(self, parent: TreeNode) -> None:
        """Remember a parent whose child died, once, for the sweep."""

        if parent.is_queued():
            return

        parent.set_queued(True)
        self._sweep.append(weakref.ref(parent))

    def _purge(self, work: int) -> int:
        """Run the purge cycle for at most work units, starting one when idle; returns the work left."""

        if self._purging is None:
            self._purging = 0
            self.history.dropped = 0

        while work > 0 and self._purging is not None:
            phase = self._purging

            if phase < 2 * len(COLLECTIONS):
                objects = self.objects if phase < len(COLLECTIONS) else self.definitions
                items = getattr(objects, COLLECTIONS[phase % len(COLLECTIONS)][0])

                if items.number_of_dead() > 0 or items.is_compacting():
                    work -= min(items.compact_step(work), work)

                if not items.is_compacting():
                    self._purging = phase + 1

                continue

            if not self._sweep:
                self._sweep = self._pinned
                self._pinned = []
                self._purging = None

                break

            parent = self._sweep[-1]()

            if parent is None:
                self._sweep.pop()
                work -= 1

                continue

            work -= min(max(parent.compact_step(work), 1), work)

            if parent.is_compacting():
                continue

            self._sweep.pop()

            if parent.has_dead():
                self._pinned.append(weakref.ref(parent))
            else:
                parent.set_queued(False)

        return work

    def _write(self, writer: _Checkpoint, work: int) -> bool:
        """Advance a checkpoint writer for at most work units; True once its message is complete."""

        while work > 0:
            phase = writer.phase

            if phase == _HEAD:
                spent = self._write_head(writer)
            elif phase < _TREE:
                spent = self._write_list(writer, False, work)
            elif phase == _TREE:
                spent = self._write_tree(writer, work)
            elif phase == _VERTICES:
                spent = self._write_vertices(writer, work)
            elif phase == _EDGES:
                spent = self._write_edges(writer, work)
            elif phase < _REST:
                spent = self._write_ordered(writer, work)
            elif phase == _REST:
                spent = self._write_rest(writer, work)
            elif phase < _INTERACTIONS:
                spent = self._write_list(writer, True, work)
            elif phase == _INTERACTIONS:
                spent = self._write_interactions(writer, work)
            else:
                return self._assemble(writer, work)

            work -= min(max(spent, 1), work)

        return False

    def _write_head(self, writer: _Checkpoint) -> int:
        """Write the session name and guid and the Objects head; returns one unit."""

        from .proto import session_pb2

        proto = session_pb2.Session()
        proto.name = self.name

        if self.has_guid():
            proto.guid = self.guid

        writer.sections["head"] = bytearray(proto.SerializeToString(deterministic=True))
        writer.sections["objects"] = _objects_head(self.objects)
        writer.phase = _OBJECTS

        return 1

    def _write_list(self, writer: _Checkpoint, definition: bool, work: int) -> int:
        """Write the live entries of one objects or definitions list from the cursor slot; returns the slots examined."""

        from .proto import objects_pb2

        first = _DEFINITIONS if definition else _OBJECTS
        collection = COLLECTIONS[writer.phase - first][0]
        items = getattr(self.definitions if definition else self.objects, collection)
        section = writer.sections["definitions" if definition else "objects"]
        start = writer.cursor
        end = min(items.number_of_slots(), start + work)

        for slot in range(start, end):
            if items.is_dead(slot):
                continue

            proto = objects_pb2.Objects()
            getattr(proto, collection).add().CopyFrom(items.get_item(slot).to_proto())
            section += proto.SerializeToString(deterministic=True)

        writer.cursor = end

        if end >= items.number_of_slots():
            writer.cursor = 0
            writer.phase += 1

        return end - start

    def _write_tree(self, writer: _Checkpoint, work: int) -> int:
        """Write the live tree depth first from an explicit stack, a finished node moved to its parent in chunks; returns the children examined."""

        from .proto import tree_pb2
        from .proto import treenode_pb2

        if writer.cursor == 0:
            writer.cursor = 1
            writer.sections["tree"] = self._tree_head()
            root = self.tree.root

            if root is not None:
                writer.stack.append([root, 0, [bytearray(node_head(root))]])

        spent = 0

        while spent < work and writer.stack:
            frame = writer.stack[-1]
            child = frame[0].get_child(frame[1])
            frame[1] += 1
            spent += 1

            if child is not None:
                if not child.is_dead():
                    writer.stack.append([child, 0, [bytearray(node_head(child))]])

                continue

            node, _, chunks = writer.stack.pop()
            _append(chunks, bytearray(node_tail(node)))
            length = sum(len(chunk) for chunk in chunks)

            if writer.stack:
                parent = writer.stack[-1][2]
                field = treenode_pb2.TreeNode.CHILDREN_FIELD_NUMBER
            else:
                parent = writer.tree
                field = tree_pb2.Tree.ROOT_FIELD_NUMBER

            _append(parent, _prefix(field, length))

            for chunk in chunks:
                _append(parent, chunk)

        if not writer.stack:
            writer.cursor = 0
            writer.phase = _VERTICES

        return spent

    def _tree_head(self) -> bytearray:
        """The guid and name fields of the Tree message."""

        from .proto import tree_pb2

        proto = tree_pb2.Tree()

        if self.tree.has_guid():
            proto.guid = self.tree.guid

        proto.name = self.tree.name

        return bytearray(proto.SerializeToString(deterministic=True))

    def _write_vertices(self, writer: _Checkpoint, work: int) -> int:
        """Write the graph head, then the vertices in name order, at most work per call; returns the vertices sorted or written."""

        from .proto import graph_pb2

        if writer.keys is None:
            if writer.unsorted is None:
                head = graph_pb2.Graph()
                head.name = self.graph.name

                if self.graph.has_guid():
                    head.guid = self.graph.guid

                writer.sections["graph"] = bytearray(head.SerializeToString())

            return writer.sort_step(self.graph.vertices, work)

        proto = graph_pb2.Graph()
        names = list(itertools.islice(writer.keys, work))

        for name in names:
            vertex_to_proto(self.graph.vertices[name], proto.vertices[name])

        writer.sections["graph"] += proto.SerializeToString(deterministic=True)

        if len(names) == work:
            return work

        writer.keys = None
        writer.phase = _EDGES

        return len(names)

    def _write_edges(self, writer: _Checkpoint, work: int) -> int:
        """Write the graph edges in vertex order, at most work entries per call, then the counts and defaults; returns the keys sorted or entries examined."""

        from .proto import graph_pb2

        if writer.keys is None:
            return writer.sort_step(self.graph.edges, work)

        spent = 0

        for u in writer.keys:
            neighbors = self.graph.edges[u]

            for v in sorted(neighbors):
                if u <= v:
                    proto = graph_pb2.Graph()
                    edge_to_proto(neighbors[v], proto.edges.add())
                    writer.sections["graph"] += proto.SerializeToString(
                        deterministic=True
                    )

            spent += max(len(neighbors), 1)

            if spent >= work:
                return spent

        proto = graph_pb2.Graph()
        proto.vertex_count = self.graph.vertex_count
        proto.edge_count = self.graph.edge_count

        for name, value in self.graph.default_vertex_attributes.items():
            proto.default_vertex_attributes[name] = value

        for name, value in self.graph.default_edge_attributes.items():
            proto.default_edge_attributes[name] = value

        writer.sections["graph"] += proto.SerializeToString(deterministic=True)
        writer.keys = None
        writer.phase = _ORDERED

        return spent

    def _write_ordered(self, writer: _Checkpoint, work: int) -> int:
        """Write the non-identity xforms of the live objects of one order() list; returns the slots examined."""

        items = getattr(self.objects, COLLECTIONS[writer.phase - _ORDERED][0])
        start = writer.cursor
        end = min(items.number_of_slots(), start + work)

        for slot in range(start, end):
            if items.is_dead(slot):
                continue

            guid = items.get_item(slot).guid
            xform = self.xforms.get(guid)

            if xform is None:
                continue

            writer.hits += 1

            if not xform.is_identity():
                self._write_xform(writer, guid, xform)

        writer.cursor = end

        if end >= items.number_of_slots():
            writer.cursor = 0
            writer.phase += 1

        return end - start

    def _write_rest(self, writer: _Checkpoint, work: int) -> int:
        """Write the non-identity xforms of guids outside order() in guid order, after a scan of xforms in slices of work that runs only when order() missed some; returns the entries examined or written."""

        if writer.hits < len(self.xforms):
            start = writer.cursor
            end = min(len(self.xforms), start + work)

            for guid, xform in itertools.islice(self.xforms.items(), start, end):
                geometry = self.lookup.get(guid)
                ordered = (
                    geometry is not None
                    and getattr(self.objects, _collection_of(geometry)[0]).get_slot(
                        guid
                    )
                    is not None
                )

                if not ordered and not xform.is_identity():
                    bisect.insort(writer.rest, guid)

            writer.cursor = end

            if end >= len(self.xforms):
                writer.hits = len(self.xforms)
                writer.cursor = 0

            return end - start

        start = writer.cursor
        end = min(len(writer.rest), start + work)

        for guid in writer.rest[start:end]:
            self._write_xform(writer, guid, self.xforms[guid])

        writer.cursor = end

        if end < len(writer.rest):
            return end - start

        writer.cursor = 0
        writer.phase = _INTERACTIONS

        if self.definition_lookup:
            writer.sections["definitions"] = _objects_head(self.definitions)
            writer.phase = _DEFINITIONS

        return end - start

    def _write_xform(self, writer: _Checkpoint, guid: str, xform: Xform) -> None:
        """Append one XformEntry to the xforms section."""

        from .proto import session_pb2

        proto = session_pb2.Session()
        item = proto.xforms.add()
        item.guid = guid
        item.xform.CopyFrom(xform.to_proto())
        writer.sections["xforms"] += proto.SerializeToString(deterministic=True)

    def _write_interactions(self, writer: _Checkpoint, work: int) -> int:
        """Write the interactions per edge guid in guid order, at most work per call; returns the keys sorted or entries written."""

        from .proto import session_pb2

        if writer.keys is None:
            return writer.sort_step(self.interactions, work)

        edges = list(itertools.islice(writer.keys, work))

        for edge in edges:
            proto = session_pb2.Session()
            item = proto.interactions.add()
            item.guid = edge

            for interaction in self.interactions[edge]:
                item.interactions.add().CopyFrom(interaction.to_proto())

            writer.sections["interactions"] += proto.SerializeToString(
                deterministic=True
            )

        if len(edges) == work:
            return work

        writer.keys = None
        writer.phase = _ASSEMBLY

        return len(edges)

    def _assemble(self, writer: _Checkpoint, work: int) -> bool:
        """Join the sections into one Session message, copying at most work KiB; True once complete."""

        from .proto import session_pb2

        if writer.phase == _ASSEMBLY:
            fields = session_pb2.Session.DESCRIPTOR.fields_by_name

            for name in _SECTIONS:
                if name == "definitions" and not self.definition_lookup:
                    continue

                body = writer.sections[name]
                chunks = writer.tree if name == "tree" else []

                if name in _FRAMED:
                    length = len(body) + sum(len(chunk) for chunk in chunks)
                    writer.pieces.append(_prefix(fields[name].number, length))

                writer.pieces.append(body)
                writer.pieces.extend(chunks)

            writer.phase += 1

        budget = work * 1024
        skip = len(writer.out)

        for piece in writer.pieces:
            if skip >= len(piece):
                skip -= len(piece)

                continue

            end = min(len(piece), skip + budget)
            writer.out += memoryview(piece)[skip:end]
            budget -= end - skip
            skip = 0

            if budget == 0:
                break

        return len(writer.out) == sum(len(piece) for piece in writer.pieces)

    def _kill(self, tomb: Tomb) -> None:
        """Flip a tomb dead: its slot and map entry, and for an object tomb its node, transform, vertex, edges and interactions; a guid a live twin owns keeps those with the twin; O(1 + d log V)."""

        if tomb.collection == "":
            return

        slot = tomb.slot
        self.revision += 1
        self.bvh_cache_dirty = True

        if tomb.definition:
            self._kill_definition(tomb)

            return

        items = getattr(self.objects, tomb.collection)
        stored = items.get_item(slot)
        guid = stored.guid
        table = self._table(tomb.collection)
        held = table.get(guid)
        owner = not self._twin(False, tomb.collection, slot, guid) and (
            held is not None or items.get_slot(guid) == slot
        )

        if owner and held is not None and held is not stored:
            items.set_item(slot, held)

        items.set_dead(slot, True)

        if owner:
            table.pop(guid, None)

        node = tomb.node

        if node is None:
            return

        parent = node.parent
        node.set_dead(True)
        node.set_tomb(tomb)

        if self.node_lookup.get(guid) is node:
            del self.node_lookup[guid]

        if parent is not None:
            self._queue(parent)

        if owner:
            self._park(tomb, guid)

    def _revive(self, tomb: Tomb) -> None:
        """Flip a tomb live again: the same slot and object, and for an object tomb the same node, transform, vertex, edges and interactions; a guid a live twin owns stays dead; O(1 + d log V)."""

        if tomb.collection == "":
            return

        slot = tomb.slot
        self.revision += 1
        self.bvh_cache_dirty = True

        if tomb.definition:
            self._revive_definition(tomb)

            return

        items = getattr(self.objects, tomb.collection)
        item = items.get_item(slot)
        guid = item.guid

        if self._twin(False, tomb.collection, slot, guid):
            return

        items.set_dead(slot, False)
        self._table(tomb.collection)[guid] = item
        node = tomb.node

        if node is None:
            self._label(guid, f"{_prefix_of(tomb.collection)}_{item.name}")

            return

        node.set_dead(False)

        if node.parent is not None:
            self.node_lookup[guid] = node

        self._unpark(tomb, guid)

    def _kill_definition(self, tomb: Tomb) -> None:
        """Flip a definition tomb dead: its slot, and its map entry when it owns the guid; O(1)."""

        slot = tomb.slot
        items = getattr(self.definitions, tomb.collection)
        stored = items.get_item(slot)
        guid = stored.guid
        owner = not self._twin(True, tomb.collection, slot, guid)
        held = self.definition_lookup.get(guid)

        if owner and held is not None and held is not stored:
            items.set_item(slot, held)

        items.set_dead(slot, True)

        if owner:
            del self.definition_lookup[guid]

    def _revive_definition(self, tomb: Tomb) -> None:
        """Flip a definition tomb live again, unless a live twin owns its guid; O(1)."""

        slot = tomb.slot
        items = getattr(self.definitions, tomb.collection)
        item = items.get_item(slot)
        guid = item.guid

        if self._twin(True, tomb.collection, slot, guid):
            return

        items.set_dead(slot, False)
        self.definition_lookup[guid] = item

    def _park(self, tomb: Tomb, guid: str) -> None:
        """Move the transform, graph vertex, incident edges and their interactions of guid into its tomb; O(d log V)."""

        tomb.xform = self.xforms.pop(guid, None)
        taken = self.graph.take_node(guid)

        if taken is None:
            return

        vertex, edges = taken

        for edge in edges:
            if edge.has_guid() and edge.guid in self.interactions:
                tomb.interactions[edge.guid] = self.interactions.pop(edge.guid)

        tomb.vertex = vertex
        tomb.edges = edges

    def _unpark(self, tomb: Tomb, guid: str) -> None:
        """Move a tomb's transform, vertex and edges back, with the interactions of every edge that returns; O(d log V)."""

        if tomb.xform is not None:
            self.xforms[guid] = tomb.xform
            tomb.xform = None

        if tomb.vertex is None:
            return

        vertex = tomb.vertex
        edges = tomb.edges
        tomb.vertex = None
        tomb.edges = []
        self.graph.put_node(vertex, edges)

        for edge in edges:
            if not edge.has_guid():
                continue

            back = self.graph.edges.get(guid, {}).get(edge.other_vertex(guid))

            if back is None or not back.has_guid() or back.guid != edge.guid:
                continue

            if edge.guid in tomb.interactions:
                self.interactions[edge.guid] = tomb.interactions.pop(edge.guid)

    def _swap(self, guid: str, obj: Any, entry: Entry) -> None:
        """Store obj under guid in the slot and map of the recorded entry, relabelling an object's vertex; a guid now live on the other side, or on the same side as another entry, is left alone; O(1)."""

        collection, prefix = _collection_for(obj)

        if entry.definition:
            items = getattr(self.definitions, collection)

            if self._is_live(guid) or items.get_slot(guid) != entry.tomb.slot:
                return

            items.set_item(entry.tomb.slot, obj)
            self.definition_lookup[guid] = obj
        else:
            if guid in self.definition_lookup or not self._owns(guid, entry.node):
                return

            items = getattr(self.objects, collection)
            slot = items.get_slot(guid)

            if slot is None:
                return

            items.set_item(slot, obj)
            self._label(guid, f"{prefix}_{obj.name}")
            self._table(collection)[guid] = obj

        self.revision += 1
        self.bvh_cache_dirty = True

    def _tree(self, op: TreeOp, back: bool) -> None:
        """Apply the before (back) or after state of a tree record: name, colour, liveness, and for a move the swap of node and ghost."""

        if back:
            name, color, dead = op.name_before, op.color_before, op.dead_before
        else:
            name, color, dead = op.name_after, op.color_after, op.dead_after

        was = op.node.is_dead()
        live = self._is_live(name)

        if op.ghost is not None:
            source = op.node.parent
            TreeNode.swap(op.node, op.ghost)
            op.ghost.set_tomb(op.tomb)

            if source is not None:
                self._queue(source)

        parent = op.node.parent
        op.node.name = name
        op.node.color = color
        op.node.set_dead(dead)
        self.revision += 1

        if dead and not was:
            op.node.set_tomb(op.tomb)

            if parent is not None:
                self._queue(parent)

        if dead and not was and not live:
            op.tomb.xform = self.xforms.pop(name, None)

        if was and not dead and not live and op.tomb.xform is not None:
            self.xforms[name] = op.tomb.xform
            op.tomb.xform = None

        if not live:
            return

        if not dead:
            self.node_lookup[name] = op.node
        elif self.node_lookup.get(name) is op.node:
            del self.node_lookup[name]

    def reindex(self) -> None:
        """Rebuild every index from the tables in O(n + N): the maps win over the slots, map-only and slot-only entries are adopted, a non-identity instance xform folds into xforms, node_lookup is refilled from the live tree."""

        for collection, prefix in COLLECTIONS[:-2]:
            _repoint(getattr(self.objects, collection), self.lookup)
            _repoint(getattr(self.definitions, collection), self.definition_lookup)

        _repoint(self.objects.components, self.component_lookup)
        _repoint(self.objects.instances, self.instance_lookup)
        _adopt(self.objects, self.lookup, "")
        _adopt(self.definitions, self.definition_lookup, "")
        _adopt(self.objects, self.component_lookup, "components")
        _adopt(self.objects, self.instance_lookup, "instances")

        for instance in self.objects.instances:
            if instance.xform.is_identity():
                continue

            self.xforms[instance.guid] = self.xform(instance.guid) * instance.xform
            instance.xform = Xform.identity()

        self.node_lookup.clear()

        for node in self.tree.nodes:
            if self._is_live(node.name) and node.name not in self.node_lookup:
                self.node_lookup[node.name] = node

        root = self.tree.root
        self._indexed = None if root is None else weakref.ref(root)

    def _place(self, guid: str, xform: Xform | None, node: TreeNode | None) -> None:
        """Set or drop (None) the local transform under guid, unrecorded; a guid whose entry is not the recorded node is left alone."""

        if not self._owns(guid, node):
            return

        if xform is not None:
            self.xforms[guid] = xform.duplicate()
        else:
            self.xforms.pop(guid, None)

        self.bvh_cache_dirty = True
        self.revision += 1

    def _xforms_ordered(self) -> list[tuple[str, Xform]]:
        """The xforms in canonical order() sequence, identity entries omitted, the exact sequence __jsondump__ and pb_dumps write."""

        ordered = []
        rest = {}

        for obj_guid, obj_xform in self.xforms.items():
            if not obj_xform.is_identity():
                rest[obj_guid] = obj_xform

        for obj_guid in self.order():
            obj_xform = rest.pop(obj_guid, None)

            if obj_xform is None:
                continue

            ordered.append((obj_guid, obj_xform))

        for obj_guid in sorted(rest):
            ordered.append((obj_guid, rest[obj_guid]))

        return ordered

    def _compute_boxes(self, guids: list[str]) -> list[OBB]:
        """World bounding box of every object in order() sequence, then of every instance, with the guid of each."""

        guids.clear()
        boxes = []
        world = self.world_xforms()

        for guid in self.order():
            geometry = self.lookup.get(guid)

            if geometry is None:
                continue

            boxes.append(
                self.compute_bounding_box(geometry, world.get(guid, Xform.identity()))
            )
            guids.append(guid)

        local: dict[str, OBB] = {}

        for instance in self.objects.instances:
            definition = self.definition_lookup.get(instance.definition_guid)

            if definition is None:
                continue

            if instance.definition_guid not in local:
                local[instance.definition_guid] = self.compute_bounding_box(
                    definition, Xform.identity()
                )

            placed = copy.deepcopy(local[instance.definition_guid])
            xform = world.get(instance.guid)

            if xform is not None:
                placed.transform(xform)

            boxes.append(placed)
            guids.append(instance.guid)

        return boxes

    def _rebuild_ray_bvh_cache(self) -> None:
        """Rebuild the cached SpatialBVH for ray casting."""

        self.cached_boxes = self._compute_boxes(self.cached_guids)

        if len(self.cached_boxes) == 0:
            self.cached_ray_bvh = SpatialBVH()
        else:
            self.cached_ray_bvh = SpatialBVH.from_boxes(
                self.cached_boxes, SpatialBVH.compute_world_size(self.cached_boxes)
            )

    def _ray_intersect_geometry(
        self, ray: Line, geometry: Any, tolerance: float, placement: Xform
    ) -> Point | None:
        """Test ray intersection with a specific geometry object, returning the world hit."""

        if isinstance(geometry, Point):
            return _ray_point(ray, geometry, tolerance)

        if isinstance(geometry, Line):
            return line_line(ray, geometry, tolerance)

        if isinstance(geometry, Plane):
            return line_plane(ray, geometry, True)

        if isinstance(geometry, Polyline):
            return _ray_polyline(ray, geometry, tolerance)

        if isinstance(geometry, PointCloud):
            return _ray_pointcloud(ray, geometry, tolerance)

        if isinstance(geometry, Mesh):
            return _ray_mesh(ray, geometry, tolerance, placement)

        if isinstance(geometry, OBB):
            hits = ray_box(ray, geometry, 0.0, 1.0)

            if not hits:
                return None

            return hits[0]

        return None
