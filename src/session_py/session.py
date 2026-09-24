from __future__ import annotations
from typing import Any
from typing import NamedTuple
from typing import TYPE_CHECKING
import copy
import json
import uuid
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
from .tree import Tree
from .tree import TreeNode
from .graph import Graph
from .history import History
from .history import AddOp
from .history import DefinitionOp
from .history import RemoveOp
from .history import ReplaceOp
from .history import XformOp
from .history import Tombstone
from .history import clone
from .spatial_bvh import SpatialBVH
from .tolerance import Tolerance
from .xform import Xform
from .intersection import line_line
from .intersection import line_plane
from .intersection import ray_box
from .intersection import ray_mesh_bvh

if TYPE_CHECKING:
    from pathlib import Path
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


def _locate(objects: Objects, guid: str) -> tuple[str, int]:
    """Which list of objects holds a guid, and where; ("", -1) when none does."""

    for collection, prefix in COLLECTIONS:
        items = getattr(objects, collection)

        for i in range(len(items)):
            if items[i].guid == guid:
                return collection, i

    return "", -1


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


def _index_geometry(objects: Objects, lookup: dict[str, Any]) -> None:
    """Every geometry of objects under its guid."""

    for collection, prefix in COLLECTIONS:
        if collection == "components" or collection == "instances":
            continue

        for item in getattr(objects, collection):
            lookup[item.guid] = item


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


def _edge_guid(graph: Graph, a: str, b: str) -> str:
    """The guid of the edge between a and b, from whichever stored copy has one; "" when neither was minted."""

    forward = graph.edges[a][b]
    backward = graph.edges[b][a]

    if forward.has_guid():
        return forward.guid

    return backward.guid if backward.has_guid() else ""


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


def _registered(session: Session, guid: str) -> bool:
    """Whether guid is a graph node held by an object, instance or component."""

    return session.graph.has_node(guid) and (
        guid in session.lookup
        or guid in session.instance_lookup
        or guid in session.component_lookup
    )


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
        self.history = History()  # Undo/redo buffer, purged by every save.
        self.bvh = SpatialBVH()  # Bounding volume hierarchy for collision detection.
        self.cached_ray_bvh = SpatialBVH()  # Cached SpatialBVH for ray casting.
        self.cached_guids: list[str] = []  # GUID per leaf of cached_ray_bvh.
        self.cached_boxes: list[OBB] = []  # Box per leaf of cached_ray_bvh.
        self.bvh_cache_dirty = True  # Flag to rebuild cached_ray_bvh.

        self.tree.add(TreeNode(name=self.name))

    def __deepcopy__(self, memo):
        """Copy every table and object, guids included; caches are rebuilt on demand and history starts empty."""

        result = Session(self.name)

        if self.has_guid():
            result.guid = self.guid

        result.objects = _clone_objects(self.objects)
        result.tree = copy.deepcopy(self.tree, memo)
        result.graph = copy.deepcopy(self.graph, memo)
        result.xforms = copy.deepcopy(self.xforms, memo)
        result.definitions = _clone_objects(self.definitions)
        result._index_objects()
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
        node = self.tree.get_node_by_name(guid)

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
        """Add a point; None adds nothing and returns None."""

        if point is None:
            return None

        return self._add_object("points", point, "point", parent)

    def add_line(
        self, line: Line | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a line; None adds nothing and returns None."""

        if line is None:
            return None

        return self._add_object("lines", line, "line", parent)

    def add_plane(
        self, plane: Plane | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a plane; None adds nothing and returns None."""

        if plane is None:
            return None

        return self._add_object("planes", plane, "plane", parent)

    def add_obb(self, bbox: OBB | None) -> TreeNode | None:
        """Add a bounding box; None adds nothing and returns None."""

        if bbox is None:
            return None

        return self._add_object("bboxes", bbox, "bbox", None)

    def add_polyline(
        self, polyline: Polyline | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a polyline; None, or fewer than two points, adds nothing and returns None."""

        if polyline is None or polyline.point_count() < 2:
            return None

        return self._add_object("polylines", polyline, "polyline", parent)

    def add_pointcloud(
        self, pointcloud: PointCloud | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a point cloud; None, or no points, adds nothing and returns None."""

        if pointcloud is None or pointcloud.is_empty():
            return None

        return self._add_object("pointclouds", pointcloud, "pointcloud", parent)

    def add_mesh(
        self, mesh: Mesh | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a mesh; None, or no faces, adds nothing and returns None."""

        if mesh is None or mesh.is_empty() or mesh.number_of_faces() == 0:
            return None

        return self._add_object("meshes", mesh, "mesh", parent)

    def add_nurbscurve(
        self, nurbscurve: NurbsCurve | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a curve; None, or fewer than two control vertices, adds nothing and returns None."""

        if nurbscurve is None or nurbscurve.cv_count() < 2:
            return None

        return self._add_object("nurbscurves", nurbscurve, "nurbscurve", parent)

    def add_nurbssurface(
        self, nurbssurface: NurbsSurface | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a surface; None, or no control vertices, adds nothing and returns None."""

        if nurbssurface is None or nurbssurface.cv_count() == 0:
            return None

        return self._add_object("nurbssurfaces", nurbssurface, "nurbssurface", parent)

    def add_brep(
        self, brep: BRep | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add a brep; None, or no faces and no vertices, adds nothing and returns None."""

        if brep is None or (brep.face_count() == 0 and brep.vertex_count() == 0):
            return None

        return self._add_object("breps", brep, "brep", parent)

    def add_element(
        self, element: Element | None, parent: TreeNode | None = None
    ) -> TreeNode | None:
        """Add an element; only None adds nothing, an Element is a data record kept even without geometry."""

        if element is None:
            return None

        return self._add_object("elements", element, "element", parent)

    def add_component(
        self, component: Component, parent: TreeNode | None = None
    ) -> TreeNode:
        """Add a custom component (any object with type_name/guid/name/extra)."""
        return self._add_object("components", component, "component", parent)

    def add_definition(self, definition: Any) -> str:
        """Add a definition, geometry in its own frame that instances share; returns its guid, also when that guid is already defined, and "" for None or a guid an object, instance or component holds."""

        if definition is None:
            return ""

        guid = definition.guid

        if guid in self.definition_lookup:
            return guid

        if (
            guid in self.lookup
            or guid in self.instance_lookup
            or guid in self.component_lookup
        ):
            return ""

        if self.history.current is not None:
            self.history.record(DefinitionOp(guid, None, clone(definition)))

        self._define(guid, definition)

        return guid

    def add_instance(
        self,
        instance: InstanceRef,
        xform: Xform | None = None,
        parent: TreeNode | None = None,
    ) -> TreeNode | None:
        """Add an instance under parent, placed by xform relative to the parent with its own xform folded in; None when None or its definition_guid names no definition."""

        if instance is None or instance.definition_guid not in self.definition_lookup:
            return None

        placement = (xform if xform is not None else Xform.identity()) * instance.xform
        instance.xform = Xform.identity()
        node = self._add_object("instances", instance, "instance", parent)

        if not placement.is_identity():
            self.set_xform(instance.guid, placement)

        return node

    def add(self, node: TreeNode | None, parent: TreeNode | None = None) -> None:
        """Add a TreeNode to the tree hierarchy, under the root when no parent is given; None is ignored."""

        if node is None:
            return

        if parent is None:
            self.tree.add(node, self.tree.root)
        else:
            self.tree.add(node, parent)

    def add_group(self, group_name: str) -> TreeNode:
        """Create a named group (TreeNode) and add it to the root of the tree."""

        node = TreeNode(name=group_name)
        self.add(node)

        return node

    def add_edge(self, guid1: str, guid2: str, attribute: str = "") -> None:
        """Add an edge between two geometry objects in the graph."""
        self.graph.add_edge(guid1, guid2, attribute)

    def add_interaction(self, a: str, b: str) -> tuple[str, str]:
        """Add or reuse an undirected interaction edge between registered objects; returns its stored endpoint order. Raises ValueError for missing objects or a self-pair. Preserves an existing edge's attributes and guid."""

        if a == b or not _registered(self, a) or not _registered(self, b):
            raise ValueError(
                "Session.add_interaction: add two distinct objects to the session first"
            )

        if not self.has_interaction(a, b):
            self.graph.add_edge(a, b)

        edge = self.graph.edges[a][b]
        self.graph.edges[b][a].guid = edge.guid

        return edge.v0, edge.v1

    def has_interaction(self, a: str, b: str) -> bool:
        """True when the pair has an interaction edge in either order."""
        return self.graph.has_edge((a, b)) or self.graph.has_edge((b, a))

    def remove_interaction(self, a: str, b: str) -> None:
        """Remove the pair's edge in either order; a missing pair is a no-op."""

        if self.graph.has_edge((a, b)):
            self.graph.remove_edge((a, b))
        elif self.graph.has_edge((b, a)):
            self.graph.remove_edge((b, a))

    def add_hierarchy(self, parent_guid: str, child_guid: str) -> bool:
        """Add a parent-child relationship in the tree."""
        return self.tree.add_child_by_guid(parent_guid, child_guid)

    def add_relationship(
        self, from_guid: str, to_guid: str, relationship_type: str = "default"
    ) -> None:
        """Add a relationship edge in the graph."""
        self.graph.add_edge(from_guid, to_guid, relationship_type)

    def remove_object(self, obj_guid: str) -> bool:
        """Remove an object by its GUID from every live table at once; the removal record is the tombstone undo restores from."""

        op = self._detach(obj_guid)

        if op is None:
            return False

        self.history.record(op)

        return True

    def replace(self, guid: str, obj: Any) -> bool:
        """Swap the object stored under guid for obj, which takes over that guid; the recorded edit undo and redo restore as absolute snapshots."""

        before = self.lookup.get(guid)

        if before is None:
            return False

        obj.guid = guid

        if self.history.current is not None:
            self.history.record(ReplaceOp(guid, clone(before), clone(obj)))

        self._swap(guid, obj)

        return True

    def replace_definition(self, guid: str, definition: Any) -> bool:
        """Swap the geometry of a definition, which keeps its guid, so every instance of it changes at once; False when guid is no definition."""

        before = self.definition_lookup.get(guid)

        if before is None:
            return False

        definition.guid = guid

        if self.history.current is not None:
            self.history.record(DefinitionOp(guid, clone(before), clone(definition)))

        self._define(guid, definition)

        return True

    def remove_definition(self, guid: str) -> bool:
        """Remove a definition; False when guid is no definition or an instance still names it."""

        before = self.definition_lookup.get(guid)

        if before is None or len(self.instances_of(guid)) > 0:
            return False

        if self.history.current is not None:
            self.history.record(DefinitionOp(guid, clone(before), None))

        self._define(guid, None)

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
        removed = self._detach(guid)

        if removed is None:
            return False

        added = AddOp(
            guid,
            instance,
            "instances",
            len(self.objects.instances),
            None if placement.is_identity() else placement,
            removed.parent_guid,
            removed.index,
            removed.node,
            f"instance_{instance.name}",
            removed.edges,
        )

        self.history.record(removed)
        self.history.record(added)
        self._attach(added)

        return True

    def explode(self, instance_guid: str) -> bool:
        """Turn an instance into a standalone copy of its definition in the definition frame, keeping its guid, name, transform, tree node and edges, and on an element its features."""

        definition = self.definition_of(instance_guid)

        if definition is None:
            return False

        instance = self.instance_lookup[instance_guid]
        result = _resolve(instance, definition, Xform.identity())
        collection, prefix = _collection_of(result)
        size = len(getattr(self.objects, collection))
        removed = self._detach(instance_guid)

        if removed is None:
            return False

        added = AddOp(
            instance_guid,
            result,
            collection,
            size,
            removed.xform,
            removed.parent_guid,
            removed.index,
            removed.node,
            f"{prefix}_{instance.name}",
            removed.edges,
        )

        self.history.record(removed)
        self.history.record(added)
        self._attach(added)

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
            self.history.record(XformOp(guid, self.xforms.get(guid), xform))

        self.xforms[guid] = xform
        self.bvh_cache_dirty = True

    def remove_xform(self, guid: str) -> bool:
        """Removes an object's local transform, returning whether one was present."""

        before = self.xforms.get(guid)

        if before is None:
            return False

        if self.history.current is not None:
            self.history.record(XformOp(guid, before, None))

        del self.xforms[guid]
        self.bvh_cache_dirty = True

        return True

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
        return self.history.undo(self)

    def redo(self) -> bool:
        """Reapply the latest undone transaction, returning whether there was one."""
        return self.history.redo(self)

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

        data = {}

        if self.definition_lookup:
            data["definitions"] = self.definitions.__jsondump__()

        data["graph"] = self.graph.__jsondump__()
        data["guid"] = self.guid
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

        session._index_objects()

        return session

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""

        self.history.clear()

        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Session:
        """Deserialize from a JSON string."""
        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        self.history.clear()

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

        session._index_objects()

        return session

    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

        self.history.clear()

        return self.to_proto().SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Session:
        """Deserialize from protobuf bytes."""

        from .proto import session_pb2

        proto = session_pb2.Session()
        proto.ParseFromString(data)

        return cls.from_proto(proto)

    def pb_dump(self, filename: str | Path) -> None:
        """Write to a protobuf file."""

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
    ) -> TreeNode:
        """Store an object in its typed list, lookup, graph and tree, recording an AddOp when a transaction is open."""

        guid = obj.guid
        items = getattr(self.objects, collection)
        items.append(obj)
        obj_index = len(items) - 1

        if collection == "components":
            self.component_lookup[guid] = obj
        elif collection == "instances":
            self.instance_lookup[guid] = obj
        else:
            self.lookup[guid] = obj

        attribute = f"{type_prefix}_{obj.name}"
        self.graph.add_node(guid, attribute)
        self.bvh_cache_dirty = True
        node = TreeNode(name=guid)
        host = parent if parent is not None else self.tree.root
        parent_guid = None
        index = 0

        if host is not None:
            self.add(node, host)
            parent_guid = host.name
            index = len(host.children) - 1

        if self.history.current is not None:
            self.history.record(
                AddOp(
                    guid,
                    clone(obj),
                    collection,
                    obj_index,
                    None,
                    parent_guid,
                    index,
                    None,
                    attribute,
                    [],
                )
            )

        return node

    def _locate(self, guid: str) -> tuple[str, int]:
        """Which Objects list holds a guid, and where; ("", -1) when none does."""
        return _locate(self.objects, guid)

    def _detach(self, guid: str) -> RemoveOp | None:
        """Take an object out of every live table, unrecorded, returning its tombstone."""

        obj = self.lookup.get(guid)

        if obj is None:
            obj = self.component_lookup.get(guid)

        if obj is None:
            obj = self.instance_lookup.get(guid)

        if obj is None:
            return None

        collection, obj_index = self._locate(guid)

        if obj_index >= 0:
            getattr(self.objects, collection).pop(obj_index)

        self.lookup.pop(guid, None)
        self.component_lookup.pop(guid, None)
        self.instance_lookup.pop(guid, None)
        xform = self.xforms.pop(guid, None)
        self.bvh_cache_dirty = True
        parent_guid = None
        index = 0
        node = self.tree.get_node_by_name(guid)

        if node is not None:
            parent = node.parent

            if parent is not None:
                parent_guid = parent.name
                index = parent.children.index(node)

            node = self.tree.remove(node)

        attribute = ""
        edges = []

        if self.graph.has_node(guid):
            attribute = self.graph.node_label(guid)

            for other, label, forward in self.graph.edges_of(guid):
                edges.append(
                    (other, label, forward, _edge_guid(self.graph, guid, other))
                )

            self.graph.remove_node(guid)

        return RemoveOp(
            guid,
            clone(obj),
            collection,
            obj_index,
            xform,
            parent_guid,
            index,
            node,
            attribute,
            edges,
        )

    def _attach(self, op: Tombstone) -> None:
        """Put an object back from its tombstone, unrecorded: typed list, lookup, xform, tree node with its subtree, graph node and edges."""

        obj = clone(op.obj)
        items = getattr(self.objects, op.collection)
        items.insert(min(op.obj_index, len(items)), obj)

        if op.collection == "components":
            self.component_lookup[op.guid] = obj
        elif op.collection == "instances":
            self.instance_lookup[op.guid] = obj
        else:
            self.lookup[op.guid] = obj

        if op.xform is not None:
            self.xforms[op.guid] = op.xform

        self.bvh_cache_dirty = True
        node = op.node

        if node is None:
            node = TreeNode(name=op.guid)

        if op.parent_guid is not None:
            parent = self.tree.get_node_by_name(op.parent_guid)

            if parent is not None:
                self.tree.add(node, parent)
                children = list(parent.children)

                for i in range(min(op.index, len(children) - 1), len(children) - 1):
                    parent.add(parent.remove(children[i]))

        self.graph.add_node(op.guid, op.attribute)

        for other, attribute, forward, id in op.edges:
            if not self.graph.has_node(other):
                continue

            if forward:
                self.graph.add_edge(op.guid, other, attribute)
            else:
                self.graph.add_edge(other, op.guid, attribute)

            if id == "":
                continue

            self.graph.edges[op.guid][other].guid = id
            self.graph.edges[other][op.guid].guid = id

    def _swap(self, guid: str, obj: Any) -> None:
        """Store obj under guid in its typed list and lookup, unrecorded."""

        collection, obj_index = self._locate(guid)

        if obj_index < 0:
            return

        getattr(self.objects, collection)[obj_index] = obj

        if collection == "components":
            self.component_lookup[guid] = obj
        elif collection == "instances":
            self.instance_lookup[guid] = obj
        else:
            self.lookup[guid] = obj

        self.bvh_cache_dirty = True
        attribute = ""

        for name, prefix in COLLECTIONS:
            if name == collection:
                attribute = f"{prefix}_{obj.name}"

        if self.graph.has_node(guid):
            self.graph.node_label(guid, attribute)

    def _index_objects(self) -> None:
        """Point every lookup at the objects and definitions this session holds, folding a non-identity instance xform into xforms."""

        self.lookup.clear()
        self.component_lookup.clear()
        self.instance_lookup.clear()
        self.definition_lookup.clear()
        _index_geometry(self.objects, self.lookup)
        _index_geometry(self.definitions, self.definition_lookup)

        for component in self.objects.components:
            self.component_lookup[component.guid] = component

        for instance in self.objects.instances:
            self.instance_lookup[instance.guid] = instance

            if instance.xform.is_identity():
                continue

            self.xforms[instance.guid] = self.xform(instance.guid) * instance.xform
            instance.xform = Xform.identity()

    def _define(self, guid: str, definition: Any | None) -> None:
        """Set or drop (None) a definition under guid, unrecorded."""

        collection, position = _locate(self.definitions, guid)

        if position >= 0:
            getattr(self.definitions, collection).pop(position)

        self.definition_lookup.pop(guid, None)
        self.bvh_cache_dirty = True

        if definition is None:
            return

        items = getattr(self.definitions, _collection_of(definition)[0])
        at = len(items) if position < 0 else min(position, len(items))
        items.insert(at, definition)
        self.definition_lookup[guid] = definition

    def _place(self, guid: str, xform: Xform | None) -> None:
        """Set or drop (None) the local transform under guid, unrecorded."""

        if xform is not None:
            self.xforms[guid] = xform
        else:
            self.xforms.pop(guid, None)

        self.bvh_cache_dirty = True

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
