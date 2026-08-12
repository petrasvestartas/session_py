import uuid
from typing import Any, Dict, List, Tuple, Optional, NamedTuple
from .objects import Objects
from .point import Point
from .tree import Tree, TreeNode
from .graph import Graph
from .spatial_bvh import SpatialBVH
from .obb import OBB
from .tolerance import Tolerance
from .xform import Xform


class RayHit(NamedTuple):
    guid: str
    point: Point
    distance: float


class Session:
    """A Session containing geometry objects with hierarchical and graph structures.

    The Session class manages collections of geometry objects and provides:
    - Fast GUID-based lookup
    - Hierarchical tree structure for organization
    - Graph structure for object relationships
    - JSON serialization/deserialization

    Parameters
    ----------
    name : str, optional
        Name of the Session. Defaults to "Session".

    Attributes
    ----------
    objects : :class:`Objects`
        Collection of geometry objects in the Session.
    lookup : dict[UUID, :class:`Point`]
        Fast lookup dictionary mapping GUIDs to geometry objects.
    tree : :class:`Tree`
        Hierarchical tree structure for organizing geometry objects.
    graph : :class:`Graph`
        Graph structure for storing relationships between geometry objects.
    xforms : dict[str, :class:`Xform`]
        Guid -> LOCAL transform, relative to the tree parent. THE only place a transform is
        stored: geometry types carry no transformation member. Cumulative placement comes from
        `world_xform`, which multiplies down the tree. Serialized explicitly by
        __jsondump__/pb_dumps in `order()` sequence (a dict has no deterministic order).
    name : str
        Name of the Session.

    """

    def __init__(self, name="my_session"):
        self._guid = None
        self.name = name
        self.objects = Objects()
        self.lookup: Dict[str, Any] = {}
        self.tree = Tree(name=f"{name}_tree")
        self.graph = Graph(name=f"{name}_graph")
        self.xforms: Dict[str, Xform] = {}
        # SpatialBVH for collision detection (auto-computed world size)
        self.bvh = SpatialBVH()

        # Create empty root node with session name
        root_node = TreeNode(name=self.name)
        self.tree.add(root_node)

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str):
        self._guid = value

    def __str__(self) -> str:
        return f"Session(objects={self.objects.to_str()}, tree={self.tree.to_str()}, graph={self.graph.to_str()})"

    def __repr__(self) -> str:
        return f"Session({self.guid}, {self.name}, {self.objects.to_str()}, {self.tree.to_str()}, {self.graph.to_str()})"

    ###########################################################################################
    # XFORMS - the one place a transformation is stored
    ###########################################################################################

    def set_xform(self, guid: str, xform: Xform) -> None:
        """Set the LOCAL transform of an object, relative to its tree parent."""
        self.xforms[guid] = xform

    def xform(self, guid: str) -> Xform:
        """The LOCAL transform of an object, identity when none was set."""
        return self.xforms.get(guid, Xform.identity())

    def remove_xform(self, guid: str) -> bool:
        """Remove an object's local transform, returning whether one was present."""
        return self.xforms.pop(guid, None) is not None

    def world_xform(self, guid: str) -> Xform:
        """The CUMULATIVE placement of an object: every ancestor's transform multiplied down
        the tree onto its own. An object with no tree node is its own root and returns its
        local transform - objects added without a parent are never attached, so treating a
        missing node as identity would silently move them to the origin.
        """
        acc = self.xform(guid)
        node = self.tree.get_node_by_name(guid)
        if node is not None:
            # ancestors() runs immediate parent -> root, so left-multiplying each in turn
            # yields root * ... * parent * local - the same order the tree walk composes.
            for ancestor in node.ancestors:
                xf = self.xforms.get(ancestor.name)
                if xf is not None:
                    acc = xf * acc
        return acc

    def world_xforms(self) -> Dict[str, Xform]:
        """Every object's cumulative placement, computed in ONE downward pass. Use this instead
        of calling `world_xform` per object: that does a whole-tree scan to find each node,
        which is quadratic over a session.
        """
        out: Dict[str, Xform] = {}

        def walk(node: TreeNode, parent_xform: Xform) -> None:
            local = self.xforms.get(node.name)
            current = parent_xform * local if local is not None else parent_xform
            out[node.name] = current
            for child in node.children:
                walk(child, current)

        if self.tree.root is not None:
            walk(self.tree.root, Xform.identity())
        # Objects that were added without a parent have no tree node; they are their own roots.
        for guid, xform in self.xforms.items():
            out.setdefault(guid, xform)
        return out

    def _xforms_ordered(self) -> List[Tuple[str, Xform]]:
        """The xforms in canonical `order()` sequence, identity entries omitted - the exact
        sequence __jsondump__ and pb_dumps write, so both formats share one order."""
        ordered = []
        for guid in self.order():
            xform = self.xforms.get(guid)
            if xform is not None and not xform.is_identity():
                ordered.append((guid, xform))
        # Group nodes carry transforms but hold no geometry, so they are absent from order();
        # they follow sorted by guid, which keeps the sequence deterministic across languages.
        listed = {guid for guid, _ in ordered}
        rest = [
            (guid, xform)
            for guid, xform in self.xforms.items()
            if guid not in listed and not xform.is_identity()
        ]
        rest.sort(key=lambda item: item[0])
        ordered.extend(rest)
        return ordered

    ###########################################################################################
    # JSON (polymorphic)
    ###########################################################################################

    def __jsondump__(self) -> dict:
        """Serialize to polymorphic JSON format with type field."""
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "objects": self.objects.__jsondump__(),
            "tree": self.tree.__jsondump__(),
            "graph": self.graph.__jsondump__(),
            "xforms": [
                {"guid": guid, "xform": xform.__jsondump__()}
                for guid, xform in self._xforms_ordered()
            ],
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: Optional[str] = None, name: Optional[str] = None
    ) -> "Session":
        """Deserialize from polymorphic JSON format."""
        from .file_encoders import file_decode_node

        session = cls(name=data.get("name", "my_session"))
        session.guid = guid if guid is not None else data.get("guid", session.guid)

        # Load nested structures via file_decode_node
        if data.get("objects"):
            session.objects = file_decode_node(data["objects"])  # Objects
        if data.get("tree"):
            session.tree = file_decode_node(data["tree"])  # Tree
        if data.get("graph"):
            session.graph = file_decode_node(data["graph"])  # Graph

        for entry in data.get("xforms", []):
            xform_data = entry.get("xform")
            if xform_data:
                session.xforms[entry["guid"]] = Xform.__jsonload__(
                    xform_data, xform_data.get("guid"), xform_data.get("name")
                )

        # Rebuild lookup from all objects
        for point in session.objects.points:
            session.lookup[point.guid] = point
        for line in session.objects.lines:
            session.lookup[line.guid] = line
        for plane in session.objects.planes:
            session.lookup[plane.guid] = plane
        for bbox in session.objects.bboxes:
            session.lookup[bbox.guid] = bbox
        for polyline in session.objects.polylines:
            session.lookup[polyline.guid] = polyline
        for pointcloud in session.objects.pointclouds:
            session.lookup[pointcloud.guid] = pointcloud
        for mesh in session.objects.meshes:
            session.lookup[mesh.guid] = mesh
        for nurbscurve in session.objects.nurbscurves:
            session.lookup[nurbscurve.guid] = nurbscurve
        for nurbssurface in session.objects.nurbssurfaces:
            session.lookup[nurbssurface.guid] = nurbssurface
        for brep in session.objects.breps:
            session.lookup[brep.guid] = brep
        for element in session.objects.elements:
            session.lookup[element.guid] = element
        for component in session.objects.components:
            session.lookup[component.guid] = component

        return session

    def file_json_dumps(self):
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s):
        import json
        return cls.__jsonload__(json.loads(s))

    def file_json_dump(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            return cls.__jsonload__(json.load(f))

    def pb_dumps(self):
        from .proto import session_pb2
        proto = session_pb2.Session()
        proto.name = self.name
        proto.guid = self.guid
        proto.objects.ParseFromString(self.objects.pb_dumps())
        proto.tree.ParseFromString(self.tree.pb_dumps())
        proto.graph.ParseFromString(self.graph.pb_dumps())
        # Xforms in canonical order() sequence - a map would not be deterministic
        for guid, xform in self._xforms_ordered():
            entry = proto.xforms.add()
            entry.guid = guid
            entry.xform.guid = xform.guid
            entry.xform.name = xform.name
            entry.xform.matrix.extend(xform.m)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import session_pb2
        proto = session_pb2.Session()
        proto.ParseFromString(data)
        session = cls(name=proto.name)
        session.guid = proto.guid
        session.objects = Objects.from_proto(proto.objects)
        session.tree = Tree.pb_loads(proto.tree.SerializeToString())
        session.graph = Graph.pb_loads(proto.graph.SerializeToString())
        for entry in proto.xforms:
            if entry.HasField("xform"):
                xform = Xform.from_matrix(list(entry.xform.matrix))
                xform.guid = entry.xform.guid
                xform.name = entry.xform.name
                session.xforms[entry.guid] = xform
        for point in session.objects.points:
            session.lookup[point.guid] = point
        for line in session.objects.lines:
            session.lookup[line.guid] = line
        for plane in session.objects.planes:
            session.lookup[plane.guid] = plane
        for bbox in session.objects.bboxes:
            session.lookup[bbox.guid] = bbox
        for polyline in session.objects.polylines:
            session.lookup[polyline.guid] = polyline
        for pointcloud in session.objects.pointclouds:
            session.lookup[pointcloud.guid] = pointcloud
        for mesh in session.objects.meshes:
            session.lookup[mesh.guid] = mesh
        for nurbscurve in session.objects.nurbscurves:
            session.lookup[nurbscurve.guid] = nurbscurve
        for nurbssurface in session.objects.nurbssurfaces:
            session.lookup[nurbssurface.guid] = nurbssurface
        for brep in session.objects.breps:
            session.lookup[brep.guid] = brep
        for element in session.objects.elements:
            session.lookup[element.guid] = element
        return session

    def pb_dump(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())

    ###########################################################################################
    # Details - Add objects
    ###########################################################################################

    def _add_object(self, collection, obj, type_prefix, parent=None):
        collection.append(obj)
        self.lookup[obj.guid] = obj
        self.graph.add_node(obj.guid, f"{type_prefix}_{obj.name}")
        node = TreeNode(name=obj.guid)
        if parent is not None:
            self.add(node, parent)
        return node

    def order(self):
        """Canonical object order: the objects lists walked in one fixed type sequence -
        deterministic across runs AND languages (lookup/map iteration is neither).
        Viewers and reconcile key their rows off this."""
        return (
            [p.guid for p in self.objects.points]
            + [l.guid for l in self.objects.lines]
            + [p.guid for p in self.objects.planes]
            + [b.guid for b in self.objects.bboxes]
            + [p.guid for p in self.objects.polylines]
            + [p.guid for p in self.objects.pointclouds]
            + [m.guid for m in self.objects.meshes]
            + [n.guid for n in self.objects.nurbscurves]
            + [n.guid for n in self.objects.nurbssurfaces]
            + [b.guid for b in self.objects.breps]
            + [e.guid for e in self.objects.elements]
        )

    def add_point(self, point, parent=None) -> TreeNode:
        return self._add_object(self.objects.points, point, "point", parent)

    def add_line(self, line, parent=None) -> TreeNode:
        return self._add_object(self.objects.lines, line, "line", parent)

    def add_plane(self, plane, parent=None) -> TreeNode:
        return self._add_object(self.objects.planes, plane, "plane", parent)

    def add_obb(self, bbox, parent=None) -> TreeNode:
        return self._add_object(self.objects.bboxes, bbox, "bbox", parent)

    def add_polyline(self, polyline, parent=None) -> TreeNode:
        return self._add_object(self.objects.polylines, polyline, "polyline", parent)

    def add_pointcloud(self, pointcloud, parent=None) -> TreeNode:
        return self._add_object(self.objects.pointclouds, pointcloud, "pointcloud", parent)

    def add_mesh(self, mesh, parent=None) -> TreeNode:
        return self._add_object(self.objects.meshes, mesh, "mesh", parent)

    def add_nurbscurve(self, nurbscurve, parent=None) -> TreeNode:
        return self._add_object(self.objects.nurbscurves, nurbscurve, "nurbscurve", parent)

    def add_nurbssurface(self, nurbssurface, parent=None) -> TreeNode:
        return self._add_object(self.objects.nurbssurfaces, nurbssurface, "nurbssurface", parent)

    def add_brep(self, brep, parent=None) -> TreeNode:
        return self._add_object(self.objects.breps, brep, "brep", parent)

    def add_element(self, element, parent=None) -> TreeNode:
        return self._add_object(self.objects.elements, element, "element", parent)

    def add_component(self, component, parent=None) -> TreeNode:
        """Add a custom component (any object with guid, name, __jsondump__, __jsonload__)."""
        return self._add_object(self.objects.components, component, "component", parent)

    def add_group(self, name: str) -> TreeNode:
        node = TreeNode(name=name)
        self.add(node)
        return node

    def find_group(self, name: str) -> TreeNode:
        """Find an existing group by name.

        Raises ValueError if the group does not exist.
        """
        root = self.tree.root
        if root is not None:
            for child in root.children:
                if child.name == name:
                    return child
        raise ValueError(f"Group '{name}' not found")

    def compute_face_to_face(self, inflate=5.0, coplanar_tolerance=50.0):
        from .intersection import adjacency_search, face_to_face
        from .polyline import Polyline
        elems = self.objects.elements
        N = len(elems)
        if N == 0:
            return
        all_polys = [e.compute_polylines() for e in elems]
        all_planes = [e.compute_planes() for e in elems]
        from .aabb import AABB
        aabbs = []
        for polys in all_polys:
            pts = []
            for pl in polys:
                pts.extend(pl.get_points())
            aabbs.append(AABB.from_points(pts, inflate) if pts else AABB.from_point(Point(0,0,0), inflate))
        adjacency = []
        for i in range(N):
            for j in range(i+1, N):
                if aabbs[i].intersects(aabbs[j]):
                    adjacency.extend([i, j, -1, -1])
        joints = face_to_face(adjacency, all_polys, all_planes, coplanar_tolerance)
        g = self.add_group("Joints")
        for k, (a, b, fi, fj, type_val, poly) in enumerate(joints):
            jpl = Polyline(poly.get_points()) if not isinstance(poly, Polyline) else poly
            jpl.name = f"joint_{k}"
            self.add_polyline(jpl, g)
            self.add_edge(elems[a].guid, elems[b].guid,
                f"{fi},{fj},{type_val},{jpl.guid}")

    def add(self, node: TreeNode, parent: TreeNode = None) -> None:
        """Add a TreeNode to the tree hierarchy.

        Parameters
        ----------
        node : TreeNode
            The TreeNode to add.
        parent : TreeNode, optional
            Parent TreeNode (defaults to root if not provided).
        """
        if parent is None:
            self.tree.add(node, self.tree.root)
        else:
            self.tree.add(node, parent)

    def add_edge(self, guid1: str, guid2: str, attribute: str = "") -> None:
        """Add an edge between two geometry objects in the graph.

        Parameters
        ----------
        guid1 : str
            GUID of the first geometry object.
        guid2 : str
            GUID of the second geometry object.
        attribute : str, optional
            Edge attribute description.
        """
        self.graph.add_edge(guid1, guid2, attribute)

    ###########################################################################################
    # Details - Lookup
    ###########################################################################################

    def get_object(self, guid: str) -> Optional[Point]:
        """Get a geometry object by its GUID.

        Parameters
        ----------
        guid : str
            The string GUID of the geometry object to retrieve.

        Returns
        -------
        :class:`Point` | None
            The geometry object if found, None otherwise.
        """
        return self.lookup.get(guid)

    def remove_object(self, guid: str) -> bool:
        """Remove a geometry object by its GUID.

        Args:
            guid: The UUID of the geometry object to remove.

        Returns:
            True if the object was removed, False if not found.
        """
        geometry = self.lookup.get(guid)
        if not geometry:
            return False

        # Remove from all object collections
        self.objects.points = [p for p in self.objects.points if p.guid != guid]
        self.objects.lines = [l for l in self.objects.lines if l.guid != guid]
        self.objects.polylines = [p for p in self.objects.polylines if p.guid != guid]
        self.objects.planes = [p for p in self.objects.planes if p.guid != guid]
        self.objects.bboxes = [b for b in self.objects.bboxes if b.guid != guid]
        self.objects.meshes = [m for m in self.objects.meshes if m.guid != guid]
        self.objects.pointclouds = [p for p in self.objects.pointclouds if p.guid != guid]
        self.objects.nurbscurves = [c for c in self.objects.nurbscurves if c.guid != guid]
        self.objects.nurbssurfaces = [s for s in self.objects.nurbssurfaces if s.guid != guid]
        self.objects.breps = [b for b in self.objects.breps if b.guid != guid]
        self.objects.elements = [e for e in self.objects.elements if e.guid != guid]

        # Remove from lookup table
        del self.lookup[guid]
        self.xforms.pop(guid, None)

        # Remove from tree - find node by guid first
        node = self.tree.find_node_by_guid(guid)
        if node is not None:
            self.tree.remove(node)

        # Remove from graph using string GUID
        if self.graph.has_node(str(guid)):
            self.graph.remove_node(str(guid))

        return True

    ###########################################################################################
    # SpatialBVH Collision Detection
    ###########################################################################################

    @staticmethod
    def _compute_bounding_box(geometry, xform: Xform) -> OBB:
        """Bounding box of an object in WORLD placement, inflated by tolerance.

        Parameters
        ----------
        geometry : object
            Any geometry object (Point, Line, Mesh, etc.)
        xform : Xform
            Its cumulative transform from `world_xform` - the geometry itself stores no
            placement, so it must be supplied here.

        Returns
        -------
        OBB
            Inflated bounding box for collision detection.
        """
        inflate = Tolerance.APPROXIMATION
        tp = xform.transform_point

        # Import geometry types
        from .line import Line
        from .polyline import Polyline
        from .pointcloud import PointCloud
        from .mesh import Mesh
        from .plane import Plane
        from .brep import BRep
        from .nurbscurve import NurbsCurve
        from .nurbssurface import NurbsSurface

        if isinstance(geometry, Point):
            return OBB.from_point(tp(geometry), inflate)
        elif isinstance(geometry, Line):
            points = [tp(geometry.start()), tp(geometry.end())]
            return OBB.from_points(points, inflate)
        elif isinstance(geometry, Polyline):
            return OBB.from_points([tp(p) for p in geometry.points], inflate)
        elif isinstance(geometry, PointCloud):
            return OBB.from_points([tp(p) for p in geometry.points], inflate)
        elif isinstance(geometry, Mesh):
            # Extract vertices from mesh; the session holds the placement, so bake it
            points = [tp(v.position()) for v in geometry.vertex.values()]
            if not points:
                return OBB.from_point(Point(0, 0, 0), inflate)
            return OBB.from_points(points, inflate)
        elif isinstance(geometry, OBB):
            # Inflate existing bounding box
            from .vector import Vector

            inflated = OBB(
                center=geometry.center,
                x_axis=geometry.x_axis,
                y_axis=geometry.y_axis,
                z_axis=geometry.z_axis,
                half_size=Vector(
                    geometry.half_size[0] + inflate,
                    geometry.half_size[1] + inflate,
                    geometry.half_size[2] + inflate,
                ),
            )
            inflated.transform(xform)
            return inflated
        elif isinstance(geometry, Plane):
            # Create bounded box around plane origin
            return OBB.from_point(tp(geometry.origin), inflate * 10.0)
        elif isinstance(geometry, NurbsCurve):
            points = []
            for i in range(geometry.cv_count()):
                p = geometry.get_cv(i)
                if p is not None:
                    points.append(tp(p))
            if not points:
                return OBB.from_point(Point(0, 0, 0), inflate)
            return OBB.from_points(points, inflate)
        elif isinstance(geometry, NurbsSurface):
            points = []
            for i in range(geometry.cv_count_dir(0)):
                for j in range(geometry.cv_count_dir(1)):
                    p = geometry.get_cv(i, j)
                    if p is not None:
                        points.append(tp(p))
            if not points:
                return OBB.from_point(Point(0, 0, 0), inflate)
            return OBB.from_points(points, inflate)
        elif isinstance(geometry, BRep):
            points = [tp(p) for p in geometry.m_vertices]
            # Sample surface points to cover curved surfaces (e.g. sphere with only pole vertices)
            for srf in geometry.m_surfaces:
                u0, u1 = srf.domain(0)
                v0, v1 = srf.domain(1)
                for ui in range(3):
                    for vi in range(3):
                        u = u0 + (u1 - u0) * ui / 2.0
                        v = v0 + (v1 - v0) * vi / 2.0
                        p = srf.point_at(u, v)
                        if p is not None:
                            points.append(tp(p))
            if not points:
                return OBB.from_point(Point(0, 0, 0), inflate)
            return OBB.from_points(points, inflate)
        else:
            from .element import Element
            if isinstance(geometry, Element):
                import copy
                box = copy.deepcopy(geometry.aabb)  # never mutate the element's cached box
                box.transform(xform)
                return box
            # Fallback
            return OBB.from_point(Point(0, 0, 0), inflate)

    def get_collisions(self) -> List[Tuple[str, str]]:
        """Get all collision pairs using SpatialBVH and add them as graph edges.

        Automatically:
        - Computes bounding boxes for all objects with tolerance inflation
        - Builds/rebuilds the SpatialBVH with auto-computed world size
        - Detects all collision pairs
        - Adds collision edges to the graph

        Returns
        -------
        list of tuple
            List of (guid1, guid2) tuples representing colliding geometry pairs.
        """
        # Collect all objects with their bounding boxes and GUIDs
        boxes_with_guids = []

        identity = Xform.identity()
        world = self.world_xforms()  # one downward pass, not one tree scan per object
        for guid, geometry in self.lookup.items():
            bbox = self._compute_bounding_box(geometry, world.get(guid, identity))
            boxes_with_guids.append((bbox, guid))

        if not boxes_with_guids:
            return []

        # Build SpatialBVH with GUIDs (auto-computes world size)
        self.bvh.build_with_guids(boxes_with_guids)

        # Extract just the boxes for collision checking
        boxes = [bbox for bbox, _ in boxes_with_guids]

        # Get collision pairs as GUIDs directly
        collision_pairs = self.bvh.check_all_collisions_guids(boxes)

        # Add collision edges to graph
        for guid1, guid2 in collision_pairs:
            self.graph.add_edge(guid1, guid2, "bvh_collision")

        return collision_pairs

    def ray_cast(
        self, origin: Point, direction, tolerance: float = 1e-3
    ) -> List[RayHit]:
        from .line import Line
        from .vector import Vector
        from .polyline import Polyline
        from .plane import Plane
        from .obb import OBB
        from .mesh import Mesh
        from .intersection import line_line, line_plane, ray_box, ray_mesh_bvh

        dir_vec = Vector(direction[0], direction[1], direction[2])
        if dir_vec.magnitude() <= 0.0:
            return []
        dir_unit = dir_vec.normalized()

        FAR = 1e6
        ray_line = Line(
            origin[0],
            origin[1],
            origin[2],
            origin[0] + dir_unit[0] * FAR,
            origin[1] + dir_unit[1] * FAR,
            origin[2] + dir_unit[2] * FAR,
        )

        # Placements come from the session, not the geometry; resolve them all up front
        identity = Xform.identity()
        world = self.world_xforms()

        boxes_with_guids: List[Tuple[OBB, str]] = []
        for guid, geometry in self.lookup.items():
            bbox = self._compute_bounding_box(geometry, world.get(guid, identity))
            boxes_with_guids.append((bbox, guid))
        if not boxes_with_guids:
            return []

        self.bvh.build_with_guids(boxes_with_guids)

        candidates: List[int] = []
        self.bvh.ray_cast(origin, dir_unit, candidates, True)

        hits_all: List[RayHit] = []

        def point_hit(p: Point) -> Tuple[bool, Point, float]:
            vx = p[0] - origin[0]
            vy = p[1] - origin[1]
            vz = p[2] - origin[2]
            cx = vy * dir_unit[2] - vz * dir_unit[1]
            cy = vz * dir_unit[0] - vx * dir_unit[2]
            cz = vx * dir_unit[1] - vy * dir_unit[0]
            dist = (cx * cx + cy * cy + cz * cz) ** 0.5
            if dist > tolerance:
                return False, origin, 0.0
            t = vx * dir_unit[0] + vy * dir_unit[1] + vz * dir_unit[2]
            if t < 0.0:
                return False, origin, 0.0
            hp = Point(
                origin[0] + dir_unit[0] * t,
                origin[1] + dir_unit[1] * t,
                origin[2] + dir_unit[2] * t,
            )
            return True, hp, t

        for idx in candidates:
            if idx < 0 or idx >= len(self.bvh.object_guids):
                continue
            guid = self.bvh.object_guids[idx]
            geom = self.lookup.get(guid)
            if geom is None:
                continue
            placement = world.get(guid, identity)

            hit_point: Optional[Point] = None

            if isinstance(geom, OBB):
                pts = ray_box(ray_line, geom, 0.0, FAR)
                if pts:
                    hit_point = pts[0]
            elif isinstance(geom, Plane):
                hp = line_plane(ray_line, geom, True)
                if hp is not None:
                    hit_point = hp
            elif hasattr(geom, "start") and hasattr(geom, "end"):
                hp = line_line(ray_line, geom, Tolerance.APPROXIMATION)
                if hp is not None:
                    hit_point = hp
            elif isinstance(geom, Polyline):
                best_t = float("inf")
                best_p: Optional[Point] = None
                for i in range(len(geom.points) - 1):
                    seg = Line.from_points(geom.points[i], geom.points[i + 1])
                    hp = line_line(ray_line, seg, Tolerance.APPROXIMATION)
                    if hp is None:
                        continue
                    t = (
                        (hp[0] - origin[0]) * dir_unit[0]
                        + (hp[1] - origin[1]) * dir_unit[1]
                        + (hp[2] - origin[2]) * dir_unit[2]
                    )
                    if t >= 0.0 and t < best_t:
                        best_t = t
                        best_p = hp
                if best_p is not None:
                    hit_point = best_p
            elif isinstance(geom, Mesh):
                # The session holds the placement: cast in the mesh's LOCAL frame, return a WORLD hit
                inv = placement.inverse()
                if inv is not None:
                    local_ray = Line.from_points(
                        inv.transform_point(ray_line.start()),
                        inv.transform_point(ray_line.end()),
                    )
                    pts = ray_mesh_bvh(local_ray, geom, 1e-6, False)
                    if pts:
                        hit_point = placement.transform_point(pts[0])
            elif isinstance(geom, Point):
                ok, hp, t = point_hit(geom)
                if ok:
                    hit_point = hp

            if hit_point is None:
                continue

            d = (
                (hit_point[0] - origin[0]) * dir_unit[0]
                + (hit_point[1] - origin[1]) * dir_unit[1]
                + (hit_point[2] - origin[2]) * dir_unit[2]
            )
            if d >= 0.0:
                hits_all.append(RayHit(guid, hit_point, d))

        if not hits_all:
            return []

        min_d = min(h.distance for h in hits_all)
        eps = max(1e-6, tolerance * 1e-3)
        hits = [h for h in hits_all if abs(h.distance - min_d) <= eps]
        hits.sort(key=lambda h: h.distance)
        return hits

    ###########################################################################################
    # Details - Tree
    ###########################################################################################

    def add_hierarchy(self, parent_guid: str, child_guid: str) -> bool:
        """Add a parent-child relationship in the tree structure.

        Parameters
        ----------
        parent_guid : UUID
            The GUID of the parent geometry object.
        child_guid : UUID
            The GUID of the child geometry object.

        Returns
        -------
        bool
            True if the relationship was added successfully.
        """
        return self.tree.add_child_by_guid(parent_guid, child_guid)

    def get_children(self, guid: str) -> list[str]:
        """Get all children GUIDs of a geometry object in the tree.

        Parameters
        ----------
        guid : str
            The string GUID to search for.

        Returns
        -------
        list[UUID]
            List of children GUIDs.
        """
        return self.tree.get_children_guids(guid)

    ###########################################################################################
    # Details - Graph
    ###########################################################################################

    def add_relationship(
        self, from_guid: str, to_guid: str, relationship_type: str = "default"
    ) -> None:
        """Add a relationship edge in the graph structure.

        Parameters
        ----------
        from_guid : UUID
            The GUID of the source geometry object.
        to_guid : UUID
            The GUID of the target geometry object.
        relationship_type : str, optional
            The type of relationship. Defaults to "default".
        """
        self.graph.add_edge(from_guid, to_guid, relationship_type)

    def get_neighbours(self, guid: str) -> list[str]:
        """Get all GUIDs connected to the given GUID in the graph.

        Parameters
        ----------
        guid : UUID
            The GUID of the geometry object to find connections for.

        Returns
        -------
        list[str]
            List of connected geometry GUIDs as strings.
        """
        return self.graph.get_neighbors(guid)

    ###########################################################################################
    # Details - Transformed Geometry
    ###########################################################################################

    def get_geometry(self) -> Objects:
        """All geometry with its hierarchical placement BAKED into the coordinates.

        Each object is transformed by its cumulative `world_xform` - its own transform with
        every ancestor's multiplied down the tree. The result is a FLATTENED snapshot: every
        guid's world transform is identity by construction, so never pair it back with
        `self.xforms` or the placement would be applied twice.

        Returns
        -------
        Objects
            Collection of transformed geometry objects.
        """
        import copy

        objects = copy.deepcopy(self.objects)
        world = self.world_xforms()

        def placement(src):
            """The guid comes from the ORIGINAL: Point and Element mint a new guid on deepcopy,
            so the copy can no longer be matched against the session."""
            xform = world.get(src.guid)
            return None if xform is None or xform.is_identity() else xform

        def bake(originals, copies):
            """No type is skipped: the old rebuilt lookup left out nurbscurves, nurbssurfaces
            and breps, so a placement in the tree silently did nothing for them."""
            for src, dst in zip(originals, copies):
                xform = placement(src)
                if xform is None:
                    continue
                dst.transform(xform)

        bake(self.objects.points, objects.points)
        bake(self.objects.lines, objects.lines)
        bake(self.objects.planes, objects.planes)
        bake(self.objects.bboxes, objects.bboxes)
        bake(self.objects.polylines, objects.polylines)
        bake(self.objects.pointclouds, objects.pointclouds)
        bake(self.objects.meshes, objects.meshes)
        bake(self.objects.nurbscurves, objects.nurbscurves)
        bake(self.objects.nurbssurfaces, objects.nurbssurfaces)
        bake(self.objects.breps, objects.breps)

        # An Element holds its own geometry, so its placement is baked through session_geometry.
        for src, dst in zip(self.objects.elements, objects.elements):
            xform = placement(src)
            if xform is not None:
                dst.place(xform)

        return objects
