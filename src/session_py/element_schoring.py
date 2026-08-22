from collections.abc import Callable
from typing import Optional
from typing import Union
from typing import TYPE_CHECKING
import os
import copy
import json
import uuid
import pathlib

from .element import Element
from .mesh import Mesh
from .plane import Plane
from .point import Point
from .vector import Vector
from .xform import Xform

if TYPE_CHECKING:
    from .brep import BRep
    from pathlib import Path
    from .line import Line


DEFAULT_DATA_DIR = r"C:\brg\code_rust\session\session_data\elements"


class Dataset:
    schoring_foot_0 = "schoring_foot_0.json"
    schoring_head_0 = "schoring_head_0.json"
    schoring_body_start_0 = "schoring_body_start_0.json"
    schoring_body_end_0 = "schoring_body_end_0.json"
    schoring_foot_1 = "schoring_foot_1.json"
    schoring_body_start_1 = "schoring_body_start_1.json"
    schoring_body_end_1 = "schoring_body_end_1.json"
    schoring_head_1 = "schoring_head_1.json"
    schoring_body_start_2 = "schoring_body_start_2.json"
    schoring_body_start_3 = "schoring_body_start_3.json"

    @classmethod
    def select_by_type(cls, type: str = "a", target_length: int = 0, data_dir: Union[str, "pathlib.Path"] | None = None) -> list[str]:
        if type != "a":
            return []
        data_dir = pathlib.Path(data_dir or DEFAULT_DATA_DIR)
        group = [cls.schoring_foot_0, cls.schoring_body_start_0, cls.schoring_body_end_0, cls.schoring_head_0]
        choices = [cls.schoring_body_start_0, cls.schoring_body_start_2, cls.schoring_body_start_3]
        end_length = _read_extension_length(data_dir / group[2])
        totals = [_read_extension_length(data_dir / c) + end_length for c in choices]
        paired = sorted(enumerate(totals), key=lambda x: x[1])
        chosen = paired[0][0]
        for idx, tot in paired:
            if target_length < tot:
                chosen = idx
                break
        group[1] = choices[chosen]
        return group


def _read_extension_length(json_path):
    with open(json_path) as f:
        return json.load(f).get("extension_length", 0.0)


def _read_dataset(json_path):
    """Load a session-native ElementSchoring dataset JSON.

    Returns (planes, mesh, extension_length).
    """
    with open(json_path) as f:
        raw = json.load(f)
    planes = [Plane.__jsonload__(p) for p in raw.get("frames", [])]
    mesh = Mesh.__jsonload__(raw["mesh"]) if raw.get("mesh") else None
    return planes, mesh, raw.get("extension_length", 0.0)


def _xform_from_plane(p):
    return Xform.to_frame(p)


class ElementSchoring(Element):
    """Scaffolding prop element (foot / body_start / body_end / head) loaded from a dataset.

    Reads a session-native JSON that bundles the tessellated Mesh, connection
    Planes ("frames") and the extension_length of that piece.
    """

    # Class-level cache
    _dataset_cache = {}

    def __init__(
        self,
        dataset: list[str] | None = None,
        features: list[Callable] | None = None,
        name: str | None = None,
        data_dir: Union[str, "pathlib.Path"] | None = None,
        geometry_as_brep: bool = False,
    ):
        super().__init__(geometry=None, name=name or "my_schoring")
        if features:
            self._features = list(features)
        self.dataset = dataset
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.geometry_as_brep = geometry_as_brep

        if dataset is None:
            self._geometry = None
            self._fallback_mesh = None
            self.frames = []
            self.max_extension_length = 0.0
            return

        json_path = os.path.join(self.data_dir, dataset)
        planes, mesh, ext = self._load_dataset(json_path)
        self.frames = planes
        self.max_extension_length = ext
        self._fallback_mesh = mesh
        if geometry_as_brep:
            self._geometry = self._load_step_brep(dataset, self.data_dir, mesh.name)
        else:
            self._geometry = mesh

    # Per-path BRep cache (STEP parsing is not free)
    _brep_cache = {}

    @classmethod
    def _load_step_brep(cls, dataset, data_dir, name=None):
        raise NotImplementedError("STEP reading is C++ only; use the C++ session_cpp API")

    ###########################################################################################
    # Caching / loading
    ###########################################################################################

    @classmethod
    def _load_dataset(cls, json_path):
        if json_path not in cls._dataset_cache:
            cls._dataset_cache[json_path] = _read_dataset(json_path)
        planes, mesh, ext = cls._dataset_cache[json_path]
        return (
            [copy.deepcopy(p) for p in planes],
            copy.deepcopy(mesh),
            ext,
        )

    @classmethod
    def clear_cache(cls) -> None:
        cls._dataset_cache.clear()
        cls._brep_cache.clear()

    ###########################################################################################
    # Element interface
    ###########################################################################################

    def compute_element_geometry(self) -> Union["Mesh", "BRep"] | None:
        return self._geometry

    ###########################################################################################
    # Static helpers
    ###########################################################################################

    @staticmethod
    def polylines_to_elements(lines_unordered: list["Line"], data_dir: Union[str, "pathlib.Path"] | None = None) -> list[tuple["ElementSchoring", "Xform"]]:
        """Build a chain of schoring elements (foot, body_start, body_end, head) for each vertical line.

        Mirrors the compas_tf SchoringElement.polylines_to_models logic. An Element no longer
        stores a placement, so this returns a flat list of ``(ElementSchoring, Xform)`` pairs:
        hand each pair to ``session.add_element(elem, parent)`` + ``session.set_xform(elem.guid, xf)``.
        """
        from .line import Line

        lines = []
        for l in lines_unordered:
            if l[0][2] > l[1][2]:
                lines.append(Line(l[1][0], l[1][1], l[1][2], l[0][0], l[0][1], l[0][2]))
            else:
                lines.append(l)

        xaxis = Vector(1, 0, 0)
        yaxis = Vector(0, 1, 0)
        if len(lines) > 2:
            return []
        if len(lines) == 2:
            start_a, start_b = lines[0].start_point(), lines[1].start_point()
            xaxis = Vector(start_b.x - start_a.x, start_b.y - start_a.y, start_b.z - start_a.z)
            yaxis = Vector(0, 0, 1).cross(xaxis)
        else:
            d = lines[0].direction()
            if d.x != 0 or d.y != 0:
                xaxis = Vector(d.x, d.y, 0.0)
                yaxis = Vector(0, 0, 1).cross(xaxis)

        elements = []
        groups = []
        placements = {}
        for i, line in enumerate(lines):
            length = line.length()
            filenames = Dataset.select_by_type("a", length, data_dir=data_dir)
            foot = ElementSchoring(filenames[0], data_dir=data_dir)
            body_start = ElementSchoring(filenames[1], data_dir=data_dir)
            body_end = ElementSchoring(filenames[2], data_dir=data_dir)
            head = ElementSchoring(filenames[3], data_dir=data_dir)
            group = [foot, body_start, body_end, head] if i == 0 else [foot, body_start, body_end]
            groups.append(group)
            elements.extend(group)

        for idx, group in enumerate(groups):
            foot = group[0]
            body_start = group[1]
            body_end = group[2]
            head = groups[0][3]

            start_pt = lines[idx].start_point()
            end_pt = lines[idx].end_point()

            frame_source = Plane(origin=Point(start_pt.x, start_pt.y, start_pt.z), x_axis=xaxis, y_axis=yaxis)
            frame_target = Plane(
                origin=Point(end_pt.x, end_pt.y, end_pt.z),
                x_axis=frame_source.x_axis,
                y_axis=-frame_source.y_axis,
            )
            xform_frame_source = _xform_from_plane(frame_source)
            xform_frame_target = _xform_from_plane(frame_target)

            xform_base = xform_frame_source * _xform_from_plane(foot.frames[idx])
            xform_end = xform_frame_target * _xform_from_plane(head.frames[idx])

            base_point = Point(xform_base.m[3], xform_base.m[7], xform_base.m[11])
            end_point = Point(xform_end.m[3], xform_end.m[7], xform_end.m[11])
            direction = Vector(end_point.x - base_point.x, end_point.y - base_point.y, end_point.z - base_point.z)

            base_plane_world = Xform.identity()
            base_plane_world = xform_base  # reuse
            body_start_yaxis = Vector(base_plane_world.m[1], base_plane_world.m[5], base_plane_world.m[9])
            body_start_zaxis_dir = direction.cross(body_start_yaxis)

            body_start_plane = Plane(origin=base_point, x_axis=body_start_yaxis, y_axis=body_start_zaxis_dir)
            body_end_origin = Point(
                end_point.x - body_start_plane.z_axis.x * body_end.frames[0].origin.z,
                end_point.y - body_start_plane.z_axis.y * body_end.frames[0].origin.z,
                end_point.z - body_start_plane.z_axis.z * body_end.frames[0].origin.z,
            )
            body_end_plane = Plane(origin=body_end_origin, x_axis=body_start_yaxis, y_axis=body_start_zaxis_dir)

            placements[id(foot)] = xform_frame_source
            placements[id(body_start)] = _xform_from_plane(body_start_plane)
            placements[id(body_end)] = _xform_from_plane(body_end_plane)
            placements[id(head)] = xform_frame_target

        return [(e, placements.get(id(e), Xform.identity())) for e in elements]

    ###########################################################################################
    # Operators
    ###########################################################################################

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.guid = str(uuid.uuid4())
        result.name = copy.deepcopy(self.name, memo)
        result.dataset = self.dataset
        result.data_dir = self.data_dir
        result.frames = [copy.deepcopy(p, memo) for p in getattr(self, "frames", [])]
        result.max_extension_length = getattr(self, "max_extension_length", 0.0)
        result._geometry = copy.deepcopy(self._geometry, memo)
        result._features = list(self._features)
        result._is_dirty = True
        result._aabb = None
        result._obb = None
        result._collision_mesh = None
        result._point = None
        result._polylines = None
        result._planes = None
        result._edge_vectors = None
        result._axis = None
        return result

    def __eq__(self, other):
        if not isinstance(other, ElementSchoring):
            return False
        return self.name == other.name and self.dataset == other.dataset

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"ElementSchoring({self.name}, {self.dataset})"

    def __repr__(self):
        return f"ElementSchoring({self.guid}, {self.name}, {self.dataset})"

    ###########################################################################################
    # Serialization - JSON
    ###########################################################################################

    def __jsondump__(self):
        return {
            "dataset": self.dataset,
            "data_dir": self.data_dir,
            "geometry_data": self._geometry.__jsondump__() if self._geometry else None,
            "geometry_type": type(self._geometry).__name__ if self._geometry else "None",
            "guid": self.guid,
            "name": self.name,
            "type": "ElementSchoring",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .file_encoders import file_decode_node
        elem = cls(
            dataset=data.get("dataset"),
            data_dir=data.get("data_dir"),
            name=data.get("name"),
        )
        elem.guid = guid if guid is not None else data.get("guid", elem.guid)
        if name is not None:
            elem.name = name
        return elem

    ###########################################################################################
    # Serialization - Protobuf
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.guid = self.guid
        proto.name = self.name
        proto.geometry_type = "ElementSchoring"
        proto.geometry_data = json.dumps({
            "dataset": self.dataset,
            "data_dir": self.data_dir,
        }).encode()
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "ElementSchoring":
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.ParseFromString(data)
        params = json.loads(proto.geometry_data.decode())
        elem = cls(
            dataset=params.get("dataset"),
            data_dir=params.get("data_dir"),
        )
        elem.guid = proto.guid
        elem.name = proto.name
        return elem
