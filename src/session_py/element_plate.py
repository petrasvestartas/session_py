from __future__ import annotations
from typing import Optional
import uuid
import copy
from .element import Element
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .line import Line
    from .mesh import Mesh
    from .plane import Plane
    from .polyline import Polyline
    from .vector import Vector


class ElementPlate(Element):
    @staticmethod
    def _strip_closing(pts):
        from .point import Point
        result = [Point(p[0], p[1], p[2]) for p in pts]
        if len(result) > 3:
            f, l = result[0], result[-1]
            if abs(f[0]-l[0]) < 1e-6 and abs(f[1]-l[1]) < 1e-6 and abs(f[2]-l[2]) < 1e-6:
                result.pop()
        return result

    def __init__(self, polygon: Optional["Polyline"] = None, thickness: float = 0.1, name: str = "my_plate", polygon_top: Optional["Polyline"] = None):
        super().__init__(geometry=None, name=name)
        from .point import Point
        if polygon is None:
            polygon = [
                Point(-0.5, -0.5, 0),
                Point( 0.5, -0.5, 0),
                Point( 0.5,  0.5, 0),
                Point(-0.5,  0.5, 0),
            ]
        self._polygon = self._strip_closing(polygon)
        if polygon_top is not None:
            self._polygon_top = self._strip_closing(polygon_top)
            # Ensure bottom normal points toward top
            normal = self._polygon_normal(self._polygon)
            d = sum((self._polygon_top[k][i]-self._polygon[k][i])*normal[i]
                    for k in range(min(len(self._polygon), len(self._polygon_top))) for i in range(3))
            if d < 0:
                self._polygon, self._polygon_top = self._polygon_top, self._polygon
            n = min(len(self._polygon), len(self._polygon_top))
            self._thickness = sum(
                ((self._polygon_top[k][0]-self._polygon[k][0])**2 +
                 (self._polygon_top[k][1]-self._polygon[k][1])**2 +
                 (self._polygon_top[k][2]-self._polygon[k][2])**2)**0.5
                for k in range(n)) / n
        else:
            self._thickness = thickness
            normal = self._polygon_normal(self._polygon)
            self._polygon_top = [Point(
                p[0]-normal[0]*thickness, p[1]-normal[1]*thickness, p[2]-normal[2]*thickness
            ) for p in self._polygon]
        self._joint_types = []
        self._j_mf = []
        self._key = ""
        self._component_plane = None
        self._geometry = self.compute_element_geometry()

    @property
    def polygon(self) -> "Polyline":
        return self._polygon

    @polygon.setter
    def polygon(self, value: "Polyline") -> None:
        from .point import Point
        self._polygon = [Point(p[0], p[1], p[2]) for p in value]
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def thickness(self) -> float:
        return self._thickness

    @thickness.setter
    def thickness(self, value: float) -> None:
        self._thickness = value
        self._geometry = self.compute_element_geometry()
        self.reset()

    @property
    def joint_types(self) -> list:
        return self._joint_types

    @joint_types.setter
    def joint_types(self, value: list) -> None:
        self._joint_types = list(value)

    @property
    def j_mf(self) -> list:
        return self._j_mf

    @j_mf.setter
    def j_mf(self, value: list) -> None:
        self._j_mf = [list(face) for face in value]

    @property
    def key(self) -> str:
        return self._key

    @key.setter
    def key(self, value: str) -> None:
        self._key = value

    @property
    def component_plane(self) -> Optional["Plane"]:
        return self._component_plane

    @component_plane.setter
    def component_plane(self, value: Optional["Plane"]) -> None:
        self._component_plane = value

    @staticmethod
    def _polygon_normal(pts):
        from .vector import Vector
        nx, ny, nz = 0.0, 0.0, 0.0
        n = len(pts)
        for i in range(n):
            c = pts[i]
            nx_pt = pts[(i + 1) % n]
            nx += (c[1] - nx_pt[1]) * (c[2] + nx_pt[2])
            ny += (c[2] - nx_pt[2]) * (c[0] + nx_pt[0])
            nz += (c[0] - nx_pt[0]) * (c[1] + nx_pt[1])
        mag = (nx * nx + ny * ny + nz * nz) ** 0.5
        if mag < 1e-12:
            return Vector(0, 0, 1)
        return Vector(nx / mag, ny / mag, nz / mag)

    @property
    def polygon_top(self) -> "Polyline":
        return self._polygon_top

    def compute_element_geometry(self) -> "Mesh":
        from .mesh import Mesh
        from .point import Point
        n = min(len(self._polygon), len(self._polygon_top))
        bottom = [Point(p[0], p[1], p[2]) for p in self._polygon[:n]]
        top = [Point(p[0], p[1], p[2]) for p in self._polygon_top[:n]]
        vertices = bottom + top
        bottom_face = list(range(n - 1, -1, -1))
        top_face = list(range(n, 2 * n))
        faces = [bottom_face, top_face]
        for i in range(n):
            a = i
            b = (i + 1) % n
            c = b + n
            d = a + n
            faces.append([a, b, c, d])
        return Mesh.from_vertices_and_faces(vertices, faces)

    def compute_polylines(self) -> list["Polyline"]:
        from .polyline import Polyline
        from .point import Point
        n = min(len(self._polygon), len(self._polygon_top))
        bottom = [Point(p[0], p[1], p[2]) for p in self._polygon[:n]]
        top = [Point(p[0], p[1], p[2]) for p in self._polygon_top[:n]]
        # [0]=top, [1]=bottom, [2+]=sides (matching wood)
        top_pl = Polyline(top + [Point(top[0][0], top[0][1], top[0][2])])
        bot_pl = Polyline(bottom + [Point(bottom[0][0], bottom[0][1], bottom[0][2])])
        result = [top_pl, bot_pl]
        for i in range(n):
            j = (i + 1) % n
            side = Polyline([top[i], top[j], bottom[j], bottom[i], Point(top[i][0], top[i][1], top[i][2])])
            result.append(side)
        return result

    def compute_planes(self) -> list["Plane"]:
        from .plane import Plane
        from .point import Point
        from .vector import Vector
        normal = self._polygon_normal(self._polygon)
        n = min(len(self._polygon), len(self._polygon_top))
        bottom = [Point(p[0], p[1], p[2]) for p in self._polygon[:n]]
        top = [Point(p[0], p[1], p[2]) for p in self._polygon_top[:n]]
        tcx = sum(p[0] for p in top) / n
        tcy = sum(p[1] for p in top) / n
        tcz = sum(p[2] for p in top) / n
        bcx = sum(p[0] for p in bottom) / n
        bcy = sum(p[1] for p in bottom) / n
        bcz = sum(p[2] for p in bottom) / n
        # [0]=top plane, [1]=bottom plane (matching wood)
        top_plane = Plane.from_point_normal(Point(tcx, tcy, tcz), normal)
        neg_normal = Vector(-normal[0], -normal[1], -normal[2])
        bottom_plane = Plane.from_point_normal(Point(bcx, bcy, bcz), neg_normal)
        result = [top_plane, bottom_plane]
        for i in range(n):
            j = (i + 1) % n
            ax = top[i][0]-top[j][0]; ay = top[i][1]-top[j][1]; az = top[i][2]-top[j][2]
            bx = bottom[j][0]-top[j][0]; by = bottom[j][1]-top[j][1]; bz = bottom[j][2]-top[j][2]
            snx = ay*bz-az*by; sny = az*bx-ax*bz; snz = ax*by-ay*bx
            mag = (snx*snx+sny*sny+snz*snz)**0.5
            if mag > 1e-12: snx/=mag; sny/=mag; snz/=mag
            cx = (top[i][0]+top[j][0]+bottom[j][0]+bottom[i][0])*0.25
            cy = (top[i][1]+top[j][1]+bottom[j][1]+bottom[i][1])*0.25
            cz = (top[i][2]+top[j][2]+bottom[j][2]+bottom[i][2])*0.25
            result.append(Plane.from_point_normal(Point(cx, cy, cz), Vector(snx, sny, snz)))
        return result

    def compute_edge_vectors(self) -> list["Vector"]:
        from .vector import Vector
        n = len(self._polygon)
        result = []
        for i in range(n):
            j = (i + 1) % n
            dx = self._polygon[j][0] - self._polygon[i][0]
            dy = self._polygon[j][1] - self._polygon[i][1]
            dz = self._polygon[j][2] - self._polygon[i][2]
            mag = (dx * dx + dy * dy + dz * dz) ** 0.5
            if mag > 1e-12:
                result.append(Vector(dx / mag, dy / mag, dz / mag))
            else:
                result.append(Vector(0, 0, 0))
        return result

    def compute_axis(self) -> "Line":
        from .line import Line
        from .point import Point
        normal = self._polygon_normal(self._polygon)
        n = len(self._polygon)
        cx = sum(p[0] for p in self._polygon) / n
        cy = sum(p[1] for p in self._polygon) / n
        cz = sum(p[2] for p in self._polygon) / n
        return Line(
            cx, cy, cz,
            cx - normal[0] * self._thickness,
            cy - normal[1] * self._thickness,
            cz - normal[2] * self._thickness,
        )

    ###########################################################################################
    # Operators
    ###########################################################################################

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.guid = str(uuid.uuid4())
        result.name = copy.deepcopy(self.name, memo)
        result._polygon = copy.deepcopy(self._polygon, memo)
        result._thickness = self._thickness
        result._joint_types = list(self._joint_types)
        result._j_mf = copy.deepcopy(self._j_mf, memo)
        result._key = self._key
        result._component_plane = copy.deepcopy(self._component_plane, memo)
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
        if not isinstance(other, ElementPlate):
            return False
        if self.name != other.name:
            return False
        if self._thickness != other._thickness:
            return False
        if len(self._polygon) != len(other._polygon):
            return False
        for a, b in zip(self._polygon, other._polygon):
            if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"ElementPlate({self.name}, {len(self._polygon)} pts, {self._thickness})"

    def __repr__(self):
        return f"ElementPlate({self.guid}, {self.name}, {len(self._polygon)} pts, {self._thickness})"

    ###########################################################################################
    # Serialization - JSON
    ###########################################################################################

    def __jsondump__(self):
        return {
            "component_plane": self._component_plane.__jsondump__() if self._component_plane else None,
            "geometry_data": self._geometry.__jsondump__() if self._geometry else None,
            "geometry_type": type(self._geometry).__name__ if self._geometry else "None",
            "guid": self.guid,
            "j_mf": self._j_mf,
            "joint_types": self._joint_types,
            "key": self._key,
            "name": self.name,
            "polygon": [[p[0], p[1], p[2]] for p in self._polygon],
            "polygon_top": [[p[0], p[1], p[2]] for p in self._polygon_top],
            "thickness": self._thickness,
            "type": "ElementPlate",
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        from .file_encoders import file_decode_node
        from .point import Point
        polygon = [Point(p[0], p[1], p[2]) for p in data.get("polygon", [])]
        polygon_top_raw = data.get("polygon_top", [])
        polygon_top = [Point(p[0], p[1], p[2]) for p in polygon_top_raw] if polygon_top_raw else None
        elem = cls(
            polygon=polygon if polygon else None,
            thickness=data.get("thickness", 0.1),
            polygon_top=polygon_top,
        )
        elem.guid = guid if guid is not None else data.get("guid", elem.guid)
        elem.name = name if name is not None else data.get("name", elem.name)
        elem._joint_types = data.get("joint_types", [])
        elem._j_mf = data.get("j_mf", [])
        elem._key = data.get("key", "")
        cp = data.get("component_plane")
        if cp is not None:
            elem._component_plane = file_decode_node(cp)
        return elem

    ###########################################################################################
    # Serialization - Protobuf
    ###########################################################################################

    def pb_dumps(self) -> bytes:
        from .proto import element_pb2
        proto = element_pb2.Element()
        proto.guid = self.guid
        proto.name = self.name
        proto.geometry_type = "ElementPlate"
        import json
        proto.geometry_data = json.dumps({
            "polygon": [[p[0], p[1], p[2]] for p in self._polygon],
            "polygon_top": [[p[0], p[1], p[2]] for p in self._polygon_top],
            "thickness": self._thickness,
        }).encode()
        proto.joint_types.extend(self._joint_types)
        for face_joints in self._j_mf:
            fj = element_pb2.FaceJoints()
            for jid, is_male, param in face_joints:
                jc = element_pb2.JointConnection()
                jc.joint_id = jid
                jc.is_male = is_male
                jc.parameter = param
                fj.connections.append(jc)
            proto.j_mf.append(fj)
        proto.key = self._key
        if self._component_plane is not None:
            cp = self._component_plane
            proto.component_plane.name = cp.name
            proto.component_plane.frame.extend([
                cp.origin[0], cp.origin[1], cp.origin[2],
                cp.x_axis[0], cp.x_axis[1], cp.x_axis[2],
                cp.y_axis[0], cp.y_axis[1], cp.y_axis[2],
                cp.z_axis[0], cp.z_axis[1], cp.z_axis[2],
            ])
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "ElementPlate":
        from .proto import element_pb2
        from .point import Point
        import json
        proto = element_pb2.Element()
        proto.ParseFromString(data)
        params = json.loads(proto.geometry_data.decode())
        polygon = [Point(p[0], p[1], p[2]) for p in params["polygon"]]
        polygon_top_raw = params.get("polygon_top", [])
        polygon_top = [Point(p[0], p[1], p[2]) for p in polygon_top_raw] if polygon_top_raw else None
        elem = cls(
            polygon=polygon,
            thickness=params["thickness"],
            polygon_top=polygon_top,
        )
        elem.guid = proto.guid
        elem.name = proto.name
        elem._joint_types = list(proto.joint_types)
        elem._j_mf = []
        for fj in proto.j_mf:
            face = [(jc.joint_id, jc.is_male, jc.parameter) for jc in fj.connections]
            elem._j_mf.append(face)
        elem._key = proto.key
        if proto.HasField("component_plane") and len(proto.component_plane.frame) == 12:
            from .plane import Plane
            from .vector import Vector
            f = list(proto.component_plane.frame)
            elem._component_plane = Plane(
                origin=Point(f[0], f[1], f[2]),
                x_axis=Vector(f[3], f[4], f[5]),
                y_axis=Vector(f[6], f[7], f[8]),
                name=proto.component_plane.name,
            )
        return elem
