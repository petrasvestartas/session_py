from .point import Point
from .line import Line
from .plane import Plane
from .boundingbox import BoundingBox
from .polyline import Polyline
from .pointcloud import PointCloud
from .mesh import Mesh
from .cylinder import Cylinder
from .arrow import Arrow
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
import uuid


class Objects:
    """A collection of all geometry objects.

    Attributes
    ----------
    name : str
        The name of the collection.
    guid : UUID
        The unique identifier of the collection.
    points : list[Point]
        The list of points.
    lines : list[Line]
        The list of lines.
    planes : list[Plane]
        The list of planes.
    bboxes : list[BoundingBox]
        The list of bounding boxes.
    polylines : list[Polyline]
        The list of polylines.
    pointclouds : list[PointCloud]
        The list of point clouds.
    meshes : list[Mesh]
        The list of meshes.
    cylinders : list[Cylinder]
        The list of cylinders.
    arrows : list[Arrow]
        The list of arrows.

    """

    def __init__(self):
        self.guid = str(uuid.uuid4())
        self.name = "my_objects"
        self.points: list[Point] = []
        self.lines: list[Line] = []
        self.planes: list[Plane] = []
        self.bboxes: list[BoundingBox] = []
        self.polylines: list[Polyline] = []
        self.pointclouds: list[PointCloud] = []
        self.meshes: list[Mesh] = []
        self.cylinders: list[Cylinder] = []
        self.arrows: list[Arrow] = []
        self.nurbscurves: list[NurbsCurve] = []
        self.nurbssurfaces: list[NurbsSurface] = []

    def __str__(self):
        return f"Objects(points={len(self.points)})"

    def __repr__(self):
        return f"Objects({self.guid}, {self.name}, points={len(self.points)})"

    ###########################################################################################
    # Polymorphic JSON Serialization
    ###########################################################################################

    def __jsondump__(self):
        """Serialize to polymorphic JSON format with type field.

        Returns
        -------
        dict
            Dictionary with 'type', 'guid', 'name', and object fields.

        """
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "points": [p.__jsondump__() for p in self.points],
            "lines": [l.__jsondump__() for l in self.lines],
            "planes": [pl.__jsondump__() for pl in self.planes],
            "bboxes": [b.__jsondump__() for b in self.bboxes],
            "polylines": [pl.__jsondump__() for pl in self.polylines],
            "pointclouds": [pc.__jsondump__() for pc in self.pointclouds],
            "meshes": [m.__jsondump__() for m in self.meshes],
            "cylinders": [c.__jsondump__() for c in self.cylinders],
            "arrows": [a.__jsondump__() for a in self.arrows],
            "nurbscurves": [nc.__jsondump__() for nc in self.nurbscurves],
            "nurbssurfaces": [ns.__jsondump__() for ns in self.nurbssurfaces],
        }

    @classmethod
    def __jsonload__(cls, data, guid=None, name=None):
        """Deserialize from polymorphic JSON format.

        Parameters
        ----------
        data : dict
            Dictionary containing objects data.
        guid : str, optional
            GUID for the objects.
        name : str, optional
            Name for the objects.

        Returns
        -------
        :class:`Objects`
            Reconstructed objects instance.

        """
        from .encoders import decode_node

        obj = cls()
        obj.guid = guid if guid is not None else data.get("guid", obj.guid)
        obj.name = name if name is not None else data.get("name", obj.name)

        obj.points = [decode_node(p) for p in data.get("points", [])]
        obj.lines = [decode_node(l) for l in data.get("lines", [])]
        obj.planes = [decode_node(pl) for pl in data.get("planes", [])]
        obj.bboxes = [decode_node(b) for b in data.get("bboxes", [])]
        obj.polylines = [decode_node(pl) for pl in data.get("polylines", [])]
        obj.pointclouds = [decode_node(pc) for pc in data.get("pointclouds", [])]
        obj.meshes = [decode_node(m) for m in data.get("meshes", [])]
        obj.cylinders = [decode_node(c) for c in data.get("cylinders", [])]
        obj.arrows = [decode_node(a) for a in data.get("arrows", [])]
        obj.nurbscurves = [decode_node(nc) for nc in data.get("nurbscurves", [])]
        obj.nurbssurfaces = [decode_node(ns) for ns in data.get("nurbssurfaces", [])]

        return obj

    def json_dumps(self):
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, s):
        import json
        return cls.__jsonload__(json.loads(s))

    def json_dump(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        import json
        with open(filepath, 'r') as f:
            return cls.__jsonload__(json.load(f))

    def pb_dumps(self):
        from .proto import objects_pb2
        proto = objects_pb2.Objects()
        proto.name = self.name
        proto.guid = self.guid
        for p in self.points:
            proto.points.add().ParseFromString(p.pb_dumps())
        for l in self.lines:
            proto.lines.add().ParseFromString(l.pb_dumps())
        for pl in self.planes:
            proto.planes.add().ParseFromString(pl.pb_dumps())
        for b in self.bboxes:
            proto.bboxes.add().ParseFromString(b.pb_dumps())
        for pl in self.polylines:
            proto.polylines.add().ParseFromString(pl.pb_dumps())
        for pc in self.pointclouds:
            proto.pointclouds.add().ParseFromString(pc.pb_dumps())
        for m in self.meshes:
            proto.meshes.add().ParseFromString(m.pb_dumps())
        for c in self.cylinders:
            proto.cylinders.add().ParseFromString(c.pb_dumps())
        for a in self.arrows:
            proto.arrows.add().ParseFromString(a.pb_dumps())
        for nc in self.nurbscurves:
            proto.nurbscurves.add().ParseFromString(nc.pb_dumps())
        for ns in self.nurbssurfaces:
            proto.nurbssurfaces.add().ParseFromString(ns.pb_dumps())
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import objects_pb2
        proto = objects_pb2.Objects()
        proto.ParseFromString(data)
        objects = cls()
        objects.guid = proto.guid
        objects.name = proto.name
        for p in proto.points:
            objects.points.append(Point.pb_loads(p.SerializeToString()))
        for l in proto.lines:
            objects.lines.append(Line.pb_loads(l.SerializeToString()))
        for pl in proto.planes:
            objects.planes.append(Plane.pb_loads(pl.SerializeToString()))
        for b in proto.bboxes:
            objects.bboxes.append(BoundingBox.pb_loads(b.SerializeToString()))
        for pl in proto.polylines:
            objects.polylines.append(Polyline.pb_loads(pl.SerializeToString()))
        for pc in proto.pointclouds:
            objects.pointclouds.append(PointCloud.pb_loads(pc.SerializeToString()))
        for m in proto.meshes:
            objects.meshes.append(Mesh.pb_loads(m.SerializeToString()))
        for c in proto.cylinders:
            objects.cylinders.append(Cylinder.pb_loads(c.SerializeToString()))
        for a in proto.arrows:
            objects.arrows.append(Arrow.pb_loads(a.SerializeToString()))
        for nc in proto.nurbscurves:
            objects.nurbscurves.append(NurbsCurve.pb_loads(nc.SerializeToString()))
        for ns in proto.nurbssurfaces:
            objects.nurbssurfaces.append(NurbsSurface.pb_loads(ns.SerializeToString()))
        return objects

    def pb_dump(self, filepath):
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())

    ###########################################################################################
    # Details
    ###########################################################################################
