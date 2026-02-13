import uuid
import copy
import json

from .nurbssurface import NurbsSurface
from .nurbscurve import NurbsCurve
from .primitives import Primitives
from .xform import Xform
from .color import Color


class TrimmedSurface:

    def __init__(self):
        self.guid = str(uuid.uuid4())
        self.name = "my_trimmedsurface"
        self.width = 1.0
        self.surfacecolor = Color.black()
        self.xform = Xform.identity()
        self.m_surface = NurbsSurface()
        self.m_outer_loop = NurbsCurve()
        self.m_inner_loops = []

    @staticmethod
    def create(surface, outer_loop):
        ts = TrimmedSurface()
        ts.m_surface = surface.duplicate()
        ts.m_outer_loop = outer_loop.duplicate()
        return ts

    @staticmethod
    def create_planar(boundary):
        srf = Primitives.create_planar(boundary)
        ts = TrimmedSurface()
        ts.m_surface = srf.duplicate()
        if srf.is_trimmed():
            ts.m_outer_loop = srf.m_outer_loop.duplicate()
            srf.m_outer_loop = NurbsCurve()
        ts.m_inner_loops = [l.duplicate() for l in srf.m_inner_loops]
        return ts

    def surface(self):
        return self.m_surface

    def get_outer_loop(self):
        return self.m_outer_loop

    def set_outer_loop(self, loop):
        self.m_outer_loop = loop

    def is_trimmed(self):
        return self.m_outer_loop.is_valid()

    def is_valid(self):
        return self.m_surface.is_valid()

    def add_inner_loop(self, loop_2d):
        self.m_inner_loops.append(loop_2d)

    def add_hole(self, curve_3d):
        temp = self.m_surface.duplicate()
        temp.add_hole(curve_3d)
        if temp.inner_loop_count() > 0:
            self.m_inner_loops.append(temp.get_inner_loop(temp.inner_loop_count() - 1))

    def add_holes(self, curves_3d):
        for crv in curves_3d:
            self.add_hole(crv)

    def get_inner_loop(self, index):
        if 0 <= index < len(self.m_inner_loops):
            return self.m_inner_loops[index]
        return None

    def inner_loop_count(self):
        return len(self.m_inner_loops)

    def clear_inner_loops(self):
        self.m_inner_loops.clear()

    def point_at(self, u, v):
        return self.m_surface.point_at(u, v)

    def normal_at(self, u, v):
        return self.m_surface.normal_at(u, v)

    def mesh(self):
        temp = self.m_surface.duplicate()
        if self.is_trimmed():
            temp.set_outer_loop(self.m_outer_loop)
        temp.clear_inner_loops()
        for loop in self.m_inner_loops:
            temp.add_inner_loop(loop)
        return temp.mesh()

    def transform(self, xf=None):
        if xf is None:
            self.m_surface.transform(self.xform)
            self.xform = Xform.identity()
        else:
            self.m_surface.transform(xf)

    def transformed(self):
        ts = self.duplicate()
        ts.transform()
        return ts

    def duplicate(self):
        result = copy.deepcopy(self)
        result.guid = str(uuid.uuid4())
        return result

    def __eq__(self, other):
        if not isinstance(other, TrimmedSurface):
            return False
        if self.name != other.name:
            return False
        if self.width != other.width:
            return False
        if self.surfacecolor != other.surfacecolor:
            return False
        if self.xform != other.xform:
            return False
        if self.m_surface != other.m_surface:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_string(self):
        return f"TrimmedSurface(name={self.name}, trimmed={'true' if self.is_trimmed() else 'false'}, holes={self.inner_loop_count()})"

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return (f"TrimmedSurface(\n  name={self.name},\n"
                f"  trimmed={'true' if self.is_trimmed() else 'false'},\n"
                f"  holes={self.inner_loop_count()},\n"
                f"  surface={str(self.m_surface)}\n)")

    def __jsondump__(self):
        d = {
            'guid': self.guid,
            'inner_loops': [l.__jsondump__() for l in self.m_inner_loops],
            'name': self.name,
        }
        if self.m_outer_loop.is_valid():
            d['outer_loop'] = self.m_outer_loop.__jsondump__()
        d['surface'] = self.m_surface.__jsondump__()
        d['surfacecolor'] = self.surfacecolor.__jsondump__()
        d['type'] = 'TrimmedSurface'
        d['width'] = self.width
        d['xform'] = self.xform.__jsondump__()
        return d

    @classmethod
    def __jsonload__(cls, data):
        ts = cls()
        ts.guid = data.get('guid', ts.guid)
        ts.name = data.get('name', 'my_trimmedsurface')
        ts.width = data.get('width', 1.0)
        if 'surfacecolor' in data:
            ts.surfacecolor = Color.__jsonload__(data['surfacecolor'])
        if 'xform' in data:
            ts.xform = Xform.__jsonload__(data['xform'])
        if 'surface' in data:
            ts.m_surface = NurbsSurface.__jsonload__(data['surface'])
        if 'outer_loop' in data:
            ts.m_outer_loop = NurbsCurve.__jsonload__(data['outer_loop'])
        if 'inner_loops' in data:
            ts.m_inner_loops = [NurbsCurve.__jsonload__(l) for l in data['inner_loops']]
        return ts

    def json_dump(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def json_load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.__jsonload__(data)

    def json_dumps(self):
        return json.dumps(self.__jsondump__())

    @classmethod
    def json_loads(cls, json_string):
        return cls.__jsonload__(json.loads(json_string))

    def pb_dumps(self):
        from .proto import trimmedsurface_pb2
        proto = trimmedsurface_pb2.TrimmedSurface()
        proto.guid = self.guid
        proto.name = self.name
        proto.width = self.width

        # Surface
        srf_data = self.m_surface.pb_dumps()
        proto.surface.ParseFromString(srf_data)

        # Outer loop
        if self.is_trimmed():
            loop_data = self.m_outer_loop.pb_dumps()
            proto.outer_loop.ParseFromString(loop_data)

        # Inner loops
        for inner in self.m_inner_loops:
            loop_data = inner.pb_dumps()
            il = proto.inner_loops.add()
            il.ParseFromString(loop_data)

        # Color
        proto.surfacecolor.name = self.surfacecolor.name
        proto.surfacecolor.r = self.surfacecolor[0]
        proto.surfacecolor.g = self.surfacecolor[1]
        proto.surfacecolor.b = self.surfacecolor[2]
        proto.surfacecolor.a = self.surfacecolor[3]

        # Transform
        proto.xform.name = self.xform.name
        proto.xform.matrix.extend(self.xform.m)

        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data):
        from .proto import trimmedsurface_pb2
        proto = trimmedsurface_pb2.TrimmedSurface()
        proto.ParseFromString(data)

        ts = cls()
        ts.guid = proto.guid
        ts.name = proto.name
        ts.width = proto.width

        # Surface
        if proto.HasField('surface'):
            srf_data = proto.surface.SerializeToString()
            ts.m_surface = NurbsSurface.pb_loads(srf_data)

        # Outer loop
        if proto.HasField('outer_loop') and proto.outer_loop.cv_count > 0:
            loop_data = proto.outer_loop.SerializeToString()
            ts.m_outer_loop = NurbsCurve.pb_loads(loop_data)

        # Inner loops
        for il in proto.inner_loops:
            loop_data = il.SerializeToString()
            ts.m_inner_loops.append(NurbsCurve.pb_loads(loop_data))

        # Color
        ts.surfacecolor = Color(
            proto.surfacecolor.r,
            proto.surfacecolor.g,
            proto.surfacecolor.b,
            proto.surfacecolor.a
        )
        ts.surfacecolor.name = proto.surfacecolor.name

        # Transform
        ts.xform = Xform()
        ts.xform.name = proto.xform.name
        ts.xform.m = list(proto.xform.matrix)

        return ts

    def pb_dump(self, filepath):
        data = self.pb_dumps()
        with open(filepath, 'wb') as f:
            f.write(data)

    @classmethod
    def pb_load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = f.read()
        return cls.pb_loads(data)
