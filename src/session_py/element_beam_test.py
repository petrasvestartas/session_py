from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("BeamElement", "Constructor")
def test_beam_constructor():
    from session_py import BeamElement
    from session_py import Mesh

    b = BeamElement(width=0.1, depth=0.2, length=3.0, name="beam1")

    geo = b.geometry
    name = b.name
    guid = b.guid
    bstr = str(b)
    brepr = repr(b)

    bcopy = b.duplicate()
    b2 = BeamElement(width=0.1, depth=0.2, length=3.0, name="beam1")
    b3 = BeamElement(width=0.1, depth=0.2, length=5.0, name="beam1")

    MINI_CHECK(name == "beam1")
    MINI_CHECK(guid is not None and len(guid) > 0)
    MINI_CHECK(isinstance(geo, Mesh))
    MINI_CHECK(b.width == 0.1)
    MINI_CHECK(b.depth == 0.2)
    MINI_CHECK(b.length == 3.0)
    MINI_CHECK(bstr == "BeamElement(beam1, 0.1, 0.2, 3.0)")
    MINI_CHECK(brepr == f"BeamElement({guid}, beam1, 0.1, 0.2, 3.0)")
    MINI_CHECK(bcopy == b and bcopy.guid != b.guid)
    MINI_CHECK(b == b2)
    MINI_CHECK(b != b3)


@MINI_TEST("BeamElement", "Setters")
def test_beam_setters():
    from session_py import BeamElement

    b = BeamElement()
    b.width = 0.15
    b.depth = 0.3
    b.length = 5.0

    MINI_CHECK(b.width == 0.15)
    MINI_CHECK(b.depth == 0.3)
    MINI_CHECK(b.length == 5.0)
    MINI_CHECK(b.geometry is not None)


@MINI_TEST("BeamElement", "Center Line")
def test_beam_center_line():
    from session_py import BeamElement
    from session_py import Line

    b = BeamElement(length=5.0)
    cl = b.center_line

    MINI_CHECK(isinstance(cl, Line))
    MINI_CHECK(TOLERANCE.is_close(cl.start()[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(cl.end()[2], 5.0))


@MINI_TEST("BeamElement", "Extend")
def test_beam_extend():
    from session_py import BeamElement

    b = BeamElement(length=3.0)
    b.extend(0.5)

    MINI_CHECK(TOLERANCE.is_close(b.length, 4.0))


@MINI_TEST("BeamElement", "Aabb")
def test_beam_aabb():
    from session_py import BeamElement

    b = BeamElement(width=0.1, depth=0.2, length=3.0)
    aabb = b.aabb

    MINI_CHECK(aabb is not None)
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 0.05))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 0.1))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 1.5))


@MINI_TEST("BeamElement", "Compute Point")
def test_beam_compute_point():
    from session_py import BeamElement

    b = BeamElement(width=0.1, depth=0.2, length=3.0)
    pt = b.point

    MINI_CHECK(TOLERANCE.is_close(pt[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 1.5))


@MINI_TEST("BeamElement", "Session Geometry")
def test_beam_session_geometry():
    from session_py import BeamElement
    from session_py import Xform
    from session_py import Mesh

    b = BeamElement(width=0.1, depth=0.2, length=3.0)
    b.session_transformation = Xform.translation(10.0, 0.0, 0.0)
    sg = b.session_geometry

    MINI_CHECK(isinstance(sg, Mesh))
    verts = list(sg.vertex.values())
    xs = [v.x for v in verts]

    MINI_CHECK(min(xs) > 9.0)


@MINI_TEST("BeamElement", "Json Roundtrip")
def test_beam_json_roundtrip():
    from session_py import BeamElement
    from session_py import Xform
    from pathlib import Path

    b = BeamElement(width=0.15, depth=0.3, length=5.0, name="json_beam")
    b.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_beam_element.json"
    b.json_dump(fname)
    loaded = BeamElement.json_load(fname)

    MINI_CHECK(isinstance(loaded, BeamElement))
    MINI_CHECK(loaded.name == "json_beam")
    MINI_CHECK(TOLERANCE.is_close(loaded.width, 0.15))
    MINI_CHECK(TOLERANCE.is_close(loaded.depth, 0.3))
    MINI_CHECK(TOLERANCE.is_close(loaded.length, 5.0))


@MINI_TEST("BeamElement", "Protobuf Roundtrip")
def test_beam_protobuf_roundtrip():
    from session_py import BeamElement
    from session_py import Xform
    from pathlib import Path

    b = BeamElement(width=0.15, depth=0.3, length=5.0, name="proto_beam")
    b.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_beam_element.bin"
    b.pb_dump(path)
    loaded = BeamElement.pb_load(path)

    MINI_CHECK(isinstance(loaded, BeamElement))
    MINI_CHECK(loaded.name == "proto_beam")
    MINI_CHECK(TOLERANCE.is_close(loaded.width, 0.15))
    MINI_CHECK(TOLERANCE.is_close(loaded.depth, 0.3))
    MINI_CHECK(TOLERANCE.is_close(loaded.length, 5.0))


@MINI_TEST("BeamElement", "Polylines")
def test_beam_polylines():
    from session_py import BeamElement
    from session_py import Polyline

    b = BeamElement(width=0.1, depth=0.2, length=3.0)
    pls = b.polylines

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(isinstance(pls[0], Polyline))
    for pl in pls:
        MINI_CHECK(pl.point_count() == 5)


@MINI_TEST("BeamElement", "Planes")
def test_beam_planes():
    from session_py import BeamElement
    from session_py import Plane

    b = BeamElement(width=0.1, depth=0.2, length=3.0)
    pls = b.planes

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(isinstance(pls[0], Plane))
    MINI_CHECK(TOLERANCE.is_close(pls[0].z_axis[2], -1.0))
    MINI_CHECK(TOLERANCE.is_close(pls[1].z_axis[2], 1.0))


@MINI_TEST("BeamElement", "Edge Vectors")
def test_beam_edge_vectors():
    from session_py import BeamElement
    from session_py import Vector

    b = BeamElement(width=0.1, depth=0.2, length=3.0)
    evs = b.edge_vectors

    MINI_CHECK(len(evs) == 12)
    MINI_CHECK(isinstance(evs[0], Vector))


@MINI_TEST("BeamElement", "Axis")
def test_beam_axis():
    from session_py import BeamElement
    from session_py import Line

    b = BeamElement(length=5.0)
    ax = b.axis

    MINI_CHECK(isinstance(ax, Line))
    MINI_CHECK(TOLERANCE.is_close(ax.start()[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(ax.end()[2], 5.0))


if __name__ == "__main__":
    run_all(language="python")
