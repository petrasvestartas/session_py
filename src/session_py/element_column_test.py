from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("ColumnElement", "Constructor")
def test_column_constructor():
    from session_py import ColumnElement
    from session_py import Mesh

    c = ColumnElement(width=0.4, depth=0.4, height=3.0, name="col1")

    geo = c.geometry
    name = c.name
    guid = c.guid
    cstr = str(c)
    crepr = repr(c)

    ccopy = c.duplicate()
    c2 = ColumnElement(width=0.4, depth=0.4, height=3.0, name="col1")
    c3 = ColumnElement(width=0.5, depth=0.4, height=3.0, name="col1")

    MINI_CHECK(name == "col1")
    MINI_CHECK(guid is not None and len(guid) > 0)
    MINI_CHECK(isinstance(geo, Mesh))
    MINI_CHECK(c.width == 0.4)
    MINI_CHECK(c.depth == 0.4)
    MINI_CHECK(c.height == 3.0)
    MINI_CHECK(cstr == "ColumnElement(col1, 0.4, 0.4, 3.0)")
    MINI_CHECK(crepr == f"ColumnElement({guid}, col1, 0.4, 0.4, 3.0)")
    MINI_CHECK(ccopy == c and ccopy.guid != c.guid)
    MINI_CHECK(c == c2)
    MINI_CHECK(c != c3)


@MINI_TEST("ColumnElement", "Setters")
def test_column_setters():
    from session_py import ColumnElement

    c = ColumnElement()
    c.width = 0.5
    c.depth = 0.6
    c.height = 4.0

    MINI_CHECK(c.width == 0.5)
    MINI_CHECK(c.depth == 0.6)
    MINI_CHECK(c.height == 4.0)
    MINI_CHECK(c.geometry is not None)


@MINI_TEST("ColumnElement", "Center Line")
def test_column_center_line():
    from session_py import ColumnElement
    from session_py import Line

    c = ColumnElement(height=5.0)
    cl = c.center_line

    MINI_CHECK(isinstance(cl, Line))
    MINI_CHECK(TOLERANCE.is_close(cl.start()[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(cl.end()[2], 5.0))


@MINI_TEST("ColumnElement", "Extend")
def test_column_extend():
    from session_py import ColumnElement

    c = ColumnElement(height=3.0)
    c.extend(0.5)

    MINI_CHECK(TOLERANCE.is_close(c.height, 4.0))


@MINI_TEST("ColumnElement", "Aabb")
def test_column_aabb():
    from session_py import ColumnElement

    c = ColumnElement(width=0.4, depth=0.4, height=3.0)
    aabb = c.aabb

    MINI_CHECK(aabb is not None)
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 0.2))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 0.2))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 1.5))


@MINI_TEST("ColumnElement", "Compute Point")
def test_column_compute_point():
    from session_py import ColumnElement

    c = ColumnElement(width=0.4, depth=0.4, height=3.0)
    pt = c.point

    MINI_CHECK(TOLERANCE.is_close(pt[0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], 1.5))


@MINI_TEST("ColumnElement", "Session Geometry")
def test_column_session_geometry():
    from session_py import ColumnElement
    from session_py import Xform
    from session_py import Mesh

    c = ColumnElement(width=0.4, depth=0.4, height=3.0)
    c.session_transformation = Xform.translation(10.0, 0.0, 0.0)
    sg = c.session_geometry

    MINI_CHECK(isinstance(sg, Mesh))
    verts = list(sg.vertex.values())
    xs = [v.x for v in verts]
    MINI_CHECK(min(xs) > 9.0)


@MINI_TEST("ColumnElement", "Json Roundtrip")
def test_column_json_roundtrip():
    from session_py import ColumnElement
    from session_py import Xform
    from pathlib import Path

    c = ColumnElement(width=0.5, depth=0.6, height=4.0, name="json_col")
    c.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_column_element.json"
    c.json_dump(fname)
    loaded = ColumnElement.json_load(fname)

    MINI_CHECK(isinstance(loaded, ColumnElement))
    MINI_CHECK(loaded.name == "json_col")
    MINI_CHECK(TOLERANCE.is_close(loaded.width, 0.5))
    MINI_CHECK(TOLERANCE.is_close(loaded.depth, 0.6))
    MINI_CHECK(TOLERANCE.is_close(loaded.height, 4.0))


@MINI_TEST("ColumnElement", "Protobuf Roundtrip")
def test_column_protobuf_roundtrip():
    from session_py import ColumnElement
    from session_py import Xform
    from pathlib import Path

    c = ColumnElement(width=0.5, depth=0.6, height=4.0, name="proto_col")
    c.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_column_element.bin"
    c.pb_dump(path)
    loaded = ColumnElement.pb_load(path)

    MINI_CHECK(isinstance(loaded, ColumnElement))
    MINI_CHECK(loaded.name == "proto_col")
    MINI_CHECK(TOLERANCE.is_close(loaded.width, 0.5))
    MINI_CHECK(TOLERANCE.is_close(loaded.depth, 0.6))
    MINI_CHECK(TOLERANCE.is_close(loaded.height, 4.0))


@MINI_TEST("ColumnElement", "Polylines")
def test_column_polylines():
    from session_py import ColumnElement
    from session_py import Polyline

    c = ColumnElement(width=0.4, depth=0.4, height=3.0)
    pls = c.polylines

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(isinstance(pls[0], Polyline))
    for pl in pls:
        MINI_CHECK(pl.point_count() == 5)


@MINI_TEST("ColumnElement", "Planes")
def test_column_planes():
    from session_py import ColumnElement
    from session_py import Plane

    c = ColumnElement(width=0.4, depth=0.4, height=3.0)
    pls = c.planes

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(isinstance(pls[0], Plane))
    MINI_CHECK(TOLERANCE.is_close(pls[0].z_axis[2], -1.0))
    MINI_CHECK(TOLERANCE.is_close(pls[1].z_axis[2], 1.0))


@MINI_TEST("ColumnElement", "Edge Vectors")
def test_column_edge_vectors():
    from session_py import ColumnElement
    from session_py import Vector

    c = ColumnElement(width=0.4, depth=0.4, height=3.0)
    evs = c.edge_vectors

    MINI_CHECK(len(evs) == 12)
    MINI_CHECK(isinstance(evs[0], Vector))


@MINI_TEST("ColumnElement", "Axis")
def test_column_axis():
    from session_py import ColumnElement
    from session_py import Line

    c = ColumnElement(height=5.0)
    ax = c.axis

    MINI_CHECK(isinstance(ax, Line))
    MINI_CHECK(TOLERANCE.is_close(ax.start()[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(ax.end()[2], 5.0))


if __name__ == "__main__":
    run_all(language="python")
