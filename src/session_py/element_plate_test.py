from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("PlateElement", "Constructor")
def test_plate_constructor():
    from session_py import PlateElement
    from session_py import Point
    from session_py import Mesh

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = PlateElement(polygon=polygon, thickness=0.2, name="plate1")

    geo = p.geometry
    name = p.name
    guid = p.guid
    pstr = str(p)
    prepr = repr(p)

    pcopy = p.duplicate()
    p2 = PlateElement(polygon=polygon, thickness=0.2, name="plate1")
    p3 = PlateElement(polygon=polygon, thickness=0.5, name="plate1")

    MINI_CHECK(name == "plate1")
    MINI_CHECK(guid is not None and len(guid) > 0)
    MINI_CHECK(isinstance(geo, Mesh))
    MINI_CHECK(len(p.polygon) == 4)
    MINI_CHECK(p.thickness == 0.2)
    MINI_CHECK(pstr == "PlateElement(plate1, 4 pts, 0.2)")
    MINI_CHECK(prepr == f"PlateElement({guid}, plate1, 4 pts, 0.2)")
    MINI_CHECK(pcopy == p and pcopy.guid != p.guid)
    MINI_CHECK(p == p2)
    MINI_CHECK(p != p3)


@MINI_TEST("PlateElement", "Default Polygon")
def test_plate_default_polygon():
    from session_py import PlateElement
    from session_py import Mesh

    p = PlateElement()

    MINI_CHECK(isinstance(p.geometry, Mesh))
    MINI_CHECK(len(p.polygon) == 4)
    MINI_CHECK(p.thickness == 0.1)


@MINI_TEST("PlateElement", "Setters")
def test_plate_setters():
    from session_py import PlateElement
    from session_py import Point

    p = PlateElement()
    p.thickness = 0.3
    p.polygon = [
        Point(0, 0, 0),
        Point(3, 0, 0),
        Point(3, 3, 0),
        Point(0, 3, 0),
    ]

    MINI_CHECK(p.thickness == 0.3)
    MINI_CHECK(len(p.polygon) == 4)
    MINI_CHECK(p.geometry is not None)


@MINI_TEST("PlateElement", "Mesh Topology")
def test_plate_mesh_topology():
    from session_py import PlateElement
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    p = PlateElement(polygon=polygon, thickness=0.5)
    geo = p.geometry

    MINI_CHECK(len(geo.vertex) == 8)
    MINI_CHECK(len(geo.face) == 6)


@MINI_TEST("PlateElement", "Aabb")
def test_plate_aabb():
    from session_py import PlateElement
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = PlateElement(polygon=polygon, thickness=0.2)
    aabb = p.aabb

    MINI_CHECK(aabb is not None)
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 0.1))


@MINI_TEST("PlateElement", "Compute Point")
def test_plate_compute_point():
    from session_py import PlateElement
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = PlateElement(polygon=polygon, thickness=0.2)
    pt = p.point

    MINI_CHECK(TOLERANCE.is_close(pt[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], -0.1))


@MINI_TEST("PlateElement", "Triangle Polygon")
def test_plate_triangle():
    from session_py import PlateElement
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(0.5, 1, 0),
    ]
    p = PlateElement(polygon=polygon, thickness=0.1)
    geo = p.geometry

    MINI_CHECK(len(geo.vertex) == 6)
    MINI_CHECK(len(geo.face) == 5)


@MINI_TEST("PlateElement", "Json Roundtrip")
def test_plate_json_roundtrip():
    from session_py import PlateElement
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = PlateElement(polygon=polygon, thickness=0.3, name="json_plate")
    p.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_plate_element.json"
    p.json_dump(fname)
    loaded = PlateElement.json_load(fname)

    MINI_CHECK(isinstance(loaded, PlateElement))
    MINI_CHECK(loaded.name == "json_plate")
    MINI_CHECK(TOLERANCE.is_close(loaded.thickness, 0.3))
    MINI_CHECK(len(loaded.polygon) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.polygon[1][0], 2.0))


@MINI_TEST("PlateElement", "Protobuf Roundtrip")
def test_plate_protobuf_roundtrip():
    from session_py import PlateElement
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = PlateElement(polygon=polygon, thickness=0.3, name="proto_plate")
    p.session_transformation = Xform.translation(1.0, 2.0, 3.0)

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_plate_element.bin"
    p.pb_dump(path)
    loaded = PlateElement.pb_load(path)

    MINI_CHECK(isinstance(loaded, PlateElement))
    MINI_CHECK(loaded.name == "proto_plate")
    MINI_CHECK(TOLERANCE.is_close(loaded.thickness, 0.3))
    MINI_CHECK(len(loaded.polygon) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.polygon[1][0], 2.0))


if __name__ == "__main__":
    run_all(language="python")
