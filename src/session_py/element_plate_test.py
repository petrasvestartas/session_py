from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE


@MINI_TEST("ElementPlate", "Constructor")
def test_plate_constructor():
    from session_py import ElementPlate
    from session_py import Point
    from session_py import Mesh

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.2, name="plate1")

    geo = p.geometry
    name = p.name
    guid = p.guid
    pstr = str(p)
    prepr = repr(p)

    pcopy = p.duplicate()
    p2 = ElementPlate(polygon=polygon, thickness=0.2, name="plate1")
    p3 = ElementPlate(polygon=polygon, thickness=0.5, name="plate1")

    MINI_CHECK(name == "plate1")
    MINI_CHECK(guid is not None and len(guid) > 0)
    MINI_CHECK(isinstance(geo, Mesh))
    MINI_CHECK(len(p.polygon) == 4)
    MINI_CHECK(p.thickness == 0.2)
    MINI_CHECK(pstr == "ElementPlate(plate1, 4 pts, 0.2)")
    MINI_CHECK(prepr == f"ElementPlate({guid}, plate1, 4 pts, 0.2)")
    MINI_CHECK(pcopy == p and pcopy.guid != p.guid)
    MINI_CHECK(p == p2)
    MINI_CHECK(p != p3)


@MINI_TEST("ElementPlate", "Default Polygon")
def test_plate_default_polygon():
    from session_py import ElementPlate
    from session_py import Mesh

    p = ElementPlate()

    MINI_CHECK(isinstance(p.geometry, Mesh))
    MINI_CHECK(len(p.polygon) == 4)
    MINI_CHECK(p.thickness == 0.1)


@MINI_TEST("ElementPlate", "Setters")
def test_plate_setters():
    from session_py import ElementPlate
    from session_py import Point

    p = ElementPlate()
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


@MINI_TEST("ElementPlate", "Mesh Topology")
def test_plate_mesh_topology():
    from session_py import ElementPlate
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.5)
    geo = p.geometry

    MINI_CHECK(len(geo.vertex) == 8)
    MINI_CHECK(len(geo.face) == 6)


@MINI_TEST("ElementPlate", "AABB")
def test_plate_aabb():
    from session_py import ElementPlate
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.2)
    aabb = p.aabb

    MINI_CHECK(aabb is not None)
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(aabb.half_size[2], 0.1))


@MINI_TEST("ElementPlate", "Compute Point")
def test_plate_compute_point():
    from session_py import ElementPlate
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.2)
    pt = p.point

    MINI_CHECK(TOLERANCE.is_close(pt[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pt[2], -0.1))


@MINI_TEST("ElementPlate", "Triangle Polygon")
def test_plate_triangle():
    from session_py import ElementPlate
    from session_py import Point

    polygon = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(0.5, 1, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.1)
    geo = p.geometry

    MINI_CHECK(len(geo.vertex) == 6)
    MINI_CHECK(len(geo.face) == 5)


@MINI_TEST("ElementPlate", "Json Roundtrip")
def test_plate_json_roundtrip():
    from session_py import ElementPlate
    from session_py import Point
    from pathlib import Path

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.3, name="json_plate")

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_plate_element.json"
    p.file_json_dump(fname)
    loaded = ElementPlate.file_json_load(fname)

    MINI_CHECK(isinstance(loaded, ElementPlate))
    MINI_CHECK(loaded.name == "json_plate")
    MINI_CHECK(TOLERANCE.is_close(loaded.thickness, 0.3))
    MINI_CHECK(len(loaded.polygon) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.polygon[1][0], 2.0))


@MINI_TEST("ElementPlate", "Protobuf Roundtrip")
def test_plate_protobuf_roundtrip():
    from session_py import ElementPlate
    from session_py import Point
    from pathlib import Path

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.3, name="proto_plate")

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_plate_element.bin"
    p.pb_dump(path)
    loaded = ElementPlate.pb_load(path)

    MINI_CHECK(isinstance(loaded, ElementPlate))
    MINI_CHECK(loaded.name == "proto_plate")
    MINI_CHECK(TOLERANCE.is_close(loaded.thickness, 0.3))
    MINI_CHECK(len(loaded.polygon) == 4)
    MINI_CHECK(TOLERANCE.is_close(loaded.polygon[1][0], 2.0))


@MINI_TEST("ElementPlate", "From Top Bottom")
def test_plate_from_top_bottom():
    from session_py import ElementPlate
    from session_py import Point

    bottom = [Point(0,0,0), Point(2,0,0), Point(2,2,0), Point(0,2,0), Point(0,0,0)]
    top    = [Point(0,0,1), Point(2,0,1), Point(2,2,1), Point(0,2,1), Point(0,0,1)]
    p = ElementPlate(polygon=bottom, polygon_top=top, name="tb_plate")
    MINI_CHECK(len(p.polygon) == 4)
    MINI_CHECK(len(p.polygon_top) == 4)
    MINI_CHECK(TOLERANCE.is_close(p.thickness, 1.0))
    MINI_CHECK(TOLERANCE.is_close(p.polygon[0][2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(p.polygon_top[0][2], 1.0))
    # Reversed argument order should auto-swap
    pr = ElementPlate(polygon=top, polygon_top=bottom, name="tb_plate_r")
    MINI_CHECK(TOLERANCE.is_close(pr.polygon[0][2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(pr.polygon_top[0][2], 1.0))


@MINI_TEST("ElementPlate", "Polylines")
def test_plate_polylines():
    from session_py import ElementPlate
    from session_py import Point
    from session_py import Polyline

    polygon = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.2)
    pls = p.polylines

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(isinstance(pls[0], Polyline))
    MINI_CHECK(pls[0].point_count() == 5)
    MINI_CHECK(pls[1].point_count() == 5)
    for i in range(2, 6):
        MINI_CHECK(pls[i].point_count() == 5)


@MINI_TEST("ElementPlate", "Planes")
def test_plate_planes():
    from session_py import ElementPlate
    from session_py import Point
    from session_py import Plane

    polygon = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.2)
    pls = p.planes

    MINI_CHECK(len(pls) == 6)
    MINI_CHECK(isinstance(pls[0], Plane))
    MINI_CHECK(TOLERANCE.is_close(pls[0].z_axis[2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(pls[1].z_axis[2], -1.0))


@MINI_TEST("ElementPlate", "Edge Vectors")
def test_plate_edge_vectors():
    from session_py import ElementPlate
    from session_py import Point
    from session_py import Vector

    polygon = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.2)
    evs = p.edge_vectors

    MINI_CHECK(len(evs) == 4)
    MINI_CHECK(isinstance(evs[0], Vector))
    MINI_CHECK(TOLERANCE.is_close(evs[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(evs[0][1], 0.0))
    MINI_CHECK(TOLERANCE.is_close(evs[1][0], 0.0))
    MINI_CHECK(TOLERANCE.is_close(evs[1][1], 1.0))


@MINI_TEST("ElementPlate", "Axis")
def test_plate_axis():
    from session_py import ElementPlate
    from session_py import Point
    from session_py import Line

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.4)
    ax = p.axis
    top = [Point(0, 0, 1), Point(2, 0, 1), Point(2, 2, 1), Point(0, 2, 1)]
    p2 = ElementPlate(polygon=polygon, polygon_top=top)
    ax2 = p2.axis

    MINI_CHECK(isinstance(ax, Line))
    MINI_CHECK(TOLERANCE.is_close(ax.start()[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(ax.start()[1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(ax.start()[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(ax.end()[2], -0.4))
    MINI_CHECK(TOLERANCE.is_close(ax2.start()[2], 0.0))
    MINI_CHECK(TOLERANCE.is_close(ax2.end()[2], 1.0))


@MINI_TEST("ElementPlate", "Joint Types")
def test_plate_joint_types():
    from session_py import ElementPlate

    p = ElementPlate()

    MINI_CHECK(len(p.joint_types) == 0)
    p.joint_types = [1, 2, 3, 4]
    MINI_CHECK(len(p.joint_types) == 4)
    MINI_CHECK(p.joint_types[0] == 1)
    MINI_CHECK(p.joint_types[3] == 4)


@MINI_TEST("ElementPlate", "J Mf")
def test_plate_j_mf():
    from session_py import ElementPlate

    p = ElementPlate()

    MINI_CHECK(len(p.j_mf) == 0)
    p.j_mf = [
        [(0, True, 0.5), (1, False, 0.3)],
        [],
        [(2, True, 0.8)],
    ]
    MINI_CHECK(len(p.j_mf) == 3)
    MINI_CHECK(len(p.j_mf[0]) == 2)
    MINI_CHECK(p.j_mf[0][0] == (0, True, 0.5))
    MINI_CHECK(p.j_mf[2][0][0] == 2)


@MINI_TEST("ElementPlate", "Key")
def test_plate_key():
    from session_py import ElementPlate

    p = ElementPlate()

    MINI_CHECK(p.key == "")
    p.key = "plate_A"
    MINI_CHECK(p.key == "plate_A")


@MINI_TEST("ElementPlate", "Component Plane")
def test_plate_component_plane():
    from session_py import ElementPlate
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    p = ElementPlate()

    MINI_CHECK(p.component_plane is None)
    cp = Plane(origin=Point(1, 2, 3), x_axis=Vector(1, 0, 0), y_axis=Vector(0, 1, 0))
    p.component_plane = cp
    MINI_CHECK(TOLERANCE.is_close(p.component_plane.origin[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(p.component_plane.origin[1], 2.0))


@MINI_TEST("ElementPlate", "Json Roundtrip Joinery")
def test_plate_json_roundtrip_joinery():
    from session_py import ElementPlate
    from session_py import Point
    from session_py import Plane
    from session_py import Vector
    from pathlib import Path

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.3, name="joinery_plate")
    p.joint_types = [1, 2, 3, 4]
    p.j_mf = [[(0, True, 0.5)], [], [(1, False, 0.3)]]
    p.key = "plate_A"
    p.component_plane = Plane(origin=Point(1, 2, 3), x_axis=Vector(1, 0, 0), y_axis=Vector(0, 1, 0))

    fname = Path(__file__).resolve().parents[2] / "serialization" / "test_plate_element_joinery.json"
    p.file_json_dump(fname)
    loaded = ElementPlate.file_json_load(fname)

    MINI_CHECK(loaded.joint_types == [1, 2, 3, 4])
    MINI_CHECK(len(loaded.j_mf) == 3)
    MINI_CHECK(loaded.j_mf[0][0] == [0, True, 0.5])
    MINI_CHECK(loaded.key == "plate_A")
    MINI_CHECK(loaded.component_plane is not None)
    MINI_CHECK(TOLERANCE.is_close(loaded.component_plane.origin[0], 1.0))


@MINI_TEST("ElementPlate", "Protobuf Roundtrip Joinery")
def test_plate_protobuf_roundtrip_joinery():
    from session_py import ElementPlate
    from session_py import Point
    from session_py import Plane
    from session_py import Vector
    from pathlib import Path

    polygon = [
        Point(0, 0, 0),
        Point(2, 0, 0),
        Point(2, 2, 0),
        Point(0, 2, 0),
    ]
    p = ElementPlate(polygon=polygon, thickness=0.3, name="joinery_plate")
    p.joint_types = [1, 2, 3, 4]
    p.j_mf = [[(0, True, 0.5)], [], [(1, False, 0.3)]]
    p.key = "plate_A"
    p.component_plane = Plane(origin=Point(1, 2, 3), x_axis=Vector(1, 0, 0), y_axis=Vector(0, 1, 0))

    path = Path(__file__).resolve().parents[2] / "serialization" / "test_plate_element_joinery.bin"
    p.pb_dump(path)
    loaded = ElementPlate.pb_load(path)

    MINI_CHECK(loaded.joint_types == [1, 2, 3, 4])
    MINI_CHECK(len(loaded.j_mf) == 3)
    MINI_CHECK(loaded.j_mf[0][0] == (0, True, 0.5))
    MINI_CHECK(loaded.key == "plate_A")
    MINI_CHECK(loaded.component_plane is not None)
    MINI_CHECK(TOLERANCE.is_close(loaded.component_plane.origin[0], 1.0))


if __name__ == "__main__":
    run_all(language="python")
