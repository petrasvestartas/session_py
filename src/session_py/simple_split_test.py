from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
import copy
import math


def _mapped(surface, x, y):
    """Point at fractions x and y along the first two control edges of a surface."""
    a = surface.get_cv(0, 0)
    return a + (surface.get_cv(1, 0) - a) * x + (surface.get_cv(0, 1) - a) * y


@MINI_TEST("SimpleSplit", "Split Curve By Curves")
def test_split_curve_by_curves():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Primitives
    from session_py.simple_split import split_curve_by_curves

    curve = NurbsCurve.create(False, 1, [Point(-2.0, 0.0, 0.0), Point(2.0, 0.0, 0.0)])
    cutter = NurbsCurve.create(False, 1, [Point(0.0, -2.0, 0.0), Point(0.0, 2.0, 0.0)])
    pieces = split_curve_by_curves(curve, [cutter], 1e-6)

    MINI_CHECK(len(pieces) == 2)
    MINI_CHECK(pieces[0].point_at_start().distance(Point(-2.0, 0.0, 0.0)) < 1e-6)
    MINI_CHECK(pieces[0].point_at_end().distance(Point(0.0, 0.0, 0.0)) < 1e-6)
    MINI_CHECK(pieces[1].point_at_end().distance(Point(2.0, 0.0, 0.0)) < 1e-6)
    MINI_CHECK(curve.point_at_end().distance(Point(2.0, 0.0, 0.0)) < 1e-6)

    skew = NurbsCurve.create(False, 1, [Point(0.0, -2.0, 1.0), Point(0.0, 2.0, 1.0)])

    MINI_CHECK(len(split_curve_by_curves(curve, [skew], 1e-6)) == 1)

    crossing = NurbsCurve.create(
        False,
        1,
        [
            Point(-2.0, -2.0, 0.0),
            Point(2.0, 2.0, 0.0),
            Point(-2.0, 2.0, 0.0),
            Point(2.0, -2.0, 0.0),
        ],
    )
    short_cut = NurbsCurve.create(
        False, 1, [Point(0.0, -0.2, 0.0), Point(0.0, 0.2, 0.0)]
    )

    MINI_CHECK(len(split_curve_by_curves(crossing, [short_cut], 1e-6)) == 3)

    circle = Primitives.circle(0.0, 0.0, 0.0, 1.0)
    chord = NurbsCurve.create(False, 1, [Point(-2.0, 0.5, 0.0), Point(2.0, 0.5, 0.0)])
    arcs = split_curve_by_curves(circle, [chord], 1e-6)

    MINI_CHECK(len(arcs) == 2)
    MINI_CHECK(arcs[0].is_rational() and arcs[1].is_rational())

    tangent = NurbsCurve.create(False, 1, [Point(-2.0, 1.0, 0.0), Point(2.0, 1.0, 0.0)])

    MINI_CHECK(len(split_curve_by_curves(circle, [tangent], 1e-6)) == 1)

    rejected = False

    try:
        split_curve_by_curves(curve, [curve], 1e-6)
    except ValueError:
        rejected = True

    MINI_CHECK(rejected)

    rejected = False

    try:
        split_curve_by_curves(curve, [cutter], math.nan)
    except ValueError:
        rejected = True

    MINI_CHECK(rejected)


@MINI_TEST("SimpleSplit", "Split BRep Face By Curves")
def test_split_brep_face_by_curves():
    from session_py import BRep
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Primitives
    from session_py.simple_split import split_brep_face_by_curves

    box = BRep.create_box(10.0, 10.0, 10.0)
    surface = box.m_surfaces[0]
    cutter = NurbsCurve.create(
        False, 1, [_mapped(surface, 0.5, -1.0), _mapped(surface, 0.5, 2.0)]
    )
    split = split_brep_face_by_curves(box, 0, [cutter], 1e-6)

    MINI_CHECK(split.face_count() == 7)
    MINI_CHECK(split.is_valid() and split.is_solid())
    MINI_CHECK(box.face_count() == 6)

    meshes = split.face_meshes_q(True, 20.0, 0.005)

    MINI_CHECK(abs(meshes[0].area() - 50.0) < 1e-6)
    MINI_CHECK(abs(meshes[6].area() - 50.0) < 1e-6)

    neighbor_area = 0.0

    for i in range(1, 6):
        neighbor_area += meshes[i].area()

    MINI_CHECK(abs(neighbor_area - 500.0) < 1e-6)

    closed = NurbsCurve.create(
        False,
        3,
        [
            _mapped(surface, 0.2, 0.3),
            _mapped(surface, 0.8, 0.3),
            _mapped(surface, 0.5, 0.9),
            _mapped(surface, 0.2, 0.3),
        ],
    )
    island = split_brep_face_by_curves(box, 0, [closed], 1e-6)

    MINI_CHECK(island.face_count() == 7 and island.is_solid())

    crossing = NurbsCurve.create(
        False, 1, [_mapped(surface, -1.0, 0.5), _mapped(surface, 2.0, 0.5)]
    )
    quarters = split_brep_face_by_curves(box, 0, [cutter, crossing], 1e-6)

    MINI_CHECK(quarters.face_count() == 9 and quarters.is_solid())

    repeated = split_brep_face_by_curves(split, 0, [crossing], 1e-6)

    MINI_CHECK(repeated.face_count() == 8 and repeated.is_solid())

    loop = NurbsCurve.create(
        False,
        1,
        [
            _mapped(surface, 0.2, 0.2),
            _mapped(surface, 0.8, 0.2),
            _mapped(surface, 0.8, 0.8),
            _mapped(surface, 0.2, 0.8),
            _mapped(surface, 0.2, 0.2),
        ],
    )
    regions = split_brep_face_by_curves(box, 0, [loop], 1e-6)

    MINI_CHECK(regions.face_count() == 7 and regions.is_solid())

    restored = BRep.file_json_loads(quarters.file_json_dumps())

    MINI_CHECK(restored.face_count() == 9 and restored.is_solid())

    protobuf = BRep.pb_loads(quarters.pb_dumps())

    MINI_CHECK(protobuf.face_count() == 9 and protobuf.is_solid())

    outer = NurbsCurve.create(
        False,
        1,
        [
            Point(0.0, 0.0, 0.0),
            Point(10.0, 0.0, 0.0),
            Point(10.0, 10.0, 0.0),
            Point(0.0, 10.0, 0.0),
            Point(0.0, 0.0, 0.0),
        ],
    )
    inner = NurbsCurve.create(
        False,
        1,
        [
            Point(3.0, 3.0, 0.0),
            Point(7.0, 3.0, 0.0),
            Point(7.0, 7.0, 0.0),
            Point(3.0, 7.0, 0.0),
            Point(3.0, 3.0, 0.0),
        ],
    )
    ring = BRep.from_nurbscurves([outer], [[inner]])
    through = NurbsCurve.create(
        False, 1, [Point(5.0, -1.0, 0.0), Point(5.0, 11.0, 0.0)]
    )
    divided = split_brep_face_by_curves(ring, 0, [through], 1e-6)

    MINI_CHECK(divided.face_count() == 2)

    outside = NurbsCurve.create(
        False, 1, [Point(1.0, -1.0, 0.0), Point(1.0, 11.0, 0.0)]
    )
    preserved = split_brep_face_by_curves(ring, 0, [outside], 1e-6)
    holes = 0

    for face in preserved.m_faces:
        holes += len(face.wires) - 1

    MINI_CHECK(preserved.face_count() == 2 and holes == 1)

    disk = BRep.from_nurbscurves([Primitives.circle(0.0, 0.0, 0.0, 5.0)])
    chord = NurbsCurve.create(False, 1, [Point(-6.0, 1.2, 0.0), Point(6.0, 1.2, 0.0)])
    halves = split_brep_face_by_curves(disk, 0, [chord], 1e-6)

    MINI_CHECK(halves.face_count() == 2)
    MINI_CHECK(disk.face_count() == 1)

    cylinder = BRep.create_cylinder(5.0, 10.0)
    body = cylinder.m_surfaces[cylinder.m_faces[0].surface_index]
    domain = body.domain(0)
    generator = body.iso_curve(1, (domain[0] + domain[1]) * 0.5)
    seamed = split_brep_face_by_curves(cylinder, 0, [generator], 1e-6)

    MINI_CHECK(seamed.face_count() == 4 and seamed.is_solid())
    MINI_CHECK(cylinder.face_count() == 3)


@MINI_TEST("SimpleSplit", "Split Surface By Curves")
def test_split_surface_by_curves():
    from session_py import BRep
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.simple_split import split_surface_by_curves

    surface = BRep.create_box(10.0, 10.0, 10.0).m_surfaces[0]
    cutter = NurbsCurve.create(
        False, 1, [_mapped(surface, 0.5, -1.0), _mapped(surface, 0.5, 2.0)]
    )
    split = split_surface_by_curves(surface, [cutter], 1e-6)

    MINI_CHECK(split.face_count() == 2)
    MINI_CHECK(split.is_valid() and not split.is_solid())

    outside = NurbsCurve.create(
        False, 1, [_mapped(surface, 2.0, -1.0), _mapped(surface, 2.0, 2.0)]
    )
    untouched = split_surface_by_curves(surface, [outside], 1e-6)

    MINI_CHECK(untouched.face_count() == 1)
    MINI_CHECK(surface.is_valid())

    invalid = copy.deepcopy(surface)
    invalid.set_cv(0, 0, Point(math.nan, 0.0, 0.0))
    rejected = False

    try:
        split_surface_by_curves(invalid, [outside], 1e-6)
    except ValueError:
        rejected = True

    MINI_CHECK(rejected)


@MINI_TEST("SimpleSplit", "Split Line By Curves")
def test_split_line_by_curves():
    from session_py import Line
    from session_py import NurbsCurve
    from session_py import Point
    from session_py.simple_split import split_line_by_curves

    line = Line.from_points(Point(-2.0, 0.0, 0.0), Point(2.0, 0.0, 0.0))
    line.name = "retained"
    line.width = 3.0
    line.dash = [1.0, 2.0]
    cutter = NurbsCurve.create(False, 1, [Point(0.0, -2.0, 0.0), Point(0.0, 2.0, 0.0)])
    pieces = split_line_by_curves(line, [cutter], 1e-6)

    MINI_CHECK(len(pieces) == 2)
    MINI_CHECK(pieces[0].point_at(1.0).distance(Point(0.0, 0.0, 0.0)) < 1e-6)
    MINI_CHECK(pieces[1].point_at(0.0).distance(Point(0.0, 0.0, 0.0)) < 1e-6)
    MINI_CHECK(
        pieces[0].name == line.name
        and pieces[0].width == line.width
        and pieces[0].dash == line.dash
    )
    MINI_CHECK(line.length() == 4.0)


@MINI_TEST("SimpleSplit", "Split Polyline By Curves")
def test_split_polyline_by_curves():
    from session_py import NurbsCurve
    from session_py import Point
    from session_py import Polyline
    from session_py.simple_split import split_polyline_by_curves

    polyline = Polyline(
        [Point(-2.0, 0.0, 0.0), Point(2.0, 0.0, 0.0), Point(2.0, 3.0, 0.0)]
    )
    polyline.name = "retained"
    polyline.width = 3.0
    polyline.dash = [1.0, 2.0]
    cutter = NurbsCurve.create(False, 1, [Point(0.0, -2.0, 0.0), Point(0.0, 2.0, 0.0)])
    pieces = split_polyline_by_curves(polyline, [cutter], 1e-6)

    MINI_CHECK(len(pieces) == 2)
    MINI_CHECK(pieces[0].point_count() == 2 and pieces[1].point_count() == 3)
    MINI_CHECK(pieces[1].get_point(1).distance(Point(2.0, 0.0, 0.0)) < 1e-6)
    MINI_CHECK(pieces[1].get_point(2).distance(Point(2.0, 3.0, 0.0)) < 1e-6)
    MINI_CHECK(
        pieces[0].name == polyline.name
        and pieces[0].width == polyline.width
        and pieces[0].dash == polyline.dash
    )
    MINI_CHECK(polyline.point_count() == 3)


if __name__ == "__main__":
    run_all()
