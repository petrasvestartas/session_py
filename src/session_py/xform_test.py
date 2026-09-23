from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
from .tolerance import PI


@MINI_TEST("Xform", "Constructor")
def test_xform_constructor():
    from session_py import Xform
    from session_py import Point

    x = Xform()
    m00 = x.m[0]
    m11 = x.m[5]
    m22 = x.m[10]
    m33 = x.m[15]
    is_id = x.is_identity()
    xfrom = Xform.from_matrix([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 10.0, 15.0, 1.0])

    xset = Xform()
    xset[1, 3] = 7.0

    xstr = str(x)
    xrepr = repr(x)
    xcopy = x.duplicate()
    xother = Xform()

    t = Xform.translation(10.0, 0.0, 0.0)
    s = Xform.scale_xyz(2.0, 1.0, 1.0)
    combined = t * s
    p = Point(1.0, 0.0, 0.0)
    result = p.transformed(combined)

    t2 = Xform.translation(10.0, 0.0, 0.0)
    t2 *= s
    result2 = p.transformed(t2)

    MINI_CHECK(x.name == "my_xform")
    MINI_CHECK(x.guid != "")
    MINI_CHECK(m00 == 1.0 and m11 == 1.0 and m22 == 1.0 and m33 == 1.0)
    MINI_CHECK(is_id)
    MINI_CHECK(xfrom.m[12] == 5.0 and xfrom.m[13] == 10.0 and xfrom.m[14] == 15.0)
    MINI_CHECK(xfrom[0, 3] == 5.0 and xset.m[13] == 7.0)
    MINI_CHECK(
        xstr
        == "[1.000000, 0.000000, 0.000000, 0.000000]\n[0.000000, 1.000000, 0.000000, 0.000000]\n[0.000000, 0.000000, 1.000000, 0.000000]\n[0.000000, 0.000000, 0.000000, 1.000000]"
    )
    MINI_CHECK(xrepr == f"Xform(my_xform, {x.guid[:8]})")
    MINI_CHECK(xcopy == x and xcopy.guid != x.guid)
    MINI_CHECK(xother == x)
    MINI_CHECK(xfrom != x)
    MINI_CHECK(result[0] == 12.0 and result[1] == 0.0 and result[2] == 0.0)
    MINI_CHECK(result2[0] == 12.0 and result2[1] == 0.0 and result2[2] == 0.0)


@MINI_TEST("Xform", "From Axes")
def test_xform_from_axes():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    xf = Xform.from_axes(Vector(0, 1, 0), Vector(-1, 0, 0), Vector(0, 0, 1))
    p = Point(1, 2, 3).transformed(xf)

    MINI_CHECK(TOLERANCE.is_point_close(p, Point(-2, 1, 3)))


@MINI_TEST("Xform", "Translation")
def test_xform_translation():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    xf = Xform.translation(1.5, 1.0, 0.5)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(0.5, 0, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(2.5, 0, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(2.5, 2, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(0.5, 2, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(0.5, 0, 1.5)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(2.5, 0, 1.5)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(2.5, 2, 1.5)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(0.5, 2, 1.5)))


@MINI_TEST("Xform", "Rotation X")
def test_xform_rotation_x():
    import math
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    s = math.sqrt(2.0)
    xf = Xform.rotation_x(PI / 4.0)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1, 0, -s)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1, 0, -s)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(1, s, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-1, s, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1, -s, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1, -s, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(1, 0, s)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-1, 0, s)))


@MINI_TEST("Xform", "Rotation Y")
def test_xform_rotation_y():
    import math
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    s = math.sqrt(2.0)
    xf = Xform.rotation_y(PI / 4.0)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-s, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(0, -1, -s)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(0, 1, -s)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-s, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(0, -1, s)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(s, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(s, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(0, 1, s)))


@MINI_TEST("Xform", "Rotation Z")
def test_xform_rotation_z():
    import math
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    s = math.sqrt(2.0)
    xf = Xform.rotation_z(PI / 4.0)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(0, -s, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(s, 0, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(0, s, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-s, 0, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(0, -s, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(s, 0, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(0, s, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-s, 0, 1)))


@MINI_TEST("Xform", "Rotation Axis")
def test_xform_rotation_axis():
    import math
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    t = 1.0 / 3.0
    k = 2.0 / math.sqrt(3.0)
    axis = Vector(1.0, 1.0, 1.0)
    xf = Xform.rotation(axis, 2.0 * PI / 4.0)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1, -1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(-t, -t + k, -t - k)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(t - k, t + k, t)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-t - k, -t, -t + k)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-t + k, -t - k, -t)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(t + k, t, t - k)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(1, 1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(t, t - k, t + k)))
    MINI_CHECK(Xform.rotation(Vector.zero(), PI / 3.0).is_identity())


@MINI_TEST("Xform", "Rotation Around Line")
def test_xform_rotation_around_line():
    import math
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point
    from session_py import Line

    s = math.sqrt(2.0)
    line = Line(-1.0, -1.0, -1.0, -1.0, -1.0, 1.0)
    xf = Xform.rotation_around_line(line, PI / 4.0)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1, -1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(s - 1, s - 1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(-1, 2 * s - 1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-s - 1, s - 1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1, -1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(s - 1, s - 1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(-1, 2 * s - 1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-s - 1, s - 1, 1)))


@MINI_TEST("Xform", "Change Basis")
def test_xform_change_basis():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    o0 = Point(0, 0, 0)
    x0 = Vector(1, 0, 0)
    y0 = Vector(0, 1, 0)
    z0 = Vector(0, 0, 1)
    o1 = Point(0.5, -1.0, 0.5)
    x1 = Vector(1.2, 0.0, 0.0)
    y1 = Vector(0.3, -1.0, -0.15)
    z1 = Vector(0.0, 0.0, 1.1)
    xf = Xform.change_basis(o0, x0, y0, z0, o1, x1, y1, z1)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1, 0, -0.45)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1.4, 0, -0.45)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(2, -2, -0.75)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-0.4, -2, -0.75)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1, 0, 1.75)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1.4, 0, 1.75)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(2, -2, 1.45)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-0.4, -2, 1.45)))


@MINI_TEST("Xform", "From Change Of Basis")
def test_xform_from_change_of_basis():
    from session_py import Xform
    from session_py import Point
    from session_py import Polyline

    rect0 = Polyline(
        [
            Point(0.0, 0.0, 0.0),
            Point(2.0, 0.0, 0.0),
            Point(2.0, 3.0, 0.0),
            Point(0.0, 3.0, 0.0),
        ]
    )
    rect1 = Polyline([Point(0.0, 0.0, 4.0)])
    xf = Xform.from_change_of_basis(rect0, rect1)

    MINI_CHECK(TOLERANCE.is_close(xf.m[12], 1.0))
    MINI_CHECK(TOLERANCE.is_close(xf.m[13], 1.5))
    MINI_CHECK(TOLERANCE.is_close(xf.m[14], 2.0))


@MINI_TEST("Xform", "Plane To Plane")
def test_xform_plane_to_plane():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector
    from session_py import Plane

    pf = Plane(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
    pt = Plane(Point(2, 0, 0), Vector(0, 1, 0), Vector(-1, 0, 0))
    xf = Xform.plane_to_plane(pf, pt)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(1, 1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1, -1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(3, -1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(3, 1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(1, 1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1, -1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(3, -1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(3, 1, 1)))


@MINI_TEST("Xform", "World To Frame")
def test_xform_world_to_frame():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    origin = Point(1.0, 2.0, 3.0)
    x_axis = Vector(0.0, 1.0, 0.0)
    y_axis = Vector(0.0, 0.0, 1.0)
    z_axis = Vector(1.0, 0.0, 0.0)
    xf = Xform.world_to_frame(origin, x_axis, y_axis, z_axis)
    p = Point(1.0, 4.0, 6.0).transformed(xf)

    MINI_CHECK(TOLERANCE.is_close(p[0], 2.0))
    MINI_CHECK(TOLERANCE.is_close(p[1], 3.0))
    MINI_CHECK(TOLERANCE.is_close(p[2], 0.0))


@MINI_TEST("Xform", "Frame To World")
def test_xform_frame_to_world():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    origin = Point(1.0, 2.0, 3.0)
    x_axis = Vector(0.0, 1.0, 0.0)
    y_axis = Vector(0.0, 0.0, 1.0)
    z_axis = Vector(1.0, 0.0, 0.0)
    xf = Xform.frame_to_world(origin, x_axis, y_axis, z_axis)
    p = Point(2.0, 3.0, 0.0).transformed(xf)

    MINI_CHECK(TOLERANCE.is_close(p[0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(p[1], 4.0))
    MINI_CHECK(TOLERANCE.is_close(p[2], 6.0))


@MINI_TEST("Xform", "To Frame")
def test_xform_to_frame():
    from session_py import Xform
    from session_py import Plane
    from session_py import Point
    from session_py import Vector

    frame = Plane(Point(1, 2, 3), Vector(0, 1, 0), Vector(-1, 0, 0))
    xf = Xform.to_frame(frame)
    px = Point(1, 0, 0).transformed(xf)
    py = Point(0, 1, 0).transformed(xf)

    MINI_CHECK(TOLERANCE.is_point_close(px, Point(1, 3, 3)))
    MINI_CHECK(TOLERANCE.is_point_close(py, Point(0, 2, 3)))


@MINI_TEST("Xform", "Scale XYZ")
def test_xform_scale_xyz():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    xf = Xform.scale_xyz(1.5, 1.2, 1.8)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1.5, -1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1.5, -1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(1.5, 1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-1.5, 1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1.5, -1.2, 1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1.5, -1.2, 1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(1.5, 1.2, 1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-1.5, 1.2, 1.8)))


@MINI_TEST("Xform", "Scale Uniform")
def test_xform_scale_uniform():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    c = Point(0, 0, 0)
    xf = Xform.scale_uniform(c, 2.0)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-2, -2, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(2, -2, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(2, 2, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-2, 2, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-2, -2, 2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(2, -2, 2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(2, 2, 2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-2, 2, 2)))


@MINI_TEST("Xform", "Scale Non Uniform")
def test_xform_scale_non_uniform():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    c = Point(0, 0, 0)
    xf = Xform.scale_non_uniform(c, 1.5, 1.2, 1.8)
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1.5, -1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1.5, -1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(1.5, 1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-1.5, 1.2, -1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1.5, -1.2, 1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1.5, -1.2, 1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(1.5, 1.2, 1.8)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-1.5, 1.2, 1.8)))


@MINI_TEST("Xform", "Axis Rotation")
def test_xform_axis_rotation():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    xf = Xform.axis_rotation(90.0, Vector(0, 0, 1), True)
    p = Point(1, 0, 0).transformed(xf)

    MINI_CHECK(TOLERANCE.is_point_close(p, Point(0, 1, 0)))


@MINI_TEST("Xform", "Look At Right Handed")
def test_xform_look_at_right_handed():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    eye = Point(0, 3, 0)
    target = Point(0, 0, 0)
    xf = Xform.look_at_right_handed(eye, target, Vector(0, 0, 1))
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(1, -1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(-1, -1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(-1, -1, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(1, -1, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(1, 1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(-1, 1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(-1, 1, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(1, 1, -2)))


@MINI_TEST("Xform", "Look To Right Handed")
def test_xform_look_to_right_handed():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector

    eye = Point(0, 3, 0)
    direction = Vector(0, -1, 0)
    xf = Xform.look_to_right_handed(eye, direction, Vector(0, 0, 1))
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(1, -1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(-1, -1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(-1, -1, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(1, -1, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(1, 1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(-1, 1, -4)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(-1, 1, -2)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(1, 1, -2)))


@MINI_TEST("Xform", "Perspective")
def test_xform_perspective():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    t = 1.0 / 3.0
    view = Xform.translation(0, 0, -2)
    proj = Xform.perspective(PI / 2.0, 1.0, 1.0, 3.0)
    xf = proj * view
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-t, -t, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(t, -t, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(t, t, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-t, t, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(1, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-1, 1, 0)))


@MINI_TEST("Xform", "Orthographic")
def test_xform_orthographic():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    view = Xform.translation(0, 0, -2)
    proj = Xform.orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 3.0)
    xf = proj * view
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(xf).to_vertices_and_faces()[0]

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1, -1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1, -1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(1, 1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-1, 1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(1, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-1, 1, 0)))


@MINI_TEST("Xform", "Project To Plane")
def test_xform_project_to_plane():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector
    from session_py import Plane
    from session_py import Polyline

    plane = Plane(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
    shift = Xform.translation(0, 0, 1)
    proj = Xform.project_to_plane(plane)
    xf = proj * shift
    outline = Polyline(
        [
            Point(-1, -1, -1).transformed(xf),
            Point(1, -1, -1).transformed(xf),
            Point(1, 1, -1).transformed(xf),
            Point(-1, 1, -1).transformed(xf),
            Point(-1, -1, -1).transformed(xf),
        ]
    )
    points = outline.get_points()

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(1, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-1, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1, -1, 0)))


@MINI_TEST("Xform", "Project To Plane By Axis")
def test_xform_project_to_plane_by_axis():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector
    from session_py import Plane
    from session_py import Polyline

    plane = Plane(Point(0, 0, 0), Vector(1, 0, 0), Vector(0, 1, 0))
    direction = Vector(1, 0, 1)
    shift = Xform.translation(0, 0, 1)
    proj = Xform.project_to_plane_by_axis(plane, direction)
    xf = proj * shift
    outline = Polyline(
        [
            Point(-1, -1, 1).transformed(xf),
            Point(1, -1, -1).transformed(xf),
            Point(1, 1, -1).transformed(xf),
            Point(-1, 1, 1).transformed(xf),
            Point(-1, -1, 1).transformed(xf),
        ]
    )
    points = outline.get_points()

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-3, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1, -1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(1, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-3, 1, 0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-3, -1, 0)))


@MINI_TEST("Xform", "Transform Point")
def test_xform_transform_point():
    from session_py import Xform
    from session_py import Point

    t = Xform.translation(10.0, 20.0, 30.0)
    s = Xform.scale_xyz(2.0, 3.0, 4.0)
    composite = t * s
    p = composite.transform_point(Point(1.0, 1.0, 1.0))

    pr = Xform.identity()
    pr.m[0] = 1.2
    pr.m[5] = 0.8
    pr.m[10] = 1.1
    pr.m[14] = 0.5
    pr.m[11] = -1.0
    pr.m[15] = 0.0
    q = pr.transform_point(Point(1.0, 1.0, 2.0))

    MINI_CHECK(TOLERANCE.is_point_close(p, Point(12.0, 23.0, 34.0)))
    MINI_CHECK(TOLERANCE.is_point_close(q, Point(-0.6, -0.4, -1.35)))


@MINI_TEST("Xform", "Transform Vector")
def test_xform_transform_vector():
    from session_py import Xform
    from session_py import Vector

    t = Xform.translation(10.0, 20.0, 30.0)
    s = Xform.scale_xyz(2.0, 3.0, 4.0)
    composite = t * s
    v = composite.transform_vector(Vector(1.0, 1.0, 1.0))
    r = Xform.rotation_z(90.0, True)
    u = r.transform_vector(Vector.x_axis())

    MINI_CHECK(TOLERANCE.is_vector_close(v, Vector(2.0, 3.0, 4.0)))
    MINI_CHECK(TOLERANCE.is_vector_close(u, Vector.y_axis()))


@MINI_TEST("Xform", "Transform Geometry")
def test_xform_transform_geometry():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector
    from session_py import Line
    from session_py import Plane
    from session_py import Polyline

    t = Xform.translation(10.0, 20.0, 30.0)
    pt = Point(1.0, 2.0, 3.0)
    pt_transformed = pt.transformed(t)
    v = Vector(1.0, 0.0, 0.0)
    v_transformed = v.transformed(t)
    ln = Line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ln_transformed = ln.transformed(t)
    pl = Plane(Point(0.0, 0.0, 0.0), Vector(1.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0))
    pl_transformed = pl.transformed(t)
    poly = Polyline([Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(1.0, 1.0, 0.0)])
    points = poly.transformed(t).get_points()

    MINI_CHECK(TOLERANCE.is_point_close(pt_transformed, Point(11.0, 22.0, 33.0)))
    MINI_CHECK(
        v_transformed[0] == 1.0 and v_transformed[1] == 0.0 and v_transformed[2] == 0.0
    )
    MINI_CHECK(
        ln_transformed[0] == 10.0
        and ln_transformed[1] == 20.0
        and ln_transformed[2] == 30.0
    )
    MINI_CHECK(
        ln_transformed[3] == 11.0
        and ln_transformed[4] == 20.0
        and ln_transformed[5] == 30.0
    )
    MINI_CHECK(TOLERANCE.is_point_close(pl_transformed.origin, Point(10.0, 20.0, 30.0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(10.0, 20.0, 30.0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(11.0, 20.0, 30.0)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(11.0, 21.0, 30.0)))


@MINI_TEST("Xform", "Inverse")
def test_xform_inverse():
    from session_py import Xform
    from session_py import Mesh
    from session_py import Point

    t = Xform.translation(1.0, 0.5, 0.5)
    s = Xform.scale_xyz(1.5, 1.2, 1.3)
    composite = t * s
    inv = composite.inverse()
    mesh = Mesh.create_box(2, 2, 2)
    points = mesh.transformed(composite).transformed(inv).to_vertices_and_faces()[0]

    p = Xform.identity()
    p.m[0] = 1.2
    p.m[5] = 0.8
    p.m[10] = 1.1
    p.m[14] = 0.5
    p.m[11] = -1.0
    p.m[15] = 0.0
    pinv = p.inverse()
    prod = p * pinv

    MINI_CHECK(TOLERANCE.is_point_close(points[0], Point(-1, -1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[1], Point(1, -1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[2], Point(1, 1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[3], Point(-1, 1, -1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[4], Point(-1, -1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[5], Point(1, -1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[6], Point(1, 1, 1)))
    MINI_CHECK(TOLERANCE.is_point_close(points[7], Point(-1, 1, 1)))
    MINI_CHECK(prod.is_identity())


@MINI_TEST("Xform", "To Cols")
def test_xform_to_cols():
    from session_py import Xform

    xf = Xform.translation(1.0, 2.0, 3.0)
    cols = xf.to_cols()

    MINI_CHECK(TOLERANCE.is_close(cols[0][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(cols[1][1], 1.0))
    MINI_CHECK(TOLERANCE.is_close(cols[2][2], 1.0))
    MINI_CHECK(TOLERANCE.is_close(cols[3][3], 1.0))
    MINI_CHECK(TOLERANCE.is_close(cols[3][0], 1.0))
    MINI_CHECK(TOLERANCE.is_close(cols[3][1], 2.0))
    MINI_CHECK(TOLERANCE.is_close(cols[3][2], 3.0))


@MINI_TEST("Xform", "Uniform Scale")
def test_xform_uniform_scale():
    from session_py import Xform

    MINI_CHECK(TOLERANCE.is_close(Xform.scale_xyz(2.0, 2.0, 2.0).uniform_scale(), 2.0))
    MINI_CHECK(
        TOLERANCE.is_close(Xform.translation(1.0, 2.0, 3.0).uniform_scale(), 1.0)
    )


@MINI_TEST("Xform", "Eye")
def test_xform_eye():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    view = Xform.look_at_right_handed(
        Point(1.0, 2.0, 5.0), Point(0.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0)
    )
    perspective = Xform.perspective(PI / 2.0, 1.0, 1.0, 10.0) * view
    orthographic = Xform.orthographic(-2.0, 2.0, -1.0, 1.0, 1.0, 10.0) * view

    MINI_CHECK(TOLERANCE.is_point_close(perspective.eye(), Point(1.0, 2.0, 5.0)))
    MINI_CHECK(orthographic.eye().distance(Point(0.0, 0.0, 0.0)) > 1.0e8)


@MINI_TEST("Xform", "Ortho Half Height")
def test_xform_ortho_half_height():
    from session_py import Xform
    from session_py import Point
    from session_py import Vector

    view = Xform.look_at_right_handed(
        Point(1.0, 2.0, 5.0), Point(0.0, 0.0, 0.0), Vector(0.0, 1.0, 0.0)
    )
    perspective = Xform.perspective(PI / 2.0, 1.0, 1.0, 10.0) * view
    orthographic = Xform.orthographic(-2.0, 2.0, -1.0, 1.0, 1.0, 10.0) * view

    MINI_CHECK(TOLERANCE.is_close(perspective.ortho_half_height(), 0.0))
    MINI_CHECK(TOLERANCE.is_close(orthographic.ortho_half_height(), 1.0))


@MINI_TEST("Xform", "Json Roundtrip")
def test_xform_json_roundtrip():
    from session_py import Xform
    from pathlib import Path

    xform = Xform.translation(1.0, 2.0, 3.0)
    xform.name = "test_xform"

    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_xform.json"
    xform.file_json_dump(filename)

    loaded = Xform.file_json_load(filename)
    parsed = Xform.file_json_loads(xform.file_json_dumps())

    MINI_CHECK(loaded.name == "test_xform")
    MINI_CHECK(loaded.guid == xform.guid)
    MINI_CHECK(loaded == xform)
    MINI_CHECK(parsed == xform and parsed.guid == xform.guid)
    MINI_CHECK(
        TOLERANCE.is_close(loaded.m[12], 1.0) and TOLERANCE.is_close(loaded.m[13], 2.0)
    )
    MINI_CHECK(
        TOLERANCE.is_close(loaded.m[14], 3.0) and TOLERANCE.is_close(loaded.m[15], 1.0)
    )


@MINI_TEST("Xform", "Protobuf Roundtrip")
def test_xform_protobuf_roundtrip():
    from session_py import Xform
    from pathlib import Path

    xform = Xform.translation(1.0, 2.0, 3.0)
    xform.name = "test_xform_proto"

    guid = xform.guid
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_xform.bin"
    xform.pb_dump(filename)

    loaded = Xform.pb_load(filename)
    converted = Xform.from_proto(xform.to_proto())

    MINI_CHECK(loaded.name == "test_xform_proto")
    MINI_CHECK(loaded.guid == guid)
    MINI_CHECK(loaded == xform)
    MINI_CHECK(converted == xform and converted.guid == guid)
    MINI_CHECK(
        TOLERANCE.is_close(loaded.m[12], 1.0) and TOLERANCE.is_close(loaded.m[13], 2.0)
    )
    MINI_CHECK(
        TOLERANCE.is_close(loaded.m[14], 3.0) and TOLERANCE.is_close(loaded.m[15], 1.0)
    )


if __name__ == "__main__":
    run_all("python")
