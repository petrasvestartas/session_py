from .mini_test import MINI_TEST
from .mini_test import MINI_CHECK
from .mini_test import run_all
from .tolerance import TOLERANCE
import math


@MINI_TEST("Mesh", "Constructor")
def test_mesh_constructor():
    from session_py import Mesh
    from session_py import Polyline
    from session_py import Color
    from session_py.mesh import ColorMode

    vertices = Polyline.from_sides(6, 1.0, False).get_points()
    mesh = Mesh.from_vertices_and_faces(vertices, [[0, 1, 2, 3, 4, 5]])
    sstr = str(mesh)
    srepr = repr(mesh)
    mcopy = mesh.duplicate()
    MINI_CHECK(mesh.is_valid())
    mesh.name = "hexagon"

    palette = Color.palette()

    # set_objectcolor does not change color_mode
    mesh.set_objectcolor(Color.grey())
    MINI_CHECK(mesh.color_mode == ColorMode.OBJECTCOLOR)

    # set_pointcolors → color_mode = PointColors
    pc = []
    for i in range(mesh.number_of_vertices()):
        pc.append(palette[i % len(palette)])
    mesh.set_pointcolors(pc)
    MINI_CHECK(mesh.color_mode == ColorMode.POINTCOLORS)
    MINI_CHECK(len(mesh.get_pointcolors()) == mesh.number_of_vertices())

    # set_facecolors → color_mode = FaceColors
    fc = []
    for i in range(mesh.number_of_faces()):
        fc.append(palette[i % len(palette)])
    mesh.set_facecolors(fc)
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    MINI_CHECK(len(mesh.get_facecolors()) == mesh.number_of_faces())

    # set_linecolors does not change color_mode
    lc = []
    lw = [0.1] * mesh.number_of_edges()
    for i in range(mesh.number_of_edges()):
        lc.append(palette[i % len(palette)])
    mesh.set_linecolors(lc, lw)
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    MINI_CHECK(len(mesh.get_linecolors()) == mesh.number_of_edges())

    # clear_facecolors reverts color_mode only if currently FaceColors
    mesh.color_mode = ColorMode.FACECOLORS
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    mesh.clear_facecolors()
    MINI_CHECK(mesh.color_mode == ColorMode.OBJECTCOLOR)
    MINI_CHECK(len(mesh.get_facecolors()) == 0)

    # clear_pointcolors does not revert if color_mode != PointColors
    mesh.color_mode = ColorMode.FACECOLORS
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    mesh.clear_pointcolors()
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)

    # clear_linecolors does not change color_mode
    mesh.color_mode = ColorMode.POINTCOLORS
    mesh.clear_linecolors()
    MINI_CHECK(mesh.color_mode == ColorMode.POINTCOLORS)
    MINI_CHECK(len(mesh.get_linecolors()) == 0)


@MINI_TEST("Mesh", "From Polylines")
def test_mesh_from_polylines():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh.from_polylines([
        [
            Point(1.28955, 0, 1.127558),
            Point(0.85791, 0, 0.225512),
            Point(0.64209, -0.866025, -0.225512),
            Point(0.85791, -1.732051, 0.225512),
            Point(1.458565, -1.732051, 1.127558),
            Point(1.50537, -0.866025, 1.578581),
        ],
        [
            Point(0.64209, 0.866025, -0.225512),
            Point(0.114274, 0.866025, -0.686294),
            Point(-0.00537, 0, -1.578581),
            Point(0.21045, -0.866025, -1.127558),
            Point(0.64209, -0.866025, -0.225512),
            Point(0.85791, 0, 0.225512),
        ],
        [
            Point(1.28955, 1.732051, 1.127558),
            Point(0.85791, 1.732051, 0.225512),
            Point(0.64209, 0.866025, -0.225512),
            Point(0.85791, 0, 0.225512),
            Point(1.28955, -0, 1.127558),
            Point(1.853404, 0.866025, 1.578581),
        ],
    ], 0.001)
    MINI_CHECK(mesh.is_valid())


@MINI_TEST("Mesh", "From Lines")
def test_mesh_from_lines():
    from session_py import Mesh
    from session_py import Line
    from session_py import Point

    lines = [
        Line.from_points(Point(4.948083, -0.149798, 1.00765),
                         Point(4.395544, -0.996413, 1.196018)),
        Line.from_points(Point(3.866593, 0.371225, 1.376346),
                         Point(4.567265, 0.584361, 1.137476)),
        Line.from_points(Point(3.915298, -0.157402, 1.359741),
                         Point(3.282977, -0.051356, 1.575309)),
        Line.from_points(Point(4.286215, -0.224964, 1.23329),
                         Point(3.607284, -0.987075, 1.464748)),
        Line.from_points(Point(3.744351, 0.971574, 1.41802),
                         Point(3.266367, 0.841359, 1.580972)),
        Line.from_points(Point(4.567265, 0.584361, 1.137476),
                         Point(4.948083, -0.149798, 1.00765)),
        Line.from_points(Point(4.395544, -0.996413, 1.196018),
                         Point(3.607284, -0.987075, 1.464748)),
        Line.from_points(Point(3.915298, -0.157402, 1.359741),
                         Point(4.286215, -0.224964, 1.23329)),
        Line.from_points(Point(3.282977, -0.051356, 1.575309),
                         Point(3.266367, 0.841359, 1.580972)),
        Line.from_points(Point(3.744351, 0.971574, 1.41802),
                         Point(3.866593, 0.371225, 1.376346)),
    ]
    mesh = Mesh.from_lines(lines, True)
    MINI_CHECK(mesh.is_valid())


@MINI_TEST("Mesh", "From Polygon With Holes")
def test_mesh_from_polygon_with_holes():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh.from_polygon_with_holes([
        [
            Point(8.940934, 0.917382, 0.049546),
            Point(8.930493, 1.36458, 0.251429),
            Point(8.954508, 1.595448, 0.346958),
            Point(9.457671, 1.821395, 0.298639),
            Point(9.717078, 1.014296, -0.136839),
            Point(9.363048, 0.91534, -0.07616),
            Point(9.33327, 0.459713, -0.269899),
            Point(9.065708, 0.635281, -0.112748),
        ],
        [
            Point(7.494779, -0.556523, -0.178103),
            Point(6.542877, 0.148384, 0.416685),
            Point(6.967337, 2.119511, 1.167431),
            Point(11.204553, 2.961749, 0.289102),
            Point(9.658416, 0.465135, -0.363618),
            Point(10.247775, -1.032727, -1.203717),
        ],
        [
            Point(7.922105, 0.548716, 0.186877),
            Point(7.410178, 0.844297, 0.469625),
            Point(7.408889, 1.185147, 0.621527),
            Point(7.885956, 1.424645, 0.586947),
            Point(8.178727, 1.32996, 0.458299),
            Point(8.307609, 0.88254, 0.2213),
            Point(7.950364, 0.924872, 0.345738),
        ],
    ], True)
    MINI_CHECK(mesh.is_valid())

    mesh_sorted = Mesh.from_polygon_with_holes([
        [Point(1, 1, 0), Point(3, 1, 0), Point(3, 3, 0), Point(1, 3, 0)],
        [Point(0, 0, 0), Point(4, 0, 0), Point(4, 4, 0), Point(0, 4, 0)],
    ], True)
    MINI_CHECK(mesh_sorted.is_valid())


@MINI_TEST("Mesh", "Loft")
def test_mesh_loft():
    from session_py import Mesh
    from session_py import Point
    from session_py import Polyline

    bottom = [
        Polyline([
            Point(13.20069, -0.556523, -0.178103),
            Point(12.248787, 0.148384, 0.416685),
            Point(12.673247, 2.119511, 1.167431),
            Point(16.910464, 2.961749, 0.289102),
            Point(15.364327, 0.465135, -0.363618),
            Point(15.953685, -1.032727, -1.203717),
            Point(13.20069, -0.556523, -0.178103),
        ]),
        Polyline([
            Point(14.646845, 0.917382, 0.049546),
            Point(14.636404, 1.36458, 0.251429),
            Point(14.660418, 1.595448, 0.346958),
            Point(15.163581, 1.821395, 0.298639),
            Point(15.422988, 1.014296, -0.136839),
            Point(15.068958, 0.91534, -0.07616),
            Point(15.03918, 0.459713, -0.269899),
            Point(14.771618, 0.635281, -0.112748),
            Point(14.646845, 0.917382, 0.049546),
        ]),
        Polyline([
            Point(13.628016, 0.548716, 0.186877),
            Point(13.116088, 0.844297, 0.469625),
            Point(13.114799, 1.185147, 0.621527),
            Point(13.591866, 1.424645, 0.586947),
            Point(13.884637, 1.32996, 0.458299),
            Point(14.013519, 0.88254, 0.2213),
            Point(13.656275, 0.924872, 0.345738),
            Point(13.628016, 0.548716, 0.186877),
        ]),
    ]
    top = [
        Polyline([
            Point(13.375135, -0.818817, 0.411936),
            Point(12.423233, -0.113909, 1.006724),
            Point(12.847692, 1.857217, 1.75747),
            Point(17.084909, 2.699455, 0.879141),
            Point(15.538772, 0.202841, 0.226421),
            Point(16.12813, -1.295021, -0.613678),
            Point(13.375135, -0.818817, 0.411936),
        ]),
        Polyline([
            Point(14.82129, 0.655088, 0.639585),
            Point(14.810849, 1.102286, 0.841468),
            Point(14.834864, 1.333154, 0.936997),
            Point(15.338026, 1.559101, 0.888678),
            Point(15.597433, 0.752002, 0.4532),
            Point(15.243404, 0.653046, 0.513879),
            Point(15.213626, 0.197419, 0.32014),
            Point(14.946063, 0.372987, 0.477291),
            Point(14.82129, 0.655088, 0.639585),
        ]),
        Polyline([
            Point(13.802461, 0.286422, 0.776916),
            Point(13.290534, 0.582003, 1.059664),
            Point(13.289245, 0.922853, 1.211566),
            Point(13.766312, 1.162351, 1.176986),
            Point(14.059082, 1.067666, 1.048338),
            Point(14.187964, 0.620246, 0.811339),
            Point(13.83072, 0.662578, 0.935777),
            Point(13.802461, 0.286422, 0.776916),
        ]),
    ]
    mesh = Mesh.loft(bottom, top, True)
    MINI_CHECK(mesh.is_valid())
    MINI_CHECK(mesh.is_closed())

    mesh_no_cap = Mesh.loft(bottom, top, False)
    MINI_CHECK(mesh_no_cap.is_valid())
    MINI_CHECK(not mesh_no_cap.is_closed())


@MINI_TEST("Mesh", "From Polygon With Holes Many")
def test_mesh_from_polygon_with_holes_many():
    from session_py import Mesh
    from session_py import Point

    inputs = []
    for i in range(4):
        x = i * 7.0
        inputs.append([
            [Point(x, 0, 0), Point(x+5, 0, 0), Point(x+5, 5, 0), Point(x, 5, 0)],
            [Point(x+1, 1, 0), Point(x+4, 1, 0), Point(x+4, 4, 0), Point(x+1, 4, 0)],
        ])
    meshes = Mesh.from_polygon_with_holes_many(inputs)
    for m in meshes:
        MINI_CHECK(m.is_valid())
    meshes_seq = Mesh.from_polygon_with_holes_many(inputs, False, False)
    MINI_CHECK(meshes_seq[0].number_of_faces() == meshes[0].number_of_faces())


@MINI_TEST("Mesh", "Loft Many")
def test_mesh_loft_many():
    from session_py import Mesh
    from session_py import Point
    from session_py import Polyline

    loft_inputs = []
    for i in range(6):
        x = i * 3.0
        b = Polyline([
            Point(x, 0, 0), Point(x+1, 0, 0), Point(x+1, 1, 0),
            Point(x, 1, 0), Point(x, 0, 0),
        ])
        t = Polyline([
            Point(x, 0, 1+i*0.5), Point(x+1, 0, 1+i*0.5),
            Point(x+1, 1, 1+i*0.5), Point(x, 1, 1+i*0.5), Point(x, 0, 1+i*0.5),
        ])
        loft_inputs.append(([b], [t]))
    meshes = Mesh.loft_many(loft_inputs)
    for m in meshes:
        MINI_CHECK(m.is_valid())
        MINI_CHECK(m.is_closed())
    meshes_seq = Mesh.loft_many(loft_inputs, True, False)
    MINI_CHECK(meshes_seq[0].number_of_faces() == meshes[0].number_of_faces())


@MINI_TEST("Mesh", "Loft with quads and triangles")
def test_mesh_loft_panels():
    from session_py import Mesh
    from session_py import Point
    from session_py import Color

    top7 = [
        [
            Point(250,-250,500),
            Point(250,250,500),
            Point(-250,250,500),
            Point(-250,-250,500),
            Point(250,-250,500),
        ],
        [
            Point(-250,500,250),
            Point(-250,250,500),
            Point(250,250,500),
            Point(250,500,250),
            Point(-250,500,250),
        ],
        [
            Point(250,-250,500),
            Point(500,-250,250),
            Point(500,250,250),
            Point(250,250,500),
            Point(250,-250,500),
        ],
        [
            Point(250,500,250),
            Point(250,250,500),
            Point(500,250,250),
            Point(250,500,250),
        ],
        [
            Point(-250,500,250),
            Point(250,500,250),
            Point(250,500,-250),
            Point(-250,500,-250),
            Point(-250,500,250),
        ],
        [
            Point(250,500,250),
            Point(500,250,250),
            Point(500,250,-250),
            Point(250,500,-250),
            Point(250,500,250),
        ],
        [
            Point(500,-250,250),
            Point(500,-250,-250),
            Point(500,250,-250),
            Point(500,250,250),
            Point(500,-250,250),
        ],
    ]
    bot7 = [
        [
            Point(270.710678,-250,550),
            Point(270.710678,265.891862,550),
            Point(265.891862,270.710678,550),
            Point(-250,270.710678,550),
            Point(-250,-250,550),
            Point(270.710678,-250,550),
        ],
        [
            Point(270.710678,-250,550),
            Point(550,-250,270.710678),
            Point(550,265.891862,270.710678),
            Point(270.710678,265.891862,550),
            Point(270.710678,-250,550),
        ],
        [
            Point(-250,550,270.710678),
            Point(-250,270.710678,550),
            Point(265.891862,270.710678,550),
            Point(265.891862,550,270.710678),
            Point(-250,550,270.710678),
        ],
        [
            Point(265.891862,550,270.710678),
            Point(265.891862,270.710678,550),
            Point(270.710678,265.891862,550),
            Point(550,265.891862,270.710678),
            Point(550,270.710678,265.891862),
            Point(270.710678,550,265.891862),
            Point(265.891862,550,270.710678),
        ],
        [
            Point(-250,550,270.710678),
            Point(265.891862,550,270.710678),
            Point(270.710678,550,265.891862),
            Point(270.710678,550,-250),
            Point(-250,550,-250),
            Point(-250,550,270.710678),
        ],
        [
            Point(270.710678,550,265.891862),
            Point(550,270.710678,265.891862),
            Point(550,270.710678,-250),
            Point(270.710678,550,-250),
            Point(270.710678,550,265.891862),
        ],
        [
            Point(550,-250,270.710678),
            Point(550,-250,-250),
            Point(550,270.710678,-250),
            Point(550,270.710678,265.891862),
            Point(550,265.891862,270.710678),
            Point(550,-250,270.710678),
        ],
    ]
    panels, adj, top_mesh, bot_mesh = Mesh.loft_panels(top7, bot7, 0.001)

    # Color faces: blue=top cap, red=bot cap, gray=quad wall, yellow=tri wall
    for i, panel in enumerate(panels):
        face_colors = []
        for fk, role in panel.face_roles.items():
            if role == "TopCap": face_colors.append(Color.blue())
            elif role == "BotCap": face_colors.append(Color.red())
            elif role == "TriWall": face_colors.append(Color.yellow())
            else: face_colors.append(Color.grey())
        panel.mesh.set_facecolors(face_colors)

    # face centroids labelled with panel index
    for i, panel in enumerate(panels):
        c = panel.mesh.centroid()
        c.name = f"p{i}"

    # adjacency: for each shared edge — text dot at midpoint labelled "p{i}f{idx}<->p{j}f{idx}"
    for pair in adj:
        w = panels[pair.pi].wall_faces[pair.wi]
        pt = panels[pair.pi].mesh.face_centroid(w.face_key)
        pt.name = f"p{pair.pi} f{w.face_index} - p{pair.pj} f{panels[pair.pj].wall_faces[pair.wj].face_index}"
    MINI_CHECK(len(panels) == 7)
    MINI_CHECK(panels[0].mesh.is_valid())
    MINI_CHECK(panels[1].mesh.is_valid())
    MINI_CHECK(panels[2].mesh.is_valid())
    MINI_CHECK(panels[3].mesh.is_valid())
    MINI_CHECK(panels[4].mesh.is_valid())
    MINI_CHECK(panels[5].mesh.is_valid())
    MINI_CHECK(panels[6].mesh.is_valid())
    MINI_CHECK(len(adj) == 9)
    MINI_CHECK(adj[0].pi == 0 and adj[0].pj == 2)
    MINI_CHECK(adj[1].pi == 0 and adj[1].pj == 1)
    MINI_CHECK(adj[2].pi == 1 and adj[2].pj == 3)
    MINI_CHECK(adj[3].pi == 1 and adj[3].pj == 4)
    MINI_CHECK(adj[4].pi == 2 and adj[4].pj == 6)
    MINI_CHECK(adj[5].pi == 2 and adj[5].pj == 3)
    MINI_CHECK(adj[6].pi == 3 and adj[6].pj == 5)
    MINI_CHECK(adj[7].pi == 4 and adj[7].pj == 5)
    MINI_CHECK(adj[8].pi == 5 and adj[8].pj == 6)


@MINI_TEST("Mesh", "Boolean Queries")
def test_mesh_boolean_queries():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh.from_polylines([
        [
            Point(1.28955, 0, 1.127558),
            Point(0.85791, 0, 0.225512),
            Point(0.64209, -0.866025, -0.225512),
            Point(0.85791, -1.732051, 0.225512),
            Point(1.458565, -1.732051, 1.127558),
            Point(1.50537, -0.866025, 1.578581),
        ],
        [
            Point(0.64209, 0.866025, -0.225512),
            Point(0.114274, 0.866025, -0.686294),
            Point(-0.00537, 0, -1.578581),
            Point(0.21045, -0.866025, -1.127558),
            Point(0.64209, -0.866025, -0.225512),
            Point(0.85791, 0, 0.225512),
        ],
        [
            Point(1.28955, 1.732051, 1.127558),
            Point(0.85791, 1.732051, 0.225512),
            Point(0.64209, 0.866025, -0.225512),
            Point(0.85791, 0, 0.225512),
            Point(1.28955, -0, 1.127558),
            Point(1.853404, 0.866025, 1.578581),
        ],
    ], 0.001)
    v0 = 1
    v1 = 2
    v2 = 3
    f0 = 0

    empty = mesh.is_empty()
    MINI_CHECK(not empty)

    valid = mesh.is_valid()
    MINI_CHECK(valid)

    closed = mesh.is_closed()
    MINI_CHECK(not closed)

    vertex_on_boundary = mesh.is_vertex_on_boundary(v0)
    MINI_CHECK(not vertex_on_boundary)

    edge_not_on_boundary = mesh.is_edge_on_boundary(v0, v1)
    MINI_CHECK(not edge_not_on_boundary)

    edge_on_boundary = mesh.is_edge_on_boundary(v1, v2)
    MINI_CHECK(edge_on_boundary)

    face_on_boundary = mesh.is_face_on_boundary(f0)
    MINI_CHECK(face_on_boundary)


@MINI_TEST("Mesh", "Attributes")
def test_mesh_attributes():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh.create_box(1.0, 1.0, 1.0)

    n_vertices = mesh.number_of_vertices()
    MINI_CHECK(n_vertices == 8)

    n_faces = mesh.number_of_faces()
    MINI_CHECK(n_faces == 6)

    n_edges = mesh.number_of_edges()
    MINI_CHECK(n_edges == 12)

    euler = mesh.euler()
    MINI_CHECK(euler == 2)

    vertices, faces = mesh.to_vertices_and_faces()
    MINI_CHECK(len(faces) == n_faces)
    MINI_CHECK(len(vertices) == n_vertices)
    MINI_CHECK(TOLERANCE.is_point_close(vertices[0], Point(-0.5, -0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(vertices[1], Point( 0.5, -0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(vertices[2], Point( 0.5,  0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(vertices[3], Point(-0.5,  0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(vertices[4], Point(-0.5, -0.5,  0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(vertices[5], Point( 0.5, -0.5,  0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(vertices[6], Point( 0.5,  0.5,  0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(vertices[7], Point(-0.5,  0.5,  0.5)))
    MINI_CHECK(faces[0] == [0, 3, 2, 1])
    MINI_CHECK(faces[1] == [4, 5, 6, 7])
    MINI_CHECK(faces[2] == [0, 1, 5, 4])
    MINI_CHECK(faces[3] == [2, 3, 7, 6])
    MINI_CHECK(faces[4] == [0, 4, 7, 3])
    MINI_CHECK(faces[5] == [1, 2, 6, 5])

    vertex_to_index = mesh.vertex_index()
    MINI_CHECK(len(vertex_to_index) == n_vertices)
    MINI_CHECK(vertex_to_index[0] == 0)
    MINI_CHECK(vertex_to_index[1] == 1)
    MINI_CHECK(vertex_to_index[2] == 2)
    MINI_CHECK(vertex_to_index[3] == 3)
    MINI_CHECK(vertex_to_index[4] == 4)
    MINI_CHECK(vertex_to_index[5] == 5)
    MINI_CHECK(vertex_to_index[6] == 6)
    MINI_CHECK(vertex_to_index[7] == 7)

    # sparse keys: key != index
    mesh2 = Mesh()
    k0 = mesh2.add_vertex(Point(0.0, 0.0, 0.0))
    k1 = mesh2.add_vertex(Point(1.0, 0.0, 0.0), 5)
    k2 = mesh2.add_vertex(Point(0.0, 1.0, 0.0), 10)
    MINI_CHECK(k0 == 0)
    MINI_CHECK(k1 == 5)
    MINI_CHECK(k2 == 10)
    vertex_to_index = mesh2.vertex_index()
    v0 = vertex_to_index[0]
    v5 = vertex_to_index[5]
    v10 = vertex_to_index[10]
    MINI_CHECK(v0  == 0)
    MINI_CHECK(v5  == 1)
    MINI_CHECK(v10 == 2)


@MINI_TEST("Mesh", "Edges")
def test_mesh_edges():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(1.0, 1.0, 0.0))
    v3 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2, v3])

    edges = mesh.edges()
    MINI_CHECK(len(edges) == 4)
    MINI_CHECK(isinstance(edges[0], tuple))
    MINI_CHECK(edges[0] == (0, 3))


@MINI_TEST("Mesh", "Vertex and Face Operations")
def test_mesh_vertex_and_face_operations():
    from session_py import Mesh
    from session_py import Point

    # add_vertex — None key auto-assigns sequentially from 0
    mesh = Mesh()
    v0 = mesh.add_vertex(Point(1.0, 2.0, 3.0))
    MINI_CHECK(v0 == 0)
    MINI_CHECK(mesh.number_of_vertices() == 1)
    MINI_CHECK(not mesh.is_empty())
    v1 = mesh.add_vertex(Point(4.0, 5.0, 6.0), 42)
    MINI_CHECK(v1 == 42)
    MINI_CHECK(mesh.number_of_vertices() == 2)

    # add_face
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    f = mesh.add_face([v0, v1, v2])
    MINI_CHECK(f is not None)
    invalid1 = mesh.add_face([v0, v1])
    MINI_CHECK(invalid1 is None)
    invalid2 = mesh.add_face([v0, v1, v0])
    MINI_CHECK(invalid2 is None)

    # clear
    mesh.clear()
    MINI_CHECK(mesh.is_empty())
    MINI_CHECK(mesh.number_of_vertices() == 0)
    MINI_CHECK(mesh.number_of_faces() == 0)

    # unify_winding — two triangles sharing edge p1-p2, f1 has same-direction halfedge (wrong winding)
    p0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    p1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    p2 = mesh.add_vertex(Point(1.0, 1.0, 0.0))
    p3 = mesh.add_vertex(Point(2.0, 1.0, 0.0))
    f0 = mesh.add_face([p0, p1, p2])  # +z normal
    f1 = mesh.add_face([p1, p2, p3])  # -z normal (wrong: same halfedge dir)

    n0_before = mesh.face_normal(f0)
    n1_before = mesh.face_normal(f1)
    MINI_CHECK(n0_before is not None and n1_before is not None)
    MINI_CHECK(n0_before.dot(n1_before) < 0.0)  # wrong: normals point opposite ways

    mesh.unify_winding()

    n0_after = mesh.face_normal(f0)
    n1_after = mesh.face_normal(f1)
    MINI_CHECK(n0_after is not None and n1_after is not None)
    MINI_CHECK(n0_after.dot(n1_after) > 0.0)  # correct: normals agree

    # Unweld and weld
    box = Mesh.create_box(1.0, 1.0, 1.0)
    u = box.unweld()
    MINI_CHECK(u.number_of_vertices() == 24)

    w = u.weld(0.001)
    MINI_CHECK(w.number_of_vertices() == 8)
    MINI_CHECK(w.number_of_faces() == 6)
    for vk in w.vertex:
        MINI_CHECK(len(w.vertex_faces(vk)) == 3)


@MINI_TEST("Mesh", "Connectivity Queries")
def test_mesh_connectivity_queries():
    from session_py import Mesh
    from session_py import Point

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(1.0, 1.0, 0.0))
    v3 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    v4 = mesh.add_vertex(Point(2.0, 0.0, 0.0))
    f0 = mesh.add_face([v0, v1, v2, v3])
    f1 = mesh.add_face([v1, v4, v2])

    # vertex_position
    pos = mesh.vertex_position(v0)
    MINI_CHECK(pos is not None)
    MINI_CHECK(TOLERANCE.is_point_close(pos, Point(0.0, 0.0, 0.0)))
    MINI_CHECK(mesh.vertex_position(999) is None)

    # face_vertices
    fv = mesh.face_vertices(f0)
    MINI_CHECK(fv is not None)
    MINI_CHECK(len(fv) == 4)
    MINI_CHECK(fv[0] == v0 and fv[1] == v1 and fv[2] == v2 and fv[3] == v3)

    # vertex_neighbors
    nb = mesh.vertex_neighbors(v1)
    MINI_CHECK(len(nb) == 3)

    # vertex_faces
    vf0 = mesh.vertex_faces(v0)
    MINI_CHECK(len(vf0) == 1)
    vf1 = mesh.vertex_faces(v1)
    MINI_CHECK(len(vf1) == 2)

    # vertex_edges
    ve = mesh.vertex_edges(v1)
    MINI_CHECK(len(ve) == 3)
    MINI_CHECK((v1, v0) in ve)
    MINI_CHECK((v1, v2) in ve)
    MINI_CHECK((v1, v4) in ve)

    # face_edges
    fe = mesh.face_edges(f0)
    MINI_CHECK(len(fe) == 4)
    MINI_CHECK(fe[0] == (v0, v1))
    MINI_CHECK(fe[1] == (v1, v2))
    MINI_CHECK(fe[2] == (v2, v3))
    MINI_CHECK(fe[3] == (v3, v0))

    # face_neighbors
    fn0 = mesh.face_neighbors(f0)
    MINI_CHECK(len(fn0) == 1)
    MINI_CHECK(fn0[0] == f1)

    # edge_vertices
    ev = mesh.edge_vertices(v0, v1)
    MINI_CHECK(ev[0] == v0 and ev[1] == v1)

    # edge_faces
    ef_inner = mesh.edge_faces(v1, v2)
    MINI_CHECK(ef_inner[0] is not None and ef_inner[1] is not None)
    ef_boundary = mesh.edge_faces(v0, v1)
    MINI_CHECK((ef_boundary[0] is not None) != (ef_boundary[1] is not None))

    # edge_edges
    ee = mesh.edge_edges(v1, v2)
    MINI_CHECK(len(ee) == 4)
    MINI_CHECK((v1, v2) not in ee)
    MINI_CHECK((v2, v1) not in ee)


@MINI_TEST("Mesh", "Geometric Properties")
def test_mesh_geometric_properties():
    from session_py import Mesh
    from session_py import Point
    from session_py import NormalWeighting

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(-1.0, 0.0, 0.0))
    v3 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    f0 = mesh.add_face([v0, v1, v3])
    f1 = mesh.add_face([v0, v3, v2])

    # face_normal
    fn = mesh.face_normal(f0)
    MINI_CHECK(fn is not None)
    MINI_CHECK(TOLERANCE.is_close(fn[2], 1.0))

    # vertex_normal
    vn = mesh.vertex_normal(v0)
    MINI_CHECK(vn is not None)
    MINI_CHECK(abs(vn[2]) == 1.0)

    # vertex_normal_weighted
    vnw = mesh.vertex_normal_weighted(v0, NormalWeighting.ANGLE)
    MINI_CHECK(vnw is not None)
    MINI_CHECK(TOLERANCE.is_close(vnw[2], 1.0))

    # face_area
    area = mesh.face_area(f0)
    MINI_CHECK(area is not None)
    MINI_CHECK(TOLERANCE.is_close(area, 0.5))

    # vertex_angle_in_face
    angle = mesh.vertex_angle_in_face(v0, f0)
    MINI_CHECK(angle is not None)
    MINI_CHECK(TOLERANCE.is_close(angle, TOLERANCE.PI / 2.0))
    MINI_CHECK(mesh.vertex_angle_in_face(999, f0) is None)

    # dihedral_angle — interior edge v0-v3 shared by f0 and f1 (coplanar = PI)
    da = mesh.dihedral_angle(v3, v0)
    MINI_CHECK(da is not None)
    MINI_CHECK(TOLERANCE.is_close(da, TOLERANCE.PI))
    # boundary edge — only one face
    MINI_CHECK(mesh.dihedral_angle(v0, v1) is None)

    # face_normals
    fns = mesh.face_normals()
    MINI_CHECK(len(fns) == 2)
    MINI_CHECK(TOLERANCE.is_close(fns[f0][2], 1.0))

    # vertex_normals
    vns = mesh.vertex_normals()
    MINI_CHECK(len(vns) == mesh.number_of_vertices())
    MINI_CHECK(TOLERANCE.is_close(vns[v0][2], 1.0))

    # vertex_normals_weighted
    vnsw = mesh.vertex_normals_weighted(NormalWeighting.ANGLE)
    MINI_CHECK(len(vnsw) == mesh.number_of_vertices())
    MINI_CHECK(TOLERANCE.is_close(vnsw[v0][2], 1.0))

    # area
    box = Mesh.create_box(2.0, 2.0, 2.0)
    MINI_CHECK(TOLERANCE.is_close(box.area(), 24.0))

    # volume
    MINI_CHECK(TOLERANCE.is_close(box.volume(), 8.0))


@MINI_TEST("Mesh", "Transformation")
def test_mesh_transformation():
    from session_py import Mesh
    from session_py import Point
    from session_py import Xform

    mesh = Mesh()
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2])

    # transform() — apply stored xform in-place; xform field unchanged
    mesh1 = mesh.duplicate()
    mesh1.xform = Xform.translation(0.0, 0.0, 1.0)
    mesh1.transform()
    MINI_CHECK(not mesh1.xform.is_identity())
    MINI_CHECK(mesh1.vertex_position(v0)[2] == 1.0)

    # transform(xf) — apply given xform in-place; stored xform unchanged
    mesh2 = mesh.duplicate()
    x = Xform.translation(0.0, 0.0, 1.0)
    mesh2.transform(x)
    MINI_CHECK(mesh2.xform.is_identity())
    MINI_CHECK(mesh2.vertex_position(v0)[2] == 1.0)

    # transformed() — copy with stored xform applied
    mesh3 = mesh.duplicate()
    mesh3.xform = Xform.translation(0.0, 0.0, 10.0)
    mesh3t = mesh3.transformed()
    MINI_CHECK(not mesh3t.xform.is_identity())
    MINI_CHECK(mesh3t.vertex_position(v0)[2] == 10.0)

    # transformed(xf) — copy with given xform applied
    mesh4 = mesh.duplicate()
    x = Xform.translation(0.0, 0.0, 10.0)
    mesh4t = mesh4.transformed(x)
    MINI_CHECK(mesh4t.xform.is_identity())
    MINI_CHECK(mesh4t.vertex_position(v0)[2] == 10.0)


@MINI_TEST("Mesh", "Json Roundtrip")
def test_mesh_json_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    mesh = Mesh()
    mesh.name = "test_mesh"
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2])

    # JSON object
    from session_py import Xform
    mesh.xform = Xform.translation(1.0, 2.0, 3.0)
    d = mesh.__jsondump__()
    loaded_json = Mesh.__jsonload__(d)
    MINI_CHECK(loaded_json.name == mesh.name)
    MINI_CHECK(loaded_json.number_of_vertices() == mesh.number_of_vertices())
    MINI_CHECK(loaded_json.number_of_faces() == mesh.number_of_faces())
    MINI_CHECK(loaded_json.xform == mesh.xform)

    # String
    json_string = mesh.json_dumps()
    loaded_string = Mesh.json_loads(json_string)
    MINI_CHECK(loaded_string.name == mesh.name)
    MINI_CHECK(loaded_string.number_of_vertices() == mesh.number_of_vertices())

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.json"
    mesh.json_dump(filename)
    loaded_file = Mesh.json_load(filename)
    MINI_CHECK(loaded_file.name == mesh.name)
    MINI_CHECK(loaded_file.number_of_vertices() == mesh.number_of_vertices())
    MINI_CHECK(loaded_file.number_of_faces() == mesh.number_of_faces())

    # Triangulation roundtrip
    pmesh = Mesh.from_polylines([[Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)]])
    MINI_CHECK(len(pmesh.triangulation) > 0)
    loaded_tri = Mesh.__jsonload__(pmesh.__jsondump__())
    fk = sorted(pmesh.triangulation.keys())[0]
    MINI_CHECK(len(loaded_tri.triangulation) > 0)
    MINI_CHECK(fk in loaded_tri.triangulation)

    # Face holes roundtrip
    hmesh = Mesh.from_polygon_with_holes([
        [Point(0,0,0), Point(4,0,0), Point(4,4,0), Point(0,4,0)],
        [Point(1,1,0), Point(3,1,0), Point(3,3,0), Point(1,3,0)]], True)
    MINI_CHECK(len(hmesh.face_holes) > 0)
    loaded_holes = Mesh.__jsonload__(hmesh.__jsondump__())
    hfk = sorted(hmesh.face_holes.keys())[0]
    MINI_CHECK(len(loaded_holes.face_holes) > 0)
    MINI_CHECK(loaded_holes.face_holes[hfk] == hmesh.face_holes[hfk])


@MINI_TEST("Mesh", "Protobuf Roundtrip")
def test_mesh_protobuf_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    mesh = Mesh()
    mesh.name = "test_mesh_proto"
    v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0))
    v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0))
    v2 = mesh.add_vertex(Point(0.0, 1.0, 0.0))
    mesh.add_face([v0, v1, v2])

    # String
    proto_bytes = mesh.pb_dumps()
    loaded_string = Mesh.pb_loads(proto_bytes)
    MINI_CHECK(loaded_string.name == mesh.name)
    MINI_CHECK(loaded_string.number_of_vertices() == mesh.number_of_vertices())

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.bin"
    mesh.pb_dump(filename)
    loaded_file = Mesh.pb_load(filename)
    MINI_CHECK(loaded_file.name == mesh.name)
    MINI_CHECK(loaded_file.number_of_vertices() == mesh.number_of_vertices())
    MINI_CHECK(loaded_file.number_of_faces() == mesh.number_of_faces())
    MINI_CHECK(loaded_file.guid == mesh.guid)

    # Triangulation roundtrip
    pmesh = Mesh.from_polylines([[Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)]])
    MINI_CHECK(len(pmesh.triangulation) > 0)
    loaded_tri = Mesh.pb_loads(pmesh.pb_dumps())
    fk = sorted(pmesh.triangulation.keys())[0]
    MINI_CHECK(len(loaded_tri.triangulation) > 0)
    MINI_CHECK(fk in loaded_tri.triangulation)

    # Face holes roundtrip
    hmesh = Mesh.from_polygon_with_holes([
        [Point(0,0,0), Point(4,0,0), Point(4,4,0), Point(0,4,0)],
        [Point(1,1,0), Point(3,1,0), Point(3,3,0), Point(1,3,0)]], True)
    MINI_CHECK(len(hmesh.face_holes) > 0)
    loaded_holes = Mesh.pb_loads(hmesh.pb_dumps())
    hfk = sorted(hmesh.face_holes.keys())[0]
    MINI_CHECK(len(loaded_holes.face_holes) > 0)
    MINI_CHECK(loaded_holes.face_holes[hfk] == hmesh.face_holes[hfk])


if __name__ == "__main__":
    run_all(language="python")
