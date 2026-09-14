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
    import copy

    vertices = Polyline.from_sides(6, 1.0, False).get_points()
    mesh = Mesh.from_vertices_and_faces(vertices, [[0, 1, 2, 3, 4, 5]])
    sstr = str(mesh)
    srepr = repr(mesh)
    mcopy = copy.deepcopy(mesh)
    MINI_CHECK(mesh.is_valid())
    mesh.name = "hexagon"

    palette = Color.palette()

    mesh.set_objectcolor(Color.grey())
    MINI_CHECK(mesh.color_mode == ColorMode.OBJECTCOLOR)

    pc = []
    for i in range(mesh.number_of_vertices()):
        pc.append(palette[i % len(palette)])
    mesh.set_pointcolors(pc)
    MINI_CHECK(mesh.color_mode == ColorMode.POINTCOLORS)
    MINI_CHECK(len(mesh.get_pointcolors()) == mesh.number_of_vertices())

    fc = []
    for i in range(mesh.number_of_faces()):
        fc.append(palette[i % len(palette)])
    mesh.set_facecolors(fc)
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    MINI_CHECK(len(mesh.get_facecolors()) == mesh.number_of_faces())

    lc = []
    lw = [0.1] * mesh.number_of_edges()
    for i in range(mesh.number_of_edges()):
        lc.append(palette[i % len(palette)])
    mesh.set_linecolors(lc, lw)
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    MINI_CHECK(len(mesh.get_linecolors()) == mesh.number_of_edges())

    mesh.color_mode = ColorMode.FACECOLORS
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    mesh.clear_facecolors()
    MINI_CHECK(mesh.color_mode == ColorMode.OBJECTCOLOR)
    MINI_CHECK(len(mesh.get_facecolors()) == 0)

    mesh.color_mode = ColorMode.FACECOLORS
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)
    mesh.clear_pointcolors()
    MINI_CHECK(mesh.color_mode == ColorMode.FACECOLORS)

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
        [
            Point(1, 1, 0),
            Point(3, 1, 0),
            Point(3, 3, 0),
            Point(1, 3, 0),
        ],
        [
            Point(0, 0, 0),
            Point(4, 0, 0),
            Point(4, 4, 0),
            Point(0, 4, 0),
        ],
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


@MINI_TEST("Mesh", "Loft Concave With Holes")
def test_mesh_loft_concave_with_holes_and_collinear():
    from session_py import Mesh
    from session_py import Point
    from session_py import Polyline

    annen_bot = [
        Polyline([
            Point(2142.008, -530.170, 1172.487),
            Point(2142.008, -530.170, -318.768),
            Point(2142.008, -318.102, -318.768),
            Point(2142.008, -347.792, -414.110),
            Point(2142.008, -106.034, -414.110),
            Point(2142.008, -135.724, -318.768),
            Point(2142.008,  106.034, -318.768),
            Point(2142.008,   76.344, -414.110),
            Point(2142.008,  318.102, -414.110),
            Point(2142.008,  288.412, -318.768),
            Point(2142.008,  530.170, -318.768),
            Point(2142.008,  530.170, 1172.487),
            Point(2142.008, -530.170, 1172.487),
        ]),
        Polyline([
            Point(2142.008, 97.448,  841.097),
            Point(2142.008,  0.000,  841.097),
            Point(2142.008,  0.000, 1006.792),
            Point(2142.008, 97.448, 1006.792),
            Point(2142.008, 97.448,  841.097),
        ]),
        Polyline([
            Point(2142.008, 97.448, 178.317),
            Point(2142.008,  0.000, 178.317),
            Point(2142.008,  0.000, 344.012),
            Point(2142.008, 97.448, 344.012),
            Point(2142.008, 97.448, 178.317),
        ]),
    ]
    annen_top = [
        Polyline([
            Point(2223.416, -530.170, 1172.487),
            Point(2223.416, -530.170, -269.141),
            Point(2223.416, -318.102, -269.141),
            Point(2223.416, -347.792, -364.483),
            Point(2223.416, -106.034, -364.483),
            Point(2223.416, -135.724, -269.141),
            Point(2223.416,  106.034, -269.141),
            Point(2223.416,   76.344, -364.483),
            Point(2223.416,  318.102, -364.483),
            Point(2223.416,  288.412, -269.141),
            Point(2223.416,  530.170, -269.141),
            Point(2223.416,  530.170, 1172.487),
            Point(2223.416, -530.170, 1172.487),
        ]),
        Polyline([
            Point(2223.416, 97.448,  841.097),
            Point(2223.416,  0.000,  841.097),
            Point(2223.416,  0.000, 1006.792),
            Point(2223.416, 97.448, 1006.792),
            Point(2223.416, 97.448,  841.097),
        ]),
        Polyline([
            Point(2223.416, 97.448, 178.317),
            Point(2223.416,  0.000, 178.317),
            Point(2223.416,  0.000, 344.012),
            Point(2223.416, 97.448, 344.012),
            Point(2223.416, 97.448, 178.317),
        ]),
    ]
    annen = Mesh.loft(annen_bot, annen_top, True)
    MINI_CHECK(annen.is_valid())
    MINI_CHECK(annen.is_closed())
    MINI_CHECK(len(annen.vertex) == 40)
    MINI_CHECK(len(annen.face) == 22)

    col_bot = [
        Polyline([
            Point( 0, 0, 0),
            Point( 4, 0, 0),
            Point( 7, 0, 0),
            Point(12, 0, 0),
            Point(12, 5, 0),
            Point( 0, 5, 0),
            Point( 0, 0, 0),
        ]),
    ]
    col_top = [
        Polyline([
            Point( 0, 0, 1.5),
            Point( 4, 0, 1.5),
            Point( 7, 0, 1.5),
            Point(12, 0, 1.5),
            Point(12, 5, 1.5),
            Point( 0, 5, 1.5),
            Point( 0, 0, 1.5),
        ]),
    ]
    colmesh = Mesh.loft(col_bot, col_top, True)
    MINI_CHECK(colmesh.is_valid())
    MINI_CHECK(colmesh.is_closed())
    MINI_CHECK(len(colmesh.vertex) == 12)
    MINI_CHECK(len(colmesh.face) == 8)


@MINI_TEST("Mesh", "From Polygon With Holes Many")
def test_mesh_from_polygon_with_holes_many():
    from session_py import Mesh
    from session_py import Point

    inputs = []
    for i in range(4):
        x = i * 7.0
        inputs.append([
            [
                Point(x, 0, 0),
                Point(x+5, 0, 0),
                Point(x+5, 5, 0),
                Point(x, 5, 0),
            ],
            [
                Point(x+1, 1, 0),
                Point(x+4, 1, 0),
                Point(x+4, 4, 0),
                Point(x+1, 4, 0),
            ],
        ])
    meshes = Mesh.from_polygon_with_holes_many(inputs)

    MINI_CHECK(meshes[0].is_valid())
    MINI_CHECK(meshes[1].is_valid())
    MINI_CHECK(meshes[2].is_valid())
    MINI_CHECK(meshes[3].is_valid())
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
            Point(x, 0, 0),
            Point(x+1, 0, 0),
            Point(x+1, 1, 0),
            Point(x, 1, 0),
            Point(x, 0, 0),
        ])
        t = Polyline([
            Point(x, 0, 1+i*0.5),
            Point(x+1, 0, 1+i*0.5),
            Point(x+1, 1, 1+i*0.5),
            Point(x, 1, 1+i*0.5),
            Point(x, 0, 1+i*0.5),
        ])
        loft_inputs.append(([b], [t]))
    meshes = Mesh.loft_many(loft_inputs)

    MINI_CHECK(meshes[0].is_valid())
    MINI_CHECK(meshes[0].is_closed())
    MINI_CHECK(meshes[1].is_valid())
    MINI_CHECK(meshes[1].is_closed())
    MINI_CHECK(meshes[2].is_valid())
    MINI_CHECK(meshes[2].is_closed())
    MINI_CHECK(meshes[3].is_valid())
    MINI_CHECK(meshes[3].is_closed())
    MINI_CHECK(meshes[4].is_valid())
    MINI_CHECK(meshes[4].is_closed())
    MINI_CHECK(meshes[5].is_valid())
    MINI_CHECK(meshes[5].is_closed())
    meshes_seq = Mesh.loft_many(loft_inputs, True, False)

    MINI_CHECK(meshes_seq[0].is_valid())
    MINI_CHECK(meshes_seq[0].is_closed())
    MINI_CHECK(meshes_seq[1].is_valid())
    MINI_CHECK(meshes_seq[1].is_closed())
    MINI_CHECK(meshes_seq[2].is_valid())
    MINI_CHECK(meshes_seq[2].is_closed())
    MINI_CHECK(meshes_seq[3].is_valid())
    MINI_CHECK(meshes_seq[3].is_closed())
    MINI_CHECK(meshes_seq[4].is_valid())
    MINI_CHECK(meshes_seq[4].is_closed())
    MINI_CHECK(meshes_seq[5].is_valid())
    MINI_CHECK(meshes_seq[5].is_closed())


@MINI_TEST("Mesh", "Loft With Quads And Triangles")
def test_mesh_loft_panels():
    from session_py import Mesh
    from session_py import Point
    from session_py import Color
    from session_py.mesh import LoftFaceRole

    top7 = [
        [
            Point(250, -250, 500),
            Point(250, 250, 500),
            Point(-250, 250, 500),
            Point(-250, -250, 500),
            Point(250, -250, 500),
        ],
        [
            Point(-250, 500, 250),
            Point(-250, 250, 500),
            Point(250, 250, 500),
            Point(250, 500, 250),
            Point(-250, 500, 250),
        ],
        [
            Point(250, -250, 500),
            Point(500, -250, 250),
            Point(500, 250, 250),
            Point(250, 250, 500),
            Point(250, -250, 500),
        ],
        [
            Point(250, 500, 250),
            Point(250, 250, 500),
            Point(500, 250, 250),
            Point(250, 500, 250),
        ],
        [
            Point(-250, 500, 250),
            Point(250, 500, 250),
            Point(250, 500, -250),
            Point(-250, 500, -250),
            Point(-250, 500, 250),
        ],
        [
            Point(250, 500, 250),
            Point(500, 250, 250),
            Point(500, 250, -250),
            Point(250, 500, -250),
            Point(250, 500, 250),
        ],
        [
            Point(500, -250, 250),
            Point(500, -250, -250),
            Point(500, 250, -250),
            Point(500, 250, 250),
            Point(500, -250, 250),
        ],
    ]
    bot7 = [
        [
            Point(270.710678, -250, 550),
            Point(270.710678, 265.891862, 550),
            Point(265.891862, 270.710678, 550),
            Point(-250, 270.710678, 550),
            Point(-250, -250, 550),
            Point(270.710678, -250, 550),
        ],
        [
            Point(270.710678, -250, 550),
            Point(550, -250, 270.710678),
            Point(550, 265.891862, 270.710678),
            Point(270.710678, 265.891862, 550),
            Point(270.710678, -250, 550),
        ],
        [
            Point(-250, 550, 270.710678),
            Point(-250, 270.710678, 550),
            Point(265.891862, 270.710678, 550),
            Point(265.891862, 550, 270.710678),
            Point(-250, 550, 270.710678),
        ],
        [
            Point(265.891862, 550, 270.710678),
            Point(265.891862, 270.710678, 550),
            Point(270.710678, 265.891862, 550),
            Point(550, 265.891862, 270.710678),
            Point(550, 270.710678, 265.891862),
            Point(270.710678, 550, 265.891862),
            Point(265.891862, 550, 270.710678),
        ],
        [
            Point(-250, 550, 270.710678),
            Point(265.891862, 550, 270.710678),
            Point(270.710678, 550, 265.891862),
            Point(270.710678, 550, -250),
            Point(-250, 550, -250),
            Point(-250, 550, 270.710678),
        ],
        [
            Point(270.710678, 550, 265.891862),
            Point(550, 270.710678, 265.891862),
            Point(550, 270.710678, -250),
            Point(270.710678, 550, -250),
            Point(270.710678, 550, 265.891862),
        ],
        [
            Point(550, -250, 270.710678),
            Point(550, -250, -250),
            Point(550, 270.710678, -250),
            Point(550, 270.710678, 265.891862),
            Point(550, 265.891862, 270.710678),
            Point(550, -250, 270.710678),
        ],
    ]
    panels, adj, top_mesh, bot_mesh = Mesh.loft_panels(top7, bot7)

    for i in range(len(panels)):
        face_colors = []
        for role in panels[i].face_roles.values():
            if role == LoftFaceRole.TopCap:
                face_colors.append(Color.blue())
            elif role == LoftFaceRole.BotCap:
                face_colors.append(Color.red())
            elif role == LoftFaceRole.TriWall:
                face_colors.append(Color.yellow())
            else:
                face_colors.append(Color.grey())
        panels[i].mesh.set_facecolors(face_colors)

    for i in range(len(panels)):
        c = panels[i].mesh.centroid()
        c.name = f"p{i}"

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

    pts, fidx = mesh.to_vertices_and_faces()
    MINI_CHECK(len(fidx) == n_faces)
    MINI_CHECK(len(pts) == n_vertices)
    MINI_CHECK(TOLERANCE.is_point_close(pts[0], Point(-0.5, -0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[1], Point( 0.5, -0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[2], Point( 0.5,  0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[3], Point(-0.5,  0.5, -0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[4], Point(-0.5, -0.5,  0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[5], Point( 0.5, -0.5,  0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[6], Point( 0.5,  0.5,  0.5)))
    MINI_CHECK(TOLERANCE.is_point_close(pts[7], Point(-0.5,  0.5,  0.5)))
    MINI_CHECK(fidx[0] == [0, 3, 2, 1])
    MINI_CHECK(fidx[1] == [4, 5, 6, 7])
    MINI_CHECK(fidx[2] == [0, 1, 5, 4])
    MINI_CHECK(fidx[3] == [2, 3, 7, 6])
    MINI_CHECK(fidx[4] == [0, 4, 7, 3])
    MINI_CHECK(fidx[5] == [1, 2, 6, 5])

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

    vertices = mesh.vertices()
    MINI_CHECK(len(vertices) == 8)
    MINI_CHECK(vertices[0] == 0)
    MINI_CHECK(vertices[1] == 1)
    MINI_CHECK(vertices[2] == 2)
    MINI_CHECK(vertices[3] == 3)
    MINI_CHECK(vertices[4] == 4)
    MINI_CHECK(vertices[5] == 5)
    MINI_CHECK(vertices[6] == 6)
    MINI_CHECK(vertices[7] == 7)
    faces = mesh.faces()
    MINI_CHECK(len(faces) == 6)
    MINI_CHECK(faces[0] == 0)
    MINI_CHECK(faces[1] == 1)
    MINI_CHECK(faces[2] == 2)
    MINI_CHECK(faces[3] == 3)
    MINI_CHECK(faces[4] == 4)
    MINI_CHECK(faces[5] == 5)
    edges = mesh.edges()
    MINI_CHECK(len(edges) == 12)
    MINI_CHECK(edges[0]  == (0, 1))
    MINI_CHECK(edges[1]  == (0, 3))
    MINI_CHECK(edges[2]  == (0, 4))
    MINI_CHECK(edges[3]  == (1, 2))
    MINI_CHECK(edges[4]  == (1, 5))
    MINI_CHECK(edges[5]  == (2, 3))
    MINI_CHECK(edges[6]  == (2, 6))
    MINI_CHECK(edges[7]  == (3, 7))
    MINI_CHECK(edges[8]  == (4, 5))
    MINI_CHECK(edges[9]  == (4, 7))
    MINI_CHECK(edges[10] == (5, 6))
    MINI_CHECK(edges[11] == (6, 7))

    MINI_CHECK(len(mesh.naked_edges(True)) == 0)
    MINI_CHECK(len(mesh.naked_faces(False)) == 6)
    mesh.remove_face(mesh.faces()[0])
    ne = mesh.naked_edges(True)
    MINI_CHECK(len(ne) == 4)
    MINI_CHECK(ne[0] == (0, 1))
    ni = mesh.naked_edges(False)
    MINI_CHECK(len(ni) == 8)
    nv = mesh.naked_vertices(True)
    MINI_CHECK(len(nv) == 4)
    nvi = mesh.naked_vertices(False)
    MINI_CHECK(len(nvi) == 4)
    nf = mesh.naked_faces(True)
    MINI_CHECK(len(nf) == 4)
    nfi = mesh.naked_faces(False)
    MINI_CHECK(len(nfi) == 1)


@MINI_TEST("Mesh", "Edges")
def test_mesh_edges():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    v0 = mesh.vertices()[0]
    v1 = mesh.vertices()[1]
    edges = mesh.edges()
    MINI_CHECK(len(edges) == 12)
    MINI_CHECK(edges[0] == (v0, v1))


@MINI_TEST("Mesh", "Create Dodecahedron")
def test_mesh_create_dodecahedron():
    from session_py import Mesh

    m = Mesh.create_dodecahedron(2.0)
    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 20)
    MINI_CHECK(m.number_of_faces() == 12)


@MINI_TEST("Mesh", "Vertex And Face Operations")
def test_mesh_vertex_and_face_operations():
    from session_py import Mesh
    from session_py import Point

    hx, hy, hz = 0.5, 0.5, 0.5
    verts = [
        Point(-hx, -hy, -hz),
        Point( hx, -hy, -hz),
        Point( hx,  hy, -hz),
        Point(-hx,  hy, -hz),
        Point(-hx, -hy,  hz),
        Point( hx, -hy,  hz),
        Point( hx,  hy,  hz),
        Point(-hx,  hy,  hz),
    ]
    faces = [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5],
    ]

    mesh = Mesh()

    for v in verts: mesh.add_vertex(v)
    for f in faces: mesh.add_face(f)

    MINI_CHECK(mesh.add_face([0, 1]) is None)
    MINI_CHECK(mesh.add_face([0, 1, 0]) is None)

    mesh.remove_vertex(0)
    MINI_CHECK(mesh.number_of_vertices() == 7)
    MINI_CHECK(mesh.number_of_faces() == 3)

    mesh.remove_edge(1, 2)
    MINI_CHECK(mesh.number_of_faces() == 2)

    mesh.remove_face(1)
    MINI_CHECK(mesh.number_of_faces() == 1)

    mesh.clear()
    MINI_CHECK(mesh.is_empty())

    for v in verts: mesh.add_vertex(v)
    for f in faces: mesh.add_face(f)

    mesh = mesh.unweld()
    MINI_CHECK(mesh.number_of_vertices() == 24)
    mesh = mesh.weld(0.001)
    MINI_CHECK(mesh.number_of_vertices() == 8)
    MINI_CHECK(mesh.number_of_faces() == 6)
    fv0 = mesh.face_vertices(0); fv1 = mesh.face_vertices(1)
    fv2 = mesh.face_vertices(2); fv3 = mesh.face_vertices(3)
    fv4 = mesh.face_vertices(4); fv5 = mesh.face_vertices(5)
    MINI_CHECK(fv0[0] == 0 and fv0[1] == 1 and fv0[2] == 2 and fv0[3] == 3)
    MINI_CHECK(fv1[0] == 4 and fv1[1] == 5 and fv1[2] == 6 and fv1[3] == 7)
    MINI_CHECK(fv2[0] == 0 and fv2[1] == 3 and fv2[2] == 5 and fv2[3] == 4)
    MINI_CHECK(fv3[0] == 2 and fv3[1] == 1 and fv3[2] == 7 and fv3[3] == 6)
    MINI_CHECK(fv4[0] == 0 and fv4[1] == 4 and fv4[2] == 7 and fv4[3] == 1)
    MINI_CHECK(fv5[0] == 3 and fv5[1] == 2 and fv5[2] == 6 and fv5[3] == 5)

    mesh.flip_face(0)
    fv0 = mesh.face_vertices(0); fv1 = mesh.face_vertices(1)
    fv2 = mesh.face_vertices(2); fv3 = mesh.face_vertices(3)
    fv4 = mesh.face_vertices(4); fv5 = mesh.face_vertices(5)
    MINI_CHECK(fv0[0] == 3 and fv0[1] == 2 and fv0[2] == 1 and fv0[3] == 0)
    MINI_CHECK(fv1[0] == 4 and fv1[1] == 5 and fv1[2] == 6 and fv1[3] == 7)
    MINI_CHECK(fv2[0] == 0 and fv2[1] == 3 and fv2[2] == 5 and fv2[3] == 4)
    MINI_CHECK(fv3[0] == 2 and fv3[1] == 1 and fv3[2] == 7 and fv3[3] == 6)
    MINI_CHECK(fv4[0] == 0 and fv4[1] == 4 and fv4[2] == 7 and fv4[3] == 1)
    MINI_CHECK(fv5[0] == 3 and fv5[1] == 2 and fv5[2] == 6 and fv5[3] == 5)

    mesh.unify_winding()
    fv0 = mesh.face_vertices(0); fv1 = mesh.face_vertices(1)
    fv2 = mesh.face_vertices(2); fv3 = mesh.face_vertices(3)
    fv4 = mesh.face_vertices(4); fv5 = mesh.face_vertices(5)
    MINI_CHECK(fv0[0] == 0 and fv0[1] == 1 and fv0[2] == 2 and fv0[3] == 3)
    MINI_CHECK(fv1[0] == 4 and fv1[1] == 5 and fv1[2] == 6 and fv1[3] == 7)
    MINI_CHECK(fv2[0] == 0 and fv2[1] == 3 and fv2[2] == 5 and fv2[3] == 4)
    MINI_CHECK(fv3[0] == 2 and fv3[1] == 1 and fv3[2] == 7 and fv3[3] == 6)
    MINI_CHECK(fv4[0] == 0 and fv4[1] == 4 and fv4[2] == 7 and fv4[3] == 1)
    MINI_CHECK(fv5[0] == 3 and fv5[1] == 2 and fv5[2] == 6 and fv5[3] == 5)

    mesh.flip()
    fv0 = mesh.face_vertices(0); fv1 = mesh.face_vertices(1)
    fv2 = mesh.face_vertices(2); fv3 = mesh.face_vertices(3)
    fv4 = mesh.face_vertices(4); fv5 = mesh.face_vertices(5)
    MINI_CHECK(fv0[0] == 3 and fv0[1] == 2 and fv0[2] == 1 and fv0[3] == 0)
    MINI_CHECK(fv1[0] == 7 and fv1[1] == 6 and fv1[2] == 5 and fv1[3] == 4)
    MINI_CHECK(fv2[0] == 4 and fv2[1] == 5 and fv2[2] == 3 and fv2[3] == 0)
    MINI_CHECK(fv3[0] == 6 and fv3[1] == 7 and fv3[2] == 1 and fv3[3] == 2)
    MINI_CHECK(fv4[0] == 1 and fv4[1] == 7 and fv4[2] == 4 and fv4[3] == 0)
    MINI_CHECK(fv5[0] == 5 and fv5[1] == 6 and fv5[2] == 2 and fv5[3] == 3)

    mesh.orient_outward()
    fv0 = mesh.face_vertices(0); fv1 = mesh.face_vertices(1)
    fv2 = mesh.face_vertices(2); fv3 = mesh.face_vertices(3)
    fv4 = mesh.face_vertices(4); fv5 = mesh.face_vertices(5)
    MINI_CHECK(fv0[0] == 0 and fv0[1] == 1 and fv0[2] == 2 and fv0[3] == 3)
    MINI_CHECK(fv1[0] == 4 and fv1[1] == 5 and fv1[2] == 6 and fv1[3] == 7)
    MINI_CHECK(fv2[0] == 0 and fv2[1] == 3 and fv2[2] == 5 and fv2[3] == 4)
    MINI_CHECK(fv3[0] == 2 and fv3[1] == 1 and fv3[2] == 7 and fv3[3] == 6)
    MINI_CHECK(fv4[0] == 0 and fv4[1] == 4 and fv4[2] == 7 and fv4[3] == 1)
    MINI_CHECK(fv5[0] == 3 and fv5[1] == 2 and fv5[2] == 6 and fv5[3] == 5)


@MINI_TEST("Mesh", "Connectivity Queries")
def test_mesh_connectivity_queries():
    from session_py import Mesh
    from session_py import Point

    pts = [
        Point(0.0, 0.0, 0.0),
        Point(1.0, 0.0, 0.0),
        Point(1.0, 1.0, 0.0),
        Point(0.0, 1.0, 0.0),
        Point(2.0, 0.0, 0.0),
    ]
    mesh = Mesh.from_vertices_and_faces(pts, [[0,1,2,3], [1,4,2]])
    v = mesh.vertices()
    f = mesh.faces()

    ee = mesh.edge_edges(1, 2)
    if ee is not None:

        u0 = ee[0][0]
        v0 = ee[0][1]
        l0 = mesh.edge_line(u0, v0)
        mid0 = l0.center()
        mid0.name = "e" + str(u0) + "-" + str(v0)

        u1 = ee[1][0]
        v1 = ee[1][1]
        l1 = mesh.edge_line(u1, v1)
        mid1 = l1.center()
        mid1.name = "e" + str(u1) + "-" + str(v1)

        u2 = ee[2][0]
        v2 = ee[2][1]
        l2 = mesh.edge_line(u2, v2)
        mid2 = l2.center()
        mid2.name = "e" + str(u2) + "-" + str(v2)

        u3 = ee[3][0]
        v3 = ee[3][1]
        l3 = mesh.edge_line(u3, v3)
        mid3 = l3.center()
        mid3.name = "e" + str(u3) + "-" + str(v3)

        MINI_CHECK(len(ee) == 4)
        MINI_CHECK(ee[0] == (1, 0))
        MINI_CHECK(ee[1] == (1, 4))
        MINI_CHECK(ee[2] == (2, 3))
        MINI_CHECK(ee[3] == (2, 4))

    ef = mesh.edge_faces(1, 2)
    if ef is not None:
        ef0 = ef[0]
        ef1 = ef[1]
        efp0 = mesh.face_centroid(ef0)
        efp0.name = "f" + str(ef0)
        efp1 = mesh.face_centroid(ef1)
        efp1.name = "f" + str(ef1)
        MINI_CHECK(len(ef) == 2)
        MINI_CHECK(ef0 == 0 and ef1 == 1)

    fe = mesh.face_edges(f[0])
    if fe is not None:
        l0 = mesh.edge_line(fe[0][0], fe[0][1])
        l1 = mesh.edge_line(fe[1][0], fe[1][1])
        l2 = mesh.edge_line(fe[2][0], fe[2][1])
        l3 = mesh.edge_line(fe[3][0], fe[3][1])
        lmid0 = l0.center()
        lmid0.name = "e" + str(fe[0][0]) + "-" + str(fe[0][1])
        lmid1 = l1.center()
        lmid1.name = "e" + str(fe[1][0]) + "-" + str(fe[1][1])
        lmid2 = l2.center()
        lmid2.name = "e" + str(fe[2][0]) + "-" + str(fe[2][1])
        lmid3 = l3.center()
        lmid3.name = "e" + str(fe[3][0]) + "-" + str(fe[3][1])
        MINI_CHECK(len(fe) == 4)
        MINI_CHECK(fe[0] == (0, 1))
        MINI_CHECK(fe[1] == (1, 2))
        MINI_CHECK(fe[2] == (2, 3))
        MINI_CHECK(fe[3] == (3, 0))

    ff = mesh.face_faces(f[0])
    if ff is not None:
        ff0 = ff[0]
        ffp = mesh.face_centroid(ff0)
        ffp.name = "f" + str(ff0)
        MINI_CHECK(len(ff) == 1)
        MINI_CHECK(ff0 == 1)

    points = mesh.face_points(f[0])
    if points is not None:
        pointcount = len(points)
        MINI_CHECK(pointcount == 4)

    pl = mesh.face_polyline(f[0])
    if pl is not None:
        pointcount = len(pl.get_points())
        MINI_CHECK(pointcount == 4)

    fv = mesh.face_vertices(f[0])
    if fv is not None:
        fv0 = fv[0]
        fv1 = fv[1]
        fv2 = fv[2]
        fv3 = fv[3]
        p0 = mesh.vertex_point(fv0)
        p0.name = str(fv0)
        p1 = mesh.vertex_point(fv1)
        p1.name = str(fv1)
        p2 = mesh.vertex_point(fv2)
        p2.name = str(fv2)
        p3 = mesh.vertex_point(fv3)
        p3.name = str(fv3)
        MINI_CHECK(fv0 == 0)
        MINI_CHECK(fv1 == 1)
        MINI_CHECK(fv2 == 2)
        MINI_CHECK(fv3 == 3)
        MINI_CHECK(len(fv) == 4)

    ve = mesh.vertex_edges(v[1])
    if ve is not None:
        vp = mesh.vertex_point(v[1])
        vp.name = "v" + str(v[1])

        l0 = mesh.edge_line(ve[0][0], ve[0][1])
        l1 = mesh.edge_line(ve[1][0], ve[1][1])
        l2 = mesh.edge_line(ve[2][0], ve[2][1])
        lmid0 = l0.center()
        lmid0.name = "e" + str(ve[0][0]) + "-" + str(ve[0][1])
        lmid1 = l1.center()
        lmid1.name = "e" + str(ve[1][0]) + "-" + str(ve[1][1])
        lmid2 = l2.center()
        lmid2.name = "e" + str(ve[2][0]) + "-" + str(ve[2][1])

        MINI_CHECK(ve[0] == (1, 0))
        MINI_CHECK(ve[1] == (1, 2))
        MINI_CHECK(ve[2] == (1, 4))
        MINI_CHECK(len(ve) == 3)

    vf = mesh.vertex_faces(v[1])
    if vf is not None:

        vp = mesh.vertex_point(v[1])
        vp.name = "v" + str(v[1])

        fp0 = mesh.face_centroid(vf[0])
        fp0.name = "f" + str(vf[0])
        fp1 = mesh.face_centroid(vf[1])
        fp1.name = "f" + str(vf[1])
        MINI_CHECK(len(vf) == 2)
        MINI_CHECK(vf[0] == 0)
        MINI_CHECK(vf[1] == 1)

    vn = mesh.vertex_vertices(v[1])
    if vn is not None:
        p0 = mesh.vertex_point(v[1])
        p0.name = "main" + str(v[1])

        np0 = mesh.vertex_point(vn[0])
        np0.name = str(vn[0])
        np1 = mesh.vertex_point(vn[1])
        np1.name = str(vn[1])
        np2 = mesh.vertex_point(vn[2])
        np2.name = str(vn[2])

        MINI_CHECK(vn[0] == 0)
        MINI_CHECK(vn[1] == 2)
        MINI_CHECK(vn[2] == 4)
        MINI_CHECK(len(vn) == 3)


@MINI_TEST("Mesh", "Geometric Properties")
def test_mesh_geometric_properties():
    from session_py import Mesh
    from session_py import Point
    from session_py import Vector
    from session_py import NormalWeighting

    mesh = Mesh.create_dodecahedron(1.5)

    area = mesh.area()
    MINI_CHECK(TOLERANCE.is_close(area, 46.4528898159021))

    centroid = mesh.centroid()
    MINI_CHECK(TOLERANCE.is_point_close(centroid, Point(0.0, 0.0, 0.0)))

    angles, arcs, points = mesh.dihedral_angles(0.3)

    for angle in angles.values():
        angle_in_degrees = angle
        MINI_CHECK(TOLERANCE.is_close(angle_in_degrees, 116.565051177078))

    for f in mesh.faces():
        face_area = mesh.face_area(f)
        MINI_CHECK(face_area is not None)
        MINI_CHECK(TOLERANCE.is_close(face_area, 3.87107415132518))

    centroids = []
    for f in mesh.faces():
        centroids.append(mesh.face_centroid(f))

    MINI_CHECK(TOLERANCE.is_point_close(centroids[0],  Point( 0.878115294937453,  0.0,               1.420820393249937)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[1],  Point( 1.420820393249937,  0.878115294937453, 0.0              )))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[2],  Point( 0.0,                1.420820393249937,  0.878115294937453)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[3],  Point( 0.878115294937453,  0.0,              -1.420820393249937)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[4],  Point( 0.0,                1.420820393249937, -0.878115294937453)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[5],  Point( 0.0,               -1.420820393249937,  0.878115294937453)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[6],  Point( 1.420820393249937, -0.878115294937453, 0.0              )))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[7],  Point( 0.0,               -1.420820393249937, -0.878115294937453)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[8],  Point(-1.420820393249937,  0.878115294937453, 0.0              )))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[9],  Point(-0.878115294937453,  0.0,               1.420820393249937)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[10], Point(-0.878115294937453,  0.0,              -1.420820393249937)))
    MINI_CHECK(TOLERANCE.is_point_close(centroids[11], Point(-1.420820393249937, -0.878115294937453, 0.0              )))

    face_normals = mesh.face_normals()
    for f in mesh.faces():
        fn = mesh.face_normal(f)
        MINI_CHECK(fn is not None)
        MINI_CHECK(TOLERANCE.is_vector_close(face_normals[f], fn))

    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[0],  Vector( 0.5257311121191336,  0.0,                 0.8506508083520400)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[1],  Vector( 0.8506508083520400,  0.5257311121191336,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[2],  Vector( 0.0,                 0.8506508083520400,  0.5257311121191336)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[3],  Vector( 0.5257311121191336,  0.0,                -0.8506508083520400)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[4],  Vector( 0.0,                 0.8506508083520400, -0.5257311121191336)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[5],  Vector( 0.0,                -0.8506508083520400,  0.5257311121191336)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[6],  Vector( 0.8506508083520400, -0.5257311121191336,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[7],  Vector( 0.0,                -0.8506508083520400, -0.5257311121191336)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[8],  Vector(-0.8506508083520400,  0.5257311121191336,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[9],  Vector(-0.5257311121191336,  0.0,                 0.8506508083520400)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[10], Vector(-0.5257311121191336,  0.0,                -0.8506508083520400)))
    MINI_CHECK(TOLERANCE.is_vector_close(face_normals[11], Vector(-0.8506508083520400, -0.5257311121191336,  0.0               )))

    for f in mesh.faces():
        fv = mesh.face_vertices(f)
        for v in fv:
            angle = mesh.vertex_angle_in_face(v, f)
            MINI_CHECK(angle is not None)
            MINI_CHECK(TOLERANCE.is_close(angle, 1.8849555921538759))

    vertex_normals = mesh.vertex_normals()
    for v in mesh.vertices():
        vn = mesh.vertex_normal(v)
        MINI_CHECK(vn is not None)
        MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[v], vn))

    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[0],  Vector( 0.5773502691896258,  0.5773502691896258,  0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[1],  Vector( 0.0,                 0.3568220897730899,  0.9341723589627158)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[2],  Vector( 0.0,                -0.3568220897730899,  0.9341723589627158)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[3],  Vector( 0.5773502691896257, -0.5773502691896258,  0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[4],  Vector( 0.9341723589627158,  0.0,                 0.3568220897730899)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[5],  Vector( 0.9341723589627158,  0.0,                -0.3568220897730899)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[6],  Vector( 0.5773502691896258,  0.5773502691896257, -0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[7],  Vector( 0.3568220897730899,  0.9341723589627158,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[8],  Vector(-0.3568220897730899,  0.9341723589627157,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[9],  Vector(-0.5773502691896258,  0.5773502691896258,  0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[10], Vector( 0.5773502691896258, -0.5773502691896258, -0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[11], Vector( 0.0,                -0.3568220897730899, -0.9341723589627157)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[12], Vector( 0.0,                 0.3568220897730899, -0.9341723589627158)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[13], Vector(-0.5773502691896257,  0.5773502691896258, -0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[14], Vector(-0.5773502691896258, -0.5773502691896257,  0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[15], Vector(-0.3568220897730899, -0.9341723589627157,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[16], Vector( 0.3568220897730899, -0.9341723589627158,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[17], Vector(-0.5773502691896258, -0.5773502691896258, -0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[18], Vector(-0.9341723589627157,  0.0,                -0.3568220897730899)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[19], Vector(-0.9341723589627158,  0.0,                 0.3568220897730899)))

    vertex_normals_weighted = mesh.vertex_normals_weighted(NormalWeighting.ANGLE)
    for v in mesh.vertices():
        vnw = mesh.vertex_normal_weighted(v, NormalWeighting.ANGLE)
        MINI_CHECK(vnw is not None)
        MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[v], vnw))

    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[0],  Vector( 0.5773502691896257,  0.5773502691896257,  0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[1],  Vector( 0.0,                 0.3568220897730899,  0.9341723589627158)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[2],  Vector( 0.0,                -0.3568220897730899,  0.9341723589627158)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[3],  Vector( 0.5773502691896257, -0.5773502691896257,  0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[4],  Vector( 0.9341723589627158,  0.0,                 0.3568220897730899)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[5],  Vector( 0.9341723589627158,  0.0,                -0.3568220897730899)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[6],  Vector( 0.5773502691896258,  0.5773502691896257, -0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[7],  Vector( 0.3568220897730899,  0.9341723589627158,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[8],  Vector(-0.3568220897730899,  0.9341723589627158,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[9],  Vector(-0.5773502691896257,  0.5773502691896258,  0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[10], Vector( 0.5773502691896257, -0.5773502691896258, -0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[11], Vector( 0.0,                -0.3568220897730899, -0.9341723589627158)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[12], Vector( 0.0,                 0.3568220897730899, -0.9341723589627158)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[13], Vector(-0.5773502691896257,  0.5773502691896257, -0.5773502691896258)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[14], Vector(-0.5773502691896258, -0.5773502691896257,  0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[15], Vector(-0.3568220897730900, -0.9341723589627158,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[16], Vector( 0.3568220897730899, -0.9341723589627158,  0.0               )))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[17], Vector(-0.5773502691896257, -0.5773502691896257, -0.5773502691896257)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[18], Vector(-0.9341723589627158,  0.0,                -0.3568220897730899)))
    MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[19], Vector(-0.9341723589627158,  0.0,                 0.3568220897730899)))

    volume = mesh.volume()
    MINI_CHECK(TOLERANCE.is_close(volume, 25.8630264921081))


@MINI_TEST("Mesh", "Transformation")
def test_mesh_transformation():
    from session_py import Mesh
    from session_py import Point
    from session_py import Xform

    pts = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(0, 1, 0),
    ]
    mesh = Mesh.from_vertices_and_faces(pts, [[0,1,2]])
    v0 = mesh.vertices()[0]

    mesh1 = mesh.duplicate()
    mesh1_xf = Xform.translation(0.0, 0.0, 1.0)
    mesh1.transform(mesh1_xf)

    MINI_CHECK(mesh1.vertex_point(v0)[2] == 1.0)

    mesh2 = mesh.duplicate()
    x = Xform.translation(0.0, 0.0, 1.0)
    mesh2.transform(x)
    MINI_CHECK(mesh2.vertex_point(v0)[2] == 1.0)

    mesh3 = mesh.duplicate()
    mesh3_xf = Xform.translation(0.0, 0.0, 10.0)
    mesh3t = mesh3.transformed(mesh3_xf)
    MINI_CHECK(mesh3t.vertex_point(v0)[2] == 10.0)

    mesh4 = mesh.duplicate()
    x = Xform.translation(0.0, 0.0, 10.0)
    mesh4t = mesh4.transformed(x)
    MINI_CHECK(mesh4t.vertex_point(v0)[2] == 10.0)


@MINI_TEST("Mesh", "Json Roundtrip")
def test_mesh_json_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.name = "test_mesh"

    d = mesh.__jsondump__()
    loaded_json = Mesh.__jsonload__(d)

    json_string = mesh.file_json_dumps()
    loaded_string = Mesh.file_json_loads(json_string)

    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.json"
    mesh.file_json_dump(filename)
    loaded_file = Mesh.file_json_load(filename)

    MINI_CHECK(loaded_json == mesh)
    MINI_CHECK(loaded_string == mesh)
    MINI_CHECK(loaded_file == mesh)

    polys = [[
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]]
    pmesh = Mesh.from_polylines(polys)
    loaded_tri = Mesh.__jsonload__(pmesh.__jsondump__())
    fk = sorted(pmesh.triangulation.keys())[0]

    MINI_CHECK(len(loaded_tri.triangulation) > 0)
    MINI_CHECK(fk in loaded_tri.triangulation)

    hmesh = Mesh.from_polygon_with_holes([
        [
            Point(0, 0, 0),
            Point(4, 0, 0),
            Point(4, 4, 0),
            Point(0, 4, 0),
        ],
        [
            Point(1, 1, 0),
            Point(3, 1, 0),
            Point(3, 3, 0),
            Point(1, 3, 0),
        ]], True)
    loaded_holes = Mesh.__jsonload__(hmesh.__jsondump__())
    hfk = sorted(hmesh.face_holes.keys())[0]

    MINI_CHECK(len(loaded_holes.face_holes) > 0)
    MINI_CHECK(loaded_holes.face_holes[hfk] == hmesh.face_holes[hfk])


@MINI_TEST("Mesh", "Protobuf Roundtrip")
def test_mesh_protobuf_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from pathlib import Path

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.name = "test_mesh_proto"

    proto_bytes = mesh.pb_dumps()
    loaded_string = Mesh.pb_loads(proto_bytes)

    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.bin"
    mesh.pb_dump(filename)
    loaded_file = Mesh.pb_load(filename)

    MINI_CHECK(loaded_string == mesh)
    MINI_CHECK(loaded_file == mesh)

    polys = [[
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
    ]]
    pmesh = Mesh.from_polylines(polys)
    loaded_tri = Mesh.pb_loads(pmesh.pb_dumps())
    fk = sorted(pmesh.triangulation.keys())[0]

    MINI_CHECK(len(loaded_tri.triangulation) > 0)
    MINI_CHECK(fk in loaded_tri.triangulation)

    hmesh = Mesh.from_polygon_with_holes([
        [
            Point(0, 0, 0),
            Point(4, 0, 0),
            Point(4, 4, 0),
            Point(0, 4, 0),
        ],
        [
            Point(1, 1, 0),
            Point(3, 1, 0),
            Point(3, 3, 0),
            Point(1, 3, 0),
        ]], True)
    loaded_holes = Mesh.pb_loads(hmesh.pb_dumps())
    hfk = sorted(hmesh.face_holes.keys())[0]

    MINI_CHECK(len(loaded_holes.face_holes) > 0)
    MINI_CHECK(loaded_holes.face_holes[hfk] == hmesh.face_holes[hfk])


@MINI_TEST("Mesh", "Loft Plate Four Holes")
def test_mesh_loft_plate_four_holes():
    from session_py import Mesh
    from session_py import Point
    from session_py import Polyline
    bot = [
        Polyline([
            Point(734.392021, -1906.59468, 1101.588031),
            Point(632.396858, -1838.597905, 948.595287),
            Point(624.453132, -1769.270846, 984.70313),
            Point(113.775484, -1428.81908, 218.686657),
            Point(121.719209, -1498.146139, 182.578814),
            Point(15.607979, -1427.40532, 23.411969),
            Point(0.0, -1441.0, -18.0),
            Point(0.0, -1893.0, -357.0),
            Point(13.416408, -1917.0, -348.167184),
            Point(104.290124, -1917.0, -166.419752),
            Point(118.441096, -1964.169906, -173.495238),
            Point(664.077103, -1964.169906, 917.776777),
            Point(649.926131, -1917.0, 924.852263),
            Point(736.583592, -1917.0, 1098.167184),
            Point(734.392021, -1906.59468, 1101.588031),
        ]),
        Polyline([
            Point(322.544527, -1917.0, 270.089054),
            Point(213.417326, -1917.0, 51.834651),
            Point(199.266354, -1869.830094, 58.910137),
            Point(308.393555, -1869.830094, 277.16454),
            Point(322.544527, -1917.0, 270.089054),
        ]),
        Polyline([
            Point(540.79893, -1917.0, 706.59786),
            Point(431.671728, -1917.0, 488.343457),
            Point(417.520757, -1869.830094, 495.418943),
            Point(526.647958, -1869.830094, 713.673346),
            Point(540.79893, -1917.0, 706.59786),
        ]),
        Polyline([
            Point(424.153936, -1667.753669, 660.242619),
            Point(526.289465, -1735.844022, 813.445914),
            Point(530.261328, -1770.507552, 795.391992),
            Point(428.125798, -1702.417199, 642.188697),
            Point(424.153936, -1667.753669, 660.242619),
        ]),
        Polyline([
            Point(219.882876, -1531.572963, 353.83603),
            Point(322.018406, -1599.663316, 507.039325),
            Point(325.990269, -1634.326846, 488.985403),
            Point(223.854739, -1566.236493, 335.782108),
            Point(219.882876, -1531.572963, 353.83603),
        ]),
    ]
    top = [
        Polyline([
            Point(711.660594, -1906.59468, 1126.880036),
            Point(605.549364, -1835.85386, 967.713191),
            Point(601.577501, -1801.190331, 985.767113),
            Point(90.899853, -1460.738565, 219.75064),
            Point(94.871715, -1495.402095, 201.696718),
            Point(-9.83197, -1425.599638, 44.641191),
            Point(-25.439949, -1439.194318, 3.229221),
            Point(-25.439949, -1893.0, -337.12504),
            Point(-12.023541, -1917.0, -328.292224),
            Point(75.988181, -1917.0, -152.26878),
            Point(104.290124, -2011.339811, -166.419752),
            Point(649.926131, -2011.339811, 924.852263),
            Point(621.624188, -1917.0, 939.003234),
            Point(713.852165, -1917.0, 1123.459189),
            Point(711.660594, -1906.59468, 1126.880036),
        ]),
        Polyline([
            Point(308.393555, -1964.169906, 277.16454),
            Point(199.266354, -1964.169906, 58.910137),
            Point(185.115382, -1917.0, 65.985623),
            Point(294.242584, -1917.0, 284.240026),
            Point(308.393555, -1964.169906, 277.16454),
        ]),
        Polyline([
            Point(526.647958, -1964.169906, 713.673346),
            Point(417.520757, -1964.169906, 495.418943),
            Point(403.369785, -1917.0, 502.494429),
            Point(512.496987, -1917.0, 720.748832),
            Point(526.647958, -1964.169906, 713.673346),
        ]),
        Polyline([
            Point(401.278305, -1699.673154, 661.306602),
            Point(503.413834, -1767.763507, 814.509897),
            Point(507.385697, -1802.427037, 796.455975),
            Point(405.250167, -1734.336684, 643.25268),
            Point(401.278305, -1699.673154, 661.306602),
        ]),
        Polyline([
            Point(197.007245, -1563.492448, 354.900013),
            Point(299.142775, -1631.582801, 508.103307),
            Point(303.114638, -1666.246331, 490.049386),
            Point(200.979108, -1598.155978, 336.846091),
            Point(197.007245, -1563.492448, 354.900013),
        ]),
    ]
    m = Mesh.loft(bot, top, True, True)
    MINI_CHECK(m.is_valid())


@MINI_TEST("Mesh", "Loft Plate V2")
def test_mesh_loft_plate_v2():
    from session_py import Mesh
    from session_py import Point
    from session_py import Polyline
    top = [
        Polyline([
            Point(734.392021, -28.40532, 1101.588031),
            Point(630.839301, -97.440466, 946.258951),
            Point(602.668732, -21.881034, 974.757956),
            Point(90.636822, -363.235641, 206.710092),
            Point(118.807391, -438.795073, 178.211087),
            Point(15.607979, -507.59468, 23.411969),
            Point(21.213203, -518.0, 21.213203),
            Point(1478.786797, -518.0, 1478.786797),
            Point(1476.953362, -502.635574, 1488.476681),
            Point(1323.309106, -400.20607, 1411.654553),
            Point(1323.309106, -350.20607, 1449.154553),
            Point(921.429178, -82.286119, 1248.214589),
            Point(921.429178, -132.286119, 1210.714589),
            Point(773.046638, -33.364426, 1136.523319),
            Point(734.392021, -28.40532, 1101.588031),
        ]),
        Polyline([
            Point(1055.389154, -196.592769, 1296.444577),
            Point(1189.34913, -285.89942, 1363.424565),
            Point(1189.34913, -310.89942, 1344.674565),
            Point(1055.389154, -221.592769, 1277.694577),
            Point(1055.389154, -196.592769, 1296.444577),
        ]),
        Polyline([
            Point(411.941252, -196.202593, 653.289308),
            Point(514.347634, -127.931671, 806.898881),
            Point(528.432919, -165.711387, 792.649378),
            Point(426.026537, -233.982309, 639.039805),
            Point(411.941252, -196.202593, 653.289308),
        ]),
        Polyline([
            Point(207.128489, -332.744435, 346.070162),
            Point(309.53487, -264.473514, 499.679735),
            Point(323.620155, -302.25323, 485.430233),
            Point(221.213773, -370.524151, 331.82066),
            Point(207.128489, -332.744435, 346.070162),
        ]),
    ]
    bot = [
        Polyline([
            Point(717.764591, -24.335988, 1136.036032),
            Point(607.106921, -98.107768, 970.049526),
            Point(593.021636, -60.328052, 984.299029),
            Point(80.989727, -401.682659, 216.251164),
            Point(95.075011, -439.462375, 202.001662),
            Point(-28.206905, -521.650319, 17.078787),
            Point(-22.601681, -532.055639, 14.880022),
            Point(1489.346823, -532.055639, 1526.828525),
            Point(1487.513388, -516.691213, 1536.51841),
            Point(1323.309106, -407.221692, 1454.416269),
            Point(1323.309106, -382.221692, 1473.166269),
            Point(921.429178, -114.30174, 1272.226305),
            Point(921.429178, -139.30174, 1253.476305),
            Point(756.419209, -29.295094, 1170.97132),
            Point(717.764591, -24.335988, 1136.036032),
        ]),
        Polyline([
            Point(1055.389154, -228.608391, 1320.456293),
            Point(1189.34913, -317.915041, 1387.436281),
            Point(1189.34913, -342.915041, 1368.686281),
            Point(1055.389154, -253.608391, 1301.706293),
            Point(1055.389154, -228.608391, 1320.456293),
        ]),
        Polyline([
            Point(402.294157, -234.649611, 662.830381),
            Point(504.700539, -166.37869, 816.439954),
            Point(518.785824, -204.158406, 802.190451),
            Point(416.379442, -272.429327, 648.580878),
            Point(402.294157, -234.649611, 662.830381),
        ]),
        Polyline([
            Point(197.481393, -371.191453, 355.611235),
            Point(299.887775, -302.920532, 509.220808),
            Point(313.97306, -340.700248, 494.971305),
            Point(211.566678, -408.971169, 341.361733),
            Point(197.481393, -371.191453, 355.611235),
        ]),
    ]
    m = Mesh.loft(top, bot, True, True)
    MINI_CHECK(m.is_valid())


@MINI_TEST("Mesh", "Loft Plate V3")
def test_mesh_loft_plate_v3():
    from session_py import Mesh
    from session_py import Point
    from session_py import Polyline
    top = [
        Polyline([
            Point(734.392021, 352.59468, 1101.588031),
            Point(618.973111, 275.648741, 928.459666),
            Point(618.973111, 369.988552, 999.214525),
            Point(106.941201, 28.633945, 231.16666),
            Point(106.941201, -65.705866, 160.411802),
            Point(15.607979, -126.59468, 23.411969),
            Point(21.213203, -137.0, 21.213203),
            Point(1478.786797, -137.0, 1478.786797),
            Point(1476.953362, -121.635574, 1488.476681),
            Point(1323.309106, -19.20607, 1411.654553),
            Point(1323.309106, 30.79393, 1449.154553),
            Point(921.429178, 298.713881, 1248.214589),
            Point(921.429178, 248.713881, 1210.714589),
            Point(773.046638, 347.635574, 1136.523319),
            Point(734.392021, 352.59468, 1101.588031),
        ]),
        Polyline([
            Point(1055.389154, 184.407231, 1296.444577),
            Point(1189.34913, 95.10058, 1363.424565),
            Point(1189.34913, 70.10058, 1344.674565),
            Point(1055.389154, 159.407231, 1277.694577),
            Point(1055.389154, 184.407231, 1296.444577),
        ]),
        Polyline([
            Point(414.160347, 186.276804, 656.61795),
            Point(516.566729, 254.547725, 810.227523),
            Point(516.566729, 207.377819, 774.850093),
            Point(414.160347, 139.106898, 621.240521),
            Point(414.160347, 186.276804, 656.61795),
        ]),
        Polyline([
            Point(209.347583, 49.734961, 349.398804),
            Point(311.753965, 118.005882, 503.008377),
            Point(311.753965, 70.835977, 467.630948),
            Point(209.347583, 2.565055, 314.021375),
            Point(209.347583, 49.734961, 349.398804),
        ]),
    ]
    bot = [
        Polyline([
            Point(717.764591, 356.664012, 1136.036032),
            Point(618.973111, 290.803025, 987.848811),
            Point(618.973111, 337.972931, 1023.226241),
            Point(106.941201, -3.381676, 255.178376),
            Point(106.941201, -50.551581, 219.800947),
            Point(-28.206905, -140.650319, 17.078787),
            Point(-22.601681, -151.055639, 14.880022),
            Point(1489.346823, -151.055639, 1526.828525),
            Point(1487.513388, -135.691213, 1536.51841),
            Point(1323.309106, -26.221692, 1454.416269),
            Point(1323.309106, -1.221692, 1473.166269),
            Point(921.429178, 266.69826, 1272.226305),
            Point(921.429178, 241.69826, 1253.476305),
            Point(756.419209, 351.704906, 1170.97132),
            Point(717.764591, 356.664012, 1136.036032),
        ]),
        Polyline([
            Point(1055.389154, 152.391609, 1320.456293),
            Point(1189.34913, 63.084959, 1387.436281),
            Point(1189.34913, 38.084959, 1368.686281),
            Point(1055.389154, 127.391609, 1301.706293),
            Point(1055.389154, 152.391609, 1320.456293),
        ]),
        Polyline([
            Point(414.160347, 154.261182, 680.629666),
            Point(516.566729, 222.532104, 834.239239),
            Point(516.566729, 175.362198, 798.861809),
            Point(414.160347, 107.091277, 645.252236),
            Point(414.160347, 154.261182, 680.629666),
        ]),
        Polyline([
            Point(209.347583, 17.71934, 373.41052),
            Point(311.753965, 85.990261, 527.020093),
            Point(311.753965, 38.820356, 491.642664),
            Point(209.347583, -29.450566, 338.033091),
            Point(209.347583, 17.71934, 373.41052),
        ]),
    ]
    m = Mesh.loft(top, bot, True, True)
    MINI_CHECK(m.is_valid())


@MINI_TEST("Mesh", "Vertex Neighbors")
def test_mesh_vertex_neighbors():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    n0 = mesh.vertex_neighbors(0)
    n0v = mesh.vertex_vertices(0)
    MINI_CHECK(n0 == n0v)
    MINI_CHECK(len(n0) == 3)


@MINI_TEST("Mesh", "Vertices On Boundary")
def test_mesh_vertices_on_boundary():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    MINI_CHECK(len(mesh.vertices_on_boundary()) == 0)
    mesh.remove_face(mesh.faces()[0])
    vb = mesh.vertices_on_boundary()
    MINI_CHECK(len(vb) == 4)


@MINI_TEST("Mesh", "Edges On Boundary")
def test_mesh_edges_on_boundary():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    MINI_CHECK(len(mesh.edges_on_boundary()) == 0)
    mesh.remove_face(mesh.faces()[0])
    eb = mesh.edges_on_boundary()
    MINI_CHECK(len(eb) == 4)


@MINI_TEST("Mesh", "Faces On Boundary")
def test_mesh_faces_on_boundary():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    MINI_CHECK(len(mesh.faces_on_boundary()) == 0)
    mesh.remove_face(mesh.faces()[0])
    MINI_CHECK(len(mesh.faces_on_boundary()) == 4)


@MINI_TEST("Mesh", "Halfedge Face")
def test_mesh_halfedge_face():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    f = mesh.halfedge_face((0, 3))
    MINI_CHECK(f is not None)
    MINI_CHECK(f == 0)
    mesh.remove_face(0)
    MINI_CHECK(mesh.halfedge_face((0, 3)) is None)


@MINI_TEST("Mesh", "Halfedge After Before")
def test_mesh_halfedge_after_before():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    after = mesh.halfedge_after((0, 3))
    before = mesh.halfedge_before((0, 3))
    MINI_CHECK(after is not None)
    MINI_CHECK(after == (3, 2))
    MINI_CHECK(before is not None)
    MINI_CHECK(before == (1, 0))


@MINI_TEST("Mesh", "Halfedge Loop")
def test_mesh_halfedge_loop():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    loop = mesh.halfedge_loop((0, 3))
    MINI_CHECK(len(loop) == 1)
    MINI_CHECK(loop[0] == (0, 3))


@MINI_TEST("Mesh", "Halfedge Strip")
def test_mesh_halfedge_strip():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    strip = mesh.halfedge_strip((0, 3))
    MINI_CHECK(len(strip) == 5)
    MINI_CHECK(strip[0] == (0, 3))
    MINI_CHECK(strip[-1] == (0, 3))


@MINI_TEST("Mesh", "Vertex Sample")
def test_mesh_vertex_sample():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    s = mesh.vertex_sample(3, seed=42)
    MINI_CHECK(len(s) == 3)
    MINI_CHECK(len(set(s)) == 3)
    s2 = mesh.vertex_sample(3, seed=42)
    MINI_CHECK(s == s2)


@MINI_TEST("Mesh", "Edge Sample")
def test_mesh_edge_sample():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    s = mesh.edge_sample(2, seed=7)
    MINI_CHECK(len(s) == 2)
    s2 = mesh.edge_sample(2, seed=7)
    MINI_CHECK(s == s2)


@MINI_TEST("Mesh", "Face Sample")
def test_mesh_face_sample():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    s = mesh.face_sample(2, seed=11)
    MINI_CHECK(len(s) == 2)
    s2 = mesh.face_sample(2, seed=11)
    MINI_CHECK(s == s2)


@MINI_TEST("Mesh", "Face Center")
def test_mesh_face_center():
    from session_py import Mesh

    mesh = Mesh.create_box(2.0, 2.0, 2.0)
    c = mesh.face_center(0)
    cc = mesh.face_centroid(0)
    MINI_CHECK(c == cc)


@MINI_TEST("Mesh", "Face Polygon")
def test_mesh_face_polygon():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    poly = mesh.face_polygon(0)
    pts = poly.get_points() if hasattr(poly, "get_points") else poly.points
    MINI_CHECK(len(pts) == 5)
    MINI_CHECK(pts[0] == pts[-1])


@MINI_TEST("Mesh", "Flip Cycles")
def test_mesh_flip_cycles():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    n0 = mesh.face_normal(0)
    mesh.flip_cycles()
    n0b = mesh.face_normal(0)
    MINI_CHECK(abs(n0[0] + n0b[0]) < TOLERANCE.ZERO_TOLERANCE)
    MINI_CHECK(abs(n0[1] + n0b[1]) < TOLERANCE.ZERO_TOLERANCE)
    MINI_CHECK(abs(n0[2] + n0b[2]) < TOLERANCE.ZERO_TOLERANCE)


@MINI_TEST("Mesh", "Face Normal Unitized")
def test_mesh_face_normal_unitized():
    from session_py import Mesh
    from session_py import TOLERANCE

    mesh = Mesh.create_box(2.0, 2.0, 2.0)
    nu = mesh.face_normal_unitized(0, True)
    nn = mesh.face_normal_unitized(0, False)
    MINI_CHECK(abs(nu.magnitude() - 1.0) < TOLERANCE.ZERO_TOLERANCE)
    MINI_CHECK(nn.magnitude() > 1.0)


@MINI_TEST("Mesh", "Default Attributes")
def test_mesh_default_attributes():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_vertex_attributes({"is_support": 0.0, "load_z": 0.0})
    mesh.update_default_face_attributes({"stress": 0.0})
    mesh.update_default_edge_attributes({"weight": 1.0})
    MINI_CHECK(mesh.default_vertex_attributes["is_support"] == 0.0)
    MINI_CHECK(mesh.default_vertex_attributes["load_z"] == 0.0)
    MINI_CHECK(mesh.default_face_attributes["stress"] == 0.0)
    MINI_CHECK(mesh.default_edge_attributes["weight"] == 1.0)


@MINI_TEST("Mesh", "Vertex Attribute")
def test_mesh_vertex_attribute():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_vertex_attributes({"is_support": 0.0})
    mesh.set_vertex_attribute(0, "is_support", 1.0)
    MINI_CHECK(mesh.vertex_attribute(0, "is_support") == 1.0)
    MINI_CHECK(mesh.vertex_attribute(1, "is_support") == 0.0)


@MINI_TEST("Mesh", "Face Attribute")
def test_mesh_face_attribute():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_face_attributes({"stress": 0.0})
    mesh.set_face_attribute(0, "stress", 2.5)
    MINI_CHECK(mesh.face_attribute(0, "stress") == 2.5)
    MINI_CHECK(mesh.face_attribute(1, "stress") == 0.0)


@MINI_TEST("Mesh", "Edge Attribute")
def test_mesh_edge_attribute():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_edge_attributes({"weight": 1.0})
    mesh.set_edge_attribute((0, 1), "weight", 5.0)
    MINI_CHECK(mesh.edge_attribute((0, 1), "weight") == 5.0)
    MINI_CHECK(mesh.edge_attribute((0, 3), "weight") == 1.0)


@MINI_TEST("Mesh", "Vertices Attribute Bulk")
def test_mesh_vertices_attribute_bulk():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_vertex_attributes({"is_support": 0.0})
    keys = [0, 1, 2]
    mesh.set_vertices_attribute("is_support", 1.0, keys)
    vals = mesh.vertices_attribute("is_support")
    MINI_CHECK(vals[0] == 1.0)
    MINI_CHECK(vals[1] == 1.0)
    MINI_CHECK(vals[2] == 1.0)
    MINI_CHECK(vals[3] == 0.0)


@MINI_TEST("Mesh", "Vertices Where")
def test_mesh_vertices_where():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_vertex_attributes({"is_support": 0.0})
    keys = [0, 2, 4]
    mesh.set_vertices_attribute("is_support", 1.0, keys)
    sup = mesh.vertices_where({"is_support": 1.0})
    sup.sort()
    MINI_CHECK(len(sup) == 3)
    MINI_CHECK(sup == [0, 2, 4])


@MINI_TEST("Mesh", "Faces Where")
def test_mesh_faces_where():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_face_attributes({"tag": 0.0})
    mesh.set_face_attribute(2, "tag", 7.0)
    mesh.set_face_attribute(4, "tag", 7.0)
    out = mesh.faces_where({"tag": 7.0})
    out.sort()
    MINI_CHECK(out == [2, 4])


@MINI_TEST("Mesh", "Edges Where")
def test_mesh_edges_where():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_edge_attributes({"weight": 0.0})
    mesh.set_edge_attribute((0, 1), "weight", 3.0)
    out = mesh.edges_where({"weight": 3.0})
    MINI_CHECK(len(out) == 1)
    MINI_CHECK(out[0] == (0, 1))


@MINI_TEST("Mesh", "Vertices Where Predicate")
def test_mesh_vertices_where_predicate():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_vertex_attributes({"load": 0.0})
    mesh.set_vertex_attribute(0, "load", 5.0)
    mesh.set_vertex_attribute(1, "load", 10.0)
    big = mesh.vertices_where_predicate(lambda k, a: "load" in a and a["load"] > 4.0)
    big.sort()
    MINI_CHECK(big == [0, 1])


@MINI_TEST("Mesh", "Faces Where Predicate")
def test_mesh_faces_where_predicate():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_face_attributes({"area": 0.0})
    mesh.set_face_attribute(0, "area", 2.0)
    mesh.set_face_attribute(3, "area", 4.0)
    big = mesh.faces_where_predicate(lambda k, a: "area" in a and a["area"] > 1.0)
    big.sort()
    MINI_CHECK(big == [0, 3])


@MINI_TEST("Mesh", "Edges Where Predicate")
def test_mesh_edges_where_predicate():
    from session_py import Mesh

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.update_default_edge_attributes({"weight": 0.0})
    mesh.set_edge_attribute((0, 1), "weight", 5.0)
    big = mesh.edges_where_predicate(lambda e, a: "weight" in a and a["weight"] > 1.0)
    MINI_CHECK(len(big) == 1)
    MINI_CHECK(big[0] == (0, 1))


@MINI_TEST("Mesh", "Refresh Guid")
def test_mesh_refresh_guid():
    from session_py import Mesh
    import copy as copy_module
    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    original = mesh.guid
    copy = copy_module.deepcopy(mesh)

    MINI_CHECK(copy.guid == original)
    copy.refresh_guid()
    MINI_CHECK(copy.guid != original)
    MINI_CHECK(mesh.guid == original)


@MINI_TEST("Mesh", "Assignment Keeps Objectcolor")
def test_mesh_assignment_keeps_objectcolor():
    from session_py import Color
    from session_py import Mesh
    from session_py import Point
    source = Mesh.from_vertices_and_faces(
        [Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)],
        [[0, 1, 2, 3]])
    source.set_objectcolor(Color(0.72, 0.72, 0.74, 1.0, "grey"))

    target = source.duplicate()
    MINI_CHECK(target.get_objectcolor().r == source.get_objectcolor().r)
    MINI_CHECK(target.get_objectcolor().g == source.get_objectcolor().g)
    MINI_CHECK(target.get_objectcolor().b == source.get_objectcolor().b)
    MINI_CHECK(target.color_mode == source.color_mode)


if __name__ == "__main__":
    run_all(language="python")
