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


@MINI_TEST("Mesh", "Loft concave with holes and collinear")
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
    MINI_CHECK(len(colmesh.vertex) == 8)
    MINI_CHECK(len(colmesh.face) == 6)


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


@MINI_TEST("Mesh", "Loft with quads and triangles")
def test_mesh_loft_panels():
    from session_py import Mesh
    from session_py import Point
    from session_py import Color

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
    panels, adj, top_mesh, bot_mesh = Mesh.loft_panels(top7, bot7, 0.001)

    # Color faces: blue=top cap, red=bot cap, gray=quad wall, yellow=tri wall
    for i, panel in enumerate(panels):
        face_colors = []
        for fk, role in panel.face_roles.items():
            if role == "TopCap":
                face_colors.append(Color.blue())
            elif role == "BotCap":
                face_colors.append(Color.red())
            elif role == "TriWall":
                face_colors.append(Color.yellow())
            else:
                face_colors.append(Color.grey())
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

    # vertices / faces / edges
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

    # naked (closed box: no naked edges before removal)
    MINI_CHECK(len(mesh.naked_edges(True)) == 0)
    MINI_CHECK(len(mesh.naked_faces(False)) == 6)
    # remove one face — box becomes open, check naked
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
    MINI_CHECK(isinstance(edges[0], tuple))
    MINI_CHECK(edges[0] == (v0, v1))


@MINI_TEST("Mesh", "Create Dodecahedron")
def test_mesh_create_dodecahedron():
    from session_py import Mesh

    m = Mesh.create_dodecahedron(2.0)
    MINI_CHECK(m.is_valid())
    MINI_CHECK(m.number_of_vertices() == 20)
    MINI_CHECK(m.number_of_faces() == 12)


@MINI_TEST("Mesh", "Vertex and Face Operations")
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

    # remove_vertex(0): removes vertex 0 + 3 adjacent faces (0,2,4)
    # vertices → [1,2,3,4,5,6,7], faces → [1,3,5]
    mesh.remove_vertex(0)
    MINI_CHECK(mesh.number_of_vertices() == 7)
    MINI_CHECK(mesh.number_of_faces() == 3)

    # remove_edge(1,2): removes face 5 [1,2,6,5], faces → [1,3]
    mesh.remove_edge(1, 2)
    MINI_CHECK(mesh.number_of_faces() == 2)

    # remove_face(1): removes face 1 [4,5,6,7], faces → [3]
    mesh.remove_face(1)
    MINI_CHECK(mesh.number_of_faces() == 1)

    # clear
    mesh.clear()
    MINI_CHECK(mesh.is_empty())

    # rebuild
    for v in verts: mesh.add_vertex(v)
    for f in faces: mesh.add_face(f)

    # unweld and weld
    mesh = mesh.unweld()
    MINI_CHECK(mesh.number_of_vertices() == 24)
    mesh = mesh.weld(0.001)
    MINI_CHECK(mesh.number_of_vertices() == 8)
    MINI_CHECK(mesh.number_of_faces() == 6)
    # face 0: 0 1 2 3, face 1: 4 5 6 7, face 2: 0 3 5 4
    # face 3: 2 1 7 6, face 4: 0 4 7 1, face 5: 3 2 6 5
    fv0 = mesh.face_vertices(0); fv1 = mesh.face_vertices(1)
    fv2 = mesh.face_vertices(2); fv3 = mesh.face_vertices(3)
    fv4 = mesh.face_vertices(4); fv5 = mesh.face_vertices(5)
    MINI_CHECK(fv0[0] == 0 and fv0[1] == 1 and fv0[2] == 2 and fv0[3] == 3)
    MINI_CHECK(fv1[0] == 4 and fv1[1] == 5 and fv1[2] == 6 and fv1[3] == 7)
    MINI_CHECK(fv2[0] == 0 and fv2[1] == 3 and fv2[2] == 5 and fv2[3] == 4)
    MINI_CHECK(fv3[0] == 2 and fv3[1] == 1 and fv3[2] == 7 and fv3[3] == 6)
    MINI_CHECK(fv4[0] == 0 and fv4[1] == 4 and fv4[2] == 7 and fv4[3] == 1)
    MINI_CHECK(fv5[0] == 3 and fv5[1] == 2 and fv5[2] == 6 and fv5[3] == 5)

    # flip_face(0): face 0 → [3,2,1,0], faces 1-5 unchanged
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

    # unify_winding: face 0 restored to [0,1,2,3], faces 1-5 unchanged
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

    # flip: face 0 → [3,2,1,0], face 1 → [7,6,5,4], face 2 → [4,5,3,0]
    # face 3 → [6,7,1,2], face 4 → [1,7,4,0], face 5 → [5,6,2,3]
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

    # orient_outward: face 0 → [0,1,2,3], face 1 → [4,5,6,7], face 2 → [0,3,5,4]
    # face 3 → [2,1,7,6], face 4 → [0,4,7,1], face 5 → [3,2,6,5]
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

    # edge edges
    # edge 1 - 2, edges: 1-0, 1-4, 2-3, 2-4
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

        ee_set = set(ee)
        MINI_CHECK(len(ee) == 4)
        MINI_CHECK(ee[0] == (1, 0))
        MINI_CHECK(ee[1] == (1, 4))
        MINI_CHECK(ee[2] == (2, 3))
        MINI_CHECK(ee[3] == (2, 4))

    # edge faces
    # edge 1-2, faces: 0, 1
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

    # face_edges
    # face 0, edges: 0-1, 1-2, 2-3, 3-0
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

    # face_faces
    # face 0, adjacent faces: 1
    ff = mesh.face_faces(f[0])
    if ff is not None:
        ff0 = ff[0]
        ffp = mesh.face_centroid(ff0)
        ffp.name = "f" + str(ff0)
        MINI_CHECK(len(ff) == 1)
        MINI_CHECK(ff0 == 1)

    # face points
    points = mesh.face_points(f[0])
    if points is not None:
        pointcount = len(points)
        MINI_CHECK(pointcount == 4)

    # face polyline
    pl = mesh.face_polyline(f[0])
    if pl is not None:
        pointcount = len(pl.get_points())
        MINI_CHECK(pointcount == 4)

    # face_vertices
    # face 0 vertices: 0, 1, 2, 3
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

    # vertex_edges
    # vertex 1, edges 1-0, 1-2, 1-4
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

    # vertex_faces
    vf = mesh.vertex_faces(v[1])
    # vertex 1, faces 0, 1
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

    # vertex_vertices
    # vertex 1, neighbors 0, 2, 4
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

    # area
    area = mesh.area()
    MINI_CHECK(TOLERANCE.is_close(area, 46.4528898159021))

    # centroid
    centroid = mesh.centroid()
    MINI_CHECK(TOLERANCE.is_point_close(centroid, Point(0.0, 0.0, 0.0)))

    # dihedral angle
    angles, arcs, points = mesh.dihedral_angles(0.3)

    for edge, angle in angles.items():
        angle_in_degrees = angle
        MINI_CHECK(TOLERANCE.is_close(angle_in_degrees, 116.565051177078))

    # face area
    for f in mesh.faces():
        face_area = mesh.face_area(f)
        MINI_CHECK(TOLERANCE.is_close(face_area, 3.87107415132518))

    # face centroid
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

    # face normal / s
    face_normals = mesh.face_normals()
    for f in mesh.faces():
        normal0 = mesh.face_normal(f)
        normal1 = face_normals[f]
        MINI_CHECK(TOLERANCE.is_vector_close(face_normals[f], mesh.face_normal(f)))

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

    # vertex angle in face
    for f in mesh.faces():
        for v in mesh.face_vertices(f):
            angle = mesh.vertex_angle_in_face(v, f)
            MINI_CHECK(TOLERANCE.is_close(mesh.vertex_angle_in_face(v, f), 1.8849555921538759))

    # vertex normal / s
    vertex_normals = mesh.vertex_normals()
    for v in mesh.vertices():
        normal0 = mesh.vertex_normal(v)
        normal1 = vertex_normals[v]
        MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals[v], mesh.vertex_normal(v)))

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

    # vertex normal weighted / s
    vertex_normals_weighted = mesh.vertex_normals_weighted(NormalWeighting.ANGLE)
    for v in mesh.vertices():
        normal0 = mesh.vertex_normal_weighted(v, NormalWeighting.ANGLE)
        normal1 = vertex_normals_weighted[v]
        MINI_CHECK(TOLERANCE.is_vector_close(vertex_normals_weighted[v], mesh.vertex_normal_weighted(v, NormalWeighting.ANGLE)))

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


    # volume
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

    # transform() — apply stored xform in-place; xform field unchanged
    mesh1 = mesh.duplicate()
    mesh1.xform = Xform.translation(0.0, 0.0, 1.0)
    mesh1.transform()

    MINI_CHECK(not mesh1.xform.is_identity())
    MINI_CHECK(mesh1.vertex_point(v0)[2] == 1.0)

    # transform(xf) — apply given xform in-place; stored xform unchanged
    mesh2 = mesh.duplicate()
    x = Xform.translation(0.0, 0.0, 1.0)
    mesh2.transform(x)
    MINI_CHECK(mesh2.xform.is_identity())
    MINI_CHECK(mesh2.vertex_point(v0)[2] == 1.0)

    # transformed() — copy with stored xform applied
    mesh3 = mesh.duplicate()
    mesh3.xform = Xform.translation(0.0, 0.0, 10.0)
    mesh3t = mesh3.transformed()
    MINI_CHECK(not mesh3t.xform.is_identity())
    MINI_CHECK(mesh3t.vertex_point(v0)[2] == 10.0)

    # transformed(xf) — copy with given xform applied
    mesh4 = mesh.duplicate()
    x = Xform.translation(0.0, 0.0, 10.0)
    mesh4t = mesh4.transformed(x)
    MINI_CHECK(mesh4t.xform.is_identity())
    MINI_CHECK(mesh4t.vertex_point(v0)[2] == 10.0)


@MINI_TEST("Mesh", "Json Roundtrip")
def test_mesh_json_roundtrip():
    from session_py import Mesh
    from session_py import Point
    from session_py import Xform
    from pathlib import Path

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.name = "test_mesh"
    mesh.xform = Xform.translation(1.0, 2.0, 3.0)

    # JSON object
    d = mesh.__jsondump__()
    loaded_json = Mesh.__jsonload__(d)

    # String
    json_string = mesh.file_json_dumps()
    loaded_string = Mesh.file_json_loads(json_string)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.json"
    mesh.file_json_dump(filename)
    loaded_file = Mesh.file_json_load(filename)

    MINI_CHECK(loaded_json == mesh)
    MINI_CHECK(loaded_string == mesh)
    MINI_CHECK(loaded_file == mesh)

    # Triangulation roundtrip
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

    # Face holes roundtrip
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
    from session_py import Xform
    from pathlib import Path

    mesh = Mesh.create_box(1.0, 1.0, 1.0)
    mesh.name = "test_mesh_proto"
    mesh.xform = Xform.translation(1.0, 2.0, 3.0)

    # String
    proto_bytes = mesh.pb_dumps()
    loaded_string = Mesh.pb_loads(proto_bytes)

    # File
    filename = Path(__file__).resolve().parents[2] / "serialization" / "test_mesh.bin"
    mesh.pb_dump(filename)
    loaded_file = Mesh.pb_load(filename)

    MINI_CHECK(loaded_string == mesh)
    MINI_CHECK(loaded_file == mesh)

    # Triangulation roundtrip
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

    # Face holes roundtrip
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


if __name__ == "__main__":
    run_all(language="python")
