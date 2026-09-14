from __future__ import annotations
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("session_py")
except PackageNotFoundError:
    __version__ = "0.0.0"
"""
This module contains all the classes and functions that are exposed to the user.
"""

from .color import Color
from .point import Point
from .vector import Vector
from .plane import Plane
from .line import Line
from .instance_ref import InstanceRef
from .polyline import Polyline
from .xform import Xform
from .quaternion import Quaternion
from .tree import Tree
from .tree import TreeNode
from .graph import Graph
from .graph import Vertex
from .graph import Edge
from .objects import Objects
from .history import History
from .session import Session
from .mesh import Mesh
from .mesh import NormalWeighting
from .aabb import AABB
from .spatial_aabbtree import SpatialAABBTree
from .obb import OBB
from .pointcloud import PointCloud
from .spatial_bvh import SpatialBVH
from .spatial_rtree import SpatialRTree
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .session_config import SessionConfig
from .session_config import SESSION_CONFIG
from . import file_encoders
from .file_obj import read_file_obj
from .file_obj import write_file_obj
from .file_obj import read_file_obj_from_str
from .file_obj import write_file_obj_to_string
from .file_obj import read_file_obj_polylines
from . import file_step
from .io import read_xyz
from .io import write_xyz
from .io import read_xyz_from_str
from .io import write_xyz_to_string
from . import intersection
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .primitives import Primitives
from .nurbssurface_trimmed import NurbsSurfaceTrimmed
from .nurbssurface_trimmed import TrimLoops
from .brep import BRep
from .brep import BRepOrientation
from .brep import BRepRef
from .element import Element
from .element import ElementFeature
from .closest import Closest
from .boolean_polyline import BooleanPolyline
from .remesh_cdt import RemeshCDT
from .remesh_nurbssurface_grid import RemeshNurbsSurfaceGrid
from .remesh_nurbssurface_adaptive import RemeshNurbsSurfaceAdaptive
from .matrix import Matrix
from .convex_hull import ConvexHull
from .spatial_kdtree import SpatialKDTree
from .spatial_octree import SpatialOctree
from .mesh_offset import MeshOffset

__all__ = [
    "Color",
    "Point",
    "Vector",
    "Plane",
    "Line",
    "InstanceRef",
    "Polyline",
    "Xform",
    "Quaternion",
    "Tree",
    "TreeNode",
    "Graph",
    "Vertex",
    "Edge",
    "Objects",
    "History",
    "Session",
    "Mesh",
    "NormalWeighting",
    "OBB",
    "PointCloud",
    "SpatialAABBTree",
    "SpatialBVH",
    "SpatialRTree",
    "Tolerance",
    "TOLERANCE",
    "SessionConfig",
    "SESSION_CONFIG",
    "file_encoders",
    "read_file_obj",
    "write_file_obj",
    "read_file_obj_from_str",
    "write_file_obj_to_string",
    "read_file_obj_polylines",
    "file_step",
    "read_xyz",
    "write_xyz",
    "read_xyz_from_str",
    "write_xyz_to_string",
    "intersection",
    "NurbsCurve",
    "NurbsSurface",
    "Primitives",
    "NurbsSurfaceTrimmed",
    "TrimLoops",
    "BRep",
    "BRepOrientation",
    "BRepRef",
    "Element",
    "ElementFeature",
    "Closest",
    "BooleanPolyline",
    "RemeshCDT",
    "RemeshNurbsSurfaceGrid",
    "RemeshNurbsSurfaceAdaptive",
    "Matrix",
    "ConvexHull",
    "SpatialKDTree",
    "SpatialOctree",
    "MeshOffset",
]
