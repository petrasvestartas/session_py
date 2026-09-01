from __future__ import annotations
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
from .session import Session
from .mesh import Mesh
from .mesh import NormalWeighting
from .aabb import AABB
from .spatial_aabbtree import SpatialAABBTree
from .obb import OBB
from .pointcloud import PointCloud
from .spatial_bvh import SpatialBVH
from .spatial_bvh import SpatialBVHNode
from .spatial_rtree import SpatialRTree
from .tolerance import Tolerance
from .tolerance import TOLERANCE
from .session_config import SessionConfig
from .session_config import SESSION_CONFIG
from . import file_encoders
from .file_obj import load_file_obj
from .file_obj import save_file_obj
from .io import read_xyz
from .io import write_xyz
from .io import read_xyz_from_str
from .io import write_xyz_to_string
from . import intersection
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .primitives import Primitives
from .nurbssurface_trimmed import NurbsSurfaceTrimmed
from .brep import BRep
from .element import Element
from .element import ElementFeature
from .closest import Closest
from .remesh_cdt import RemeshCDT
from .remesh_nurbssurface_grid import remesh_nurbssurface_grid
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
    "Session",
    "Mesh",
    "NormalWeighting",
    "OBB",
    "PointCloud",
    "SpatialAABBTree",
    "SpatialBVH",
    "SpatialBVHNode",
    "SpatialRTree",
    "Tolerance",
    "TOLERANCE",
    "SessionConfig",
    "SESSION_CONFIG",
    "file_encoders",
    "load_file_obj",
    "save_file_obj",
    "read_xyz",
    "write_xyz",
    "read_xyz_from_str",
    "write_xyz_to_string",
    "intersection",
    "NurbsCurve",
    "NurbsSurface",
    "Primitives",
    "NurbsSurfaceTrimmed",
    "BRep",
    "Element",
    "ElementFeature",
    "Closest",
    "remesh_nurbssurface_grid",
    "RemeshCDT",
    "RemeshNurbsSurfaceGrid",
    "RemeshNurbsSurfaceAdaptive",
    "Matrix",
    "ConvexHull",
    "SpatialKDTree",
    "SpatialOctree",
    "MeshOffset",
]
