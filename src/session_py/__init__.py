"""
This module contains all the classes and functions that are exposed to the user.
"""

from .color import Color
from .point import Point
from .vector import Vector
from .plane import Plane
from .line import Line
from .polyline import Polyline
from .xform import Xform
from .quaternion import Quaternion
from .tree import Tree
from .treenode import TreeNode
from .graph import Graph, Vertex, Edge
from .objects import Objects
from .session import Session
from .mesh import Mesh, NormalWeighting
from .aabb import AABB
from .obb import OBB
from .pointcloud import PointCloud
from .bvh import BVH, BVHNode
from .rtree import RTree
from .tolerance import Tolerance, TOLERANCE
from .session_config import SessionConfig, SESSION_CONFIG
from . import encoders
from .obj import load_obj, save_obj
from . import intersection
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .primitives import Primitives
from .trimmedsurface import TrimmedSurface
from .brep import BRep
from .element import Element
from .element_column import ElementColumn
from .element_beam import ElementBeam
from .element_plate import ElementPlate
from .closest import Closest
from .mesh_iso import MeshIso, TpmsType, TpmsMode
from .remesh_cdt import RemeshCDT
from .remesh_nurbssurface_grid import remesh_nurbssurface_grid, RemeshNurbsSurfaceGrid
from .remesh_nurbssurface_adaptive import RemeshNurbsSurfaceAdaptive
from .matrix import Matrix
from .convex_hull import ConvexHull
from .kdtree import KDTree
from .marching_squares import MarchingSquares

__all__ = [
    "Color",
    "Point",
    "Vector",
    "Plane",
    "Line",
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
    "BVH",
    "BVHNode",
    "RTree",
    "Tolerance",
    "TOLERANCE",
    "SessionConfig",
    "SESSION_CONFIG",
    "encoders",
    "load_obj",
    "save_obj",
    "intersection",
    "NurbsCurve",
    "NurbsSurface",
    "Primitives",
    "TrimmedSurface",
    "BRep",
    "Element",
    "ElementColumn",
    "ElementBeam",
    "ElementPlate",
    "Closest",
    "MeshIso",
    "TpmsType",
    "TpmsMode",
    "remesh_nurbssurface_grid",
    "RemeshCDT",
    "RemeshNurbsSurfaceGrid",
    "RemeshNurbsSurfaceAdaptive",
    "Matrix",
    "ConvexHull",
    "KDTree",
    "MarchingSquares",
]
