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
from .graph import Graph
from .vertex import Vertex
from .edge import Edge
from .objects import Objects
from .session import Session
from .mesh import Mesh, NormalWeighting
from .boundingbox import BoundingBox
from .pointcloud import PointCloud
from .bvh import BVH, BVHNode
from .tolerance import Tolerance
from . import encoders
from .obj import load_obj, save_obj
from . import intersection
from .nurbscurve import NurbsCurve
from .nurbssurface import NurbsSurface
from .primitives import Primitives
from .trimmedsurface import TrimmedSurface
from .brep import BRep
from .ray_box_intersection import ray_box
from .closest import Closest

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
    "BoundingBox",
    "PointCloud",
    "BVH",
    "BVHNode",
    "Tolerance",
    "encoders",
    "load_obj",
    "save_obj",
    "intersection",
    "NurbsCurve",
    "NurbsSurface",
    "Primitives",
    "TrimmedSurface",
    "BRep",
    "ray_box",
    "Closest",
]
