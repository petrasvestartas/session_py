#!/usr/bin/env python3
"""Test NURBS curve-plane intersection."""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.session_py.point import Point
from src.session_py.vector import Vector
from src.session_py.plane import Plane
from src.session_py.nurbscurve import NurbsCurve

print("=== NURBS Curve-Plane Intersection Test (Python) ===\n")

# Create NURBS curve from 3 points with degree 2
points = [
    Point(0.0, 0.0, -453.0),
    Point(1500.0, 0.0, -147.0),
    Point(3000.0, 0.0, -147.0)
]

degree = 2
curve = NurbsCurve.create(periodic=False, degree=degree, points=points)

print(f"Created NURBS curve: degree={curve.degree()}, cv_count={curve.cv_count()}")

# Create planes perpendicular to X-axis at regular intervals
planes = []
for i in range(7):
    planes.append(Plane.from_point_normal(Point(i*500, 0, 0), Vector(1, 0, 0)))

print(f"\nIntersecting curve with {len(planes)} planes:")

# Intersect curve with each plane
sampled_points = []
for plane in planes:
    intersection_points = curve.intersect_plane_points(plane)
    if intersection_points:
        sampled_points.append(intersection_points[0])
        pt = intersection_points[0]
        print(f"  Plane at x={plane.origin.x}: ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})")
    else:
        print(f"  Plane at x={plane.origin.x}: No intersection")

print(f"\nTotal sampled points: {len(sampled_points)}")
