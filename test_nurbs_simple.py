#!/usr/bin/env python3
"""Simple test for NURBS curve evaluation."""

from src.session_py.point import Point
from src.session_py.nurbscurve import NurbsCurve

# Create NURBS curve from 3 points with degree 2
p0 = Point(0.0, 0.0, -453.0)
p1 = Point(1500.0, 0.0, -147.0)
p2 = Point(3000.0, 0.0, -147.0)

points = [p0, p1, p2]
degree = 2

# Create a clamped NURBS curve
curve = NurbsCurve.create(periodic=False, degree=degree, points=points)

print(f"Created NURBS curve: degree={curve.degree()}, cv_count={curve.cv_count()}")
print(f"Is valid: {curve.is_valid()}")
print(f"Dimension: {curve.dimension()}")
print(f"Order: {curve.order()}")
print(f"Knot count: {curve.knot_count()}")
print(f"Knot vector: {list(curve.m_knot)}")

t0, t1 = curve.domain()
print(f"Domain: [{t0}, {t1}]")

print(f"\nControl points:")
for i in range(curve.cv_count()):
    cv = curve.get_cv(i)
    if cv:
        print(f"  CV{i}: ({cv.x:.2f}, {cv.y:.2f}, {cv.z:.2f})")

# Divide curve into 6 points
divided_points, params = curve.divide_by_count(6, include_endpoints=True)

print(f"\nDivided into {len(divided_points)} points:")
for i, (pt, t) in enumerate(zip(divided_points, params)):
    print(f"  Point{i} (t={t:.4f}): ({pt.x:.2f}, {pt.y:.2f}, {pt.z:.2f})")
