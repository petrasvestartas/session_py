from session_py import NurbsSurface, Point
from compas_viewer import Viewer
from compas.datastructures import Mesh
from compas.geometry import Point as CompasPoint, Polyline

# Create a NURBS surface (3D, order 4, 5x5 control points)
srf = NurbsSurface(3, False, 4, 4, 5, 5)

# Setup nurbsknot vectors
srf.make_clamped_uniform_nurbsknot_vector(0, 1.0)
srf.make_clamped_uniform_nurbsknot_vector(1, 1.0)

# Set control points (hardcoded 5x5 grid)
srf.set_cv(0, 0, Point(0.0, 0.0, 2.5*-1))
srf.set_cv(0, 1, Point(0.0, 1.0, 0.0))
srf.set_cv(0, 2, Point(0.0, 2.0, 0.0))
srf.set_cv(0, 3, Point(0.0, 3.0, 0.0))
srf.set_cv(0, 4, Point(0.0, 4.0, 2.5*-1))

srf.set_cv(1, 0, Point(1.0, 0.0, 0.0))
srf.set_cv(1, 1, Point(1.0, 1.0, 0.0))
srf.set_cv(1, 2, Point(1.0, 2.0, 5.0))
srf.set_cv(1, 3, Point(1.0, 3.0, 0.0))
srf.set_cv(1, 4, Point(1.0, 4.0, 0.0))

srf.set_cv(2, 0, Point(2.0, 0.0, 0.0))
srf.set_cv(2, 1, Point(2.0, 1.0, 0.0))
srf.set_cv(2, 2, Point(2.0, 2.0, 0.0))
srf.set_cv(2, 3, Point(2.0, 3.0, 0.0))
srf.set_cv(2, 4, Point(2.0, 4.0, 0.0))

srf.set_cv(3, 0, Point(3.0, 0.0, 0.0))
srf.set_cv(3, 1, Point(3.0, 1.0, 0.0))
srf.set_cv(3, 2, Point(3.0, 2.0, 0.0))
srf.set_cv(3, 3, Point(3.0, 3.0, 0.0))
srf.set_cv(3, 4, Point(3.0, 4.0, 0.0))

srf.set_cv(4, 0, Point(4.0, 0.0, 2.5*-1))
srf.set_cv(4, 1, Point(4.0, 1.0, 0.0))
srf.set_cv(4, 2, Point(4.0, 2.0, 0.0))
srf.set_cv(4, 3, Point(4.0, 3.0, 0.0))
srf.set_cv(4, 4, Point(4.0, 4.0, 2.5*-1))

print("NURBS Surface Grid Evaluation\n")

# Evaluate surface at subdivided grid points
grid_size = 10  # Increase for finer mesh
u_min, u_max = srf.domain(0)
v_min, v_max = srf.domain(1)

# Store evaluated points in 2D grid
points_grid = []
for i in range(grid_size):
    row = []
    for j in range(grid_size):
        # Calculate parameters
        u = u_min + (u_max - u_min) * i / (grid_size - 1)
        v = v_min + (v_max - v_min) * j / (grid_size - 1)
        
        # Evaluate point
        pt = srf.point_at(u, v)
        row.append(CompasPoint(pt.x, pt.y, pt.z))
    points_grid.append(row)

print(f"Evaluated {grid_size}x{grid_size} = {grid_size*grid_size} points")

# Create COMPAS Mesh
mesh = Mesh()

# Add vertices and store their keys in a 2D array
vertex_keys = []
for i in range(grid_size):
    row_keys = []
    for j in range(grid_size):
        pt = points_grid[i][j]
        key = mesh.add_vertex(x=pt.x, y=pt.y, z=pt.z)
        row_keys.append(key)
    vertex_keys.append(row_keys)

print(f"Added {mesh.number_of_vertices()} vertices")

# Create quad faces
face_count = 0
for i in range(grid_size - 1):
    for j in range(grid_size - 1):
        # Create quad face from 4 adjacent vertices
        v0 = vertex_keys[i][j]
        v1 = vertex_keys[i+1][j]
        v2 = vertex_keys[i+1][j+1]
        v3 = vertex_keys[i][j+1]
        mesh.add_face([v0, v1, v2, v3])
        face_count += 1

print(f"Added {face_count} quad faces")
print(f"Mesh: {mesh.number_of_vertices()} vertices, {mesh.number_of_faces()} faces, {mesh.number_of_edges()} edges")

# Extract isocurves and convert to polylines
print("\nExtracting Isocurves...")
iso_polylines = []
num_iso_curves = 5  # Number of isocurves in each direction
curve_divisions = 10  # Points per curve

# Extract iso-u curves (v varies, u constant)
print(f"Extracting {num_iso_curves} iso-u curves...")
for i in range(num_iso_curves):
    u_param = u_min + (u_max - u_min) * i / (num_iso_curves - 1)
    iso_curve = srf.iso_curve(0, u_param)
    
    if iso_curve:
        curve_points = []
        t_min, t_max = iso_curve.domain()
        
        for j in range(curve_divisions):
            t = t_min + (t_max - t_min) * j / (curve_divisions - 1)
            pt = iso_curve.point_at(t)
            curve_points.append(CompasPoint(pt.x, pt.y, pt.z))
        
        polyline = Polyline(curve_points)
        iso_polylines.append(polyline)
        print(f"  Iso-u at u={u_param:.2f}: {len(curve_points)} points")

# Extract iso-v curves (u varies, v constant)
print(f"Extracting {num_iso_curves} iso-v curves...")
for i in range(num_iso_curves):
    v_param = v_min + (v_max - v_min) * i / (num_iso_curves - 1)
    iso_curve = srf.iso_curve(1, v_param)
    
    if iso_curve:
        curve_points = []
        t_min, t_max = iso_curve.domain()
        
        for j in range(curve_divisions):
            t = t_min + (t_max - t_min) * j / (curve_divisions - 1)
            pt = iso_curve.point_at(t)
            curve_points.append(CompasPoint(pt.x, pt.y, pt.z))
        
        polyline = Polyline(curve_points)
        iso_polylines.append(polyline)
        print(f"  Iso-v at v={v_param:.2f}: {len(curve_points)} points")

print(f"\nTotal isocurve polylines: {len(iso_polylines)}")

# Visualize with COMPAS Viewer
viewer = Viewer()
viewer.scene.add(mesh, show_vertices=False, show_edges=True, show_faces=True, opacity=0.7)

# Add isocurve polylines with different color
for polyline in iso_polylines:
    viewer.scene.add(polyline, linewidth=2, linecolor=(1, 0, 0))  # Red isocurves

viewer.show()

print("\n✅ Done!")