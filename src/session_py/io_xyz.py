from __future__ import annotations
from .point import Point
from .pointcloud import PointCloud


def write_xyz_to_string(cloud: PointCloud) -> str:
    """Return the cloud points as "x y z" lines at full double precision."""

    s = ""

    for p in cloud.get_points():
        s += f"{p[0]} {p[1]} {p[2]}\n"

    return s


def write_xyz(cloud: PointCloud, filepath: str) -> None:
    """Write the cloud points as "x y z" lines to filepath."""
    with open(filepath, "w") as f:
        f.write(write_xyz_to_string(cloud))


def read_xyz_from_str(content: str) -> PointCloud:
    """Return the cloud read from "x y z" lines; blank and # lines skipped."""

    cloud = PointCloud()

    for line in content.splitlines():
        if not line or line[0] == "#":
            continue

        parts = line.split()

        if len(parts) < 3:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
        except ValueError:
            continue

        cloud.add_point(Point(x, y, z))

    return cloud


def read_xyz(filepath: str) -> PointCloud:
    """Return the cloud read from an .xyz file."""
    with open(filepath) as f:
        content = f.read()

    return read_xyz_from_str(content)
