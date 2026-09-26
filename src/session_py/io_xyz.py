from __future__ import annotations
from .point import Point
from .pointcloud import PointCloud


# ═══════════════════════════════════════════════════════════════════════════
# Write
# ═══════════════════════════════════════════════════════════════════════════
def _format_number(value: float) -> str:
    """Return the shortest round-trip text of value, without a trailing ".0"."""

    text = repr(value)

    return text.removesuffix(".0")


def write_xyz_to_string(cloud: PointCloud) -> str:
    """Return the cloud points as "x y z" lines, each number the shortest round-trip text."""

    out = ""

    for p in cloud.get_points():
        out += f"{_format_number(p[0])} {_format_number(p[1])} {_format_number(p[2])}\n"

    return out


def write_xyz(cloud: PointCloud, filepath: str) -> None:
    """Write the cloud points as "x y z" lines to filepath; raises if it cannot be opened."""

    with open(filepath, "w", newline="") as file:
        file.write(write_xyz_to_string(cloud))


# ═══════════════════════════════════════════════════════════════════════════
# Read
# ═══════════════════════════════════════════════════════════════════════════
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
    """Return the cloud read from an .xyz file; raises if it cannot be opened."""

    with open(filepath) as file:
        content = file.read()

    return read_xyz_from_str(content)
