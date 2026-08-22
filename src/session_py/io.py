from .point import Point
from .pointcloud import PointCloud


def write_xyz_to_string(cloud: PointCloud) -> str:
    s = ""
    for p in cloud.get_points():
        s += f"{p.x} {p.y} {p.z}\n"
    return s


def write_xyz(cloud: PointCloud, filepath: str) -> None:
    with open(filepath, "w") as f:
        f.write(write_xyz_to_string(cloud))


def read_xyz(filepath: str) -> PointCloud:
    with open(filepath, "r") as f:
        content = f.read()
    return read_xyz_from_str(content)


def read_xyz_from_str(content: str) -> PointCloud:
    cloud = PointCloud()
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        cloud.add_point(Point(x, y, z))
    return cloud


save_xyz = write_xyz
load_xyz = read_xyz
