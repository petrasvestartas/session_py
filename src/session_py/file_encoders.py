from __future__ import annotations
from typing import Any
import importlib
import json


_CLASS_MODULE_MAP = {
    "Vertex": "graph",
    "Edge": "graph",
    "TreeNode": "tree",
}

_EXTERNAL_CLASS_MAP: dict = {}


def file_register_class(name: str, cls) -> None:
    """Register an external class by its "type" name for polymorphic decoding"""
    _EXTERNAL_CLASS_MAP[name] = cls


def _get_class_from_name(class_name: str):
    """Find a class by name in the registry or by importing session_py.<lowercase name>"""

    if class_name in _EXTERNAL_CLASS_MAP:
        return _EXTERNAL_CLASS_MAP[class_name]

    try:
        mod = _CLASS_MODULE_MAP.get(class_name, class_name.lower())
        module = importlib.import_module(f"session_py.{mod}")

        return getattr(module, class_name, None)
    except (ImportError, AttributeError):
        return None


def _decode_typed(node: dict) -> Any:
    """Rebuild a geometry object from a dict with a "type" field, or return the dict"""

    try:
        class_name = node["type"].rsplit("/", 1)[-1]
        cls = _get_class_from_name(class_name)

        if cls is None or not hasattr(cls, "__jsonload__"):
            return node

        return cls.__jsonload__(node, node.get("guid"), node.get("name"))
    except Exception:
        return node


class GeometryFileEncoder(json.JSONEncoder):
    """JSON encoder that serializes geometry objects through __jsondump__"""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "__jsondump__"):
            return obj.__jsondump__()

        if hasattr(obj, "__next__"):
            return list(obj)

        return super().default(obj)


class GeometryFileDecoder(json.JSONDecoder):
    """JSON decoder that rebuilds geometry objects from their "type" field"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: dict) -> Any:
        if "type" not in obj:
            return obj

        return _decode_typed(obj)


def file_json_dump(data: Any, filepath: str, pretty: bool = True) -> None:
    """Write data to a json file"""
    with open(filepath, "w") as f:
        f.write(file_json_dumps(data, pretty))


def file_json_load(filepath: str) -> Any:
    """Read data from a json file"""
    with open(filepath) as f:
        return file_json_loads(f.read())


def file_json_dumps(data: Any, pretty: bool = True) -> str:
    """Serialize data to a json string"""
    if pretty:
        return json.dumps(data, cls=GeometryFileEncoder, indent=4)

    return json.dumps(data, cls=GeometryFileEncoder)


def file_json_loads(json_str: str) -> Any:
    """Deserialize data from a json string"""
    return json.loads(json_str, cls=GeometryFileDecoder)


def file_decode_node(node: Any) -> Any:
    """Recursively rebuild geometry objects inside a decoded json node"""

    if isinstance(node, list):
        return [file_decode_node(x) for x in node]

    if not isinstance(node, dict):
        return node

    if "type" in node:
        return _decode_typed(node)

    return {k: file_decode_node(v) for k, v in node.items()}
