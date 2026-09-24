from __future__ import annotations
from typing import Any
import importlib
import json


_CLASS_MODULE_MAP = {
    "Vertex": "graph",
    "Edge": "graph",
    "ElementFeature": "element",
    "InstanceRef": "instance_ref",
    "NurbsSurfaceTrimmed": "nurbssurface_trimmed",
    "TreeNode": "tree",
}

_EXTERNAL_CLASS_MAP: dict = {}


# ═══════════════════════════════════════════════════════════════════════════
# JSON string
# ═══════════════════════════════════════════════════════════════════════════
def file_json_dumps(data: Any, pretty: bool = True) -> str:
    """Serialize an object to a JSON string."""

    if pretty:
        return json.dumps(data, cls=GeometryFileEncoder, indent=4)

    return json.dumps(data, cls=GeometryFileEncoder)


def file_json_loads(json_str: str) -> Any:
    """Deserialize an object from a JSON string."""

    return json.loads(json_str, cls=GeometryFileDecoder)


# ═══════════════════════════════════════════════════════════════════════════
# JSON file
# ═══════════════════════════════════════════════════════════════════════════
def file_json_dump(data: Any, filepath: str, pretty: bool = True) -> None:
    """Write an object to a JSON file."""

    with open(filepath, "w") as file:
        file.write(file_json_dumps(data, pretty))


def file_json_load_data(filepath: str) -> Any:
    """Read a JSON value from a file."""

    with open(filepath) as file:
        return json.load(file)


def file_json_load(filepath: str) -> Any:
    """Read an object from a JSON file."""

    with open(filepath) as file:
        return file_json_loads(file.read())


# ═══════════════════════════════════════════════════════════════════════════
# Collections
# ═══════════════════════════════════════════════════════════════════════════
def file_encode_collection(collection: list) -> list:
    """Encode a collection of objects to a JSON array, skipping None entries."""

    array = []

    for item in collection:
        if item is not None:
            array.append(item.__jsondump__())

    return array


def file_decode_collection(data: Any, cls: type) -> list:
    """Decode a JSON array to a collection of objects."""

    result = []

    if not isinstance(data, list):
        return result

    for item in data:
        result.append(cls.__jsonload__(item))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Polymorphic decoding
# ═══════════════════════════════════════════════════════════════════════════
def file_register_class(name: str, cls: type) -> None:
    """Register an external class by its "type" name for polymorphic decoding."""

    _EXTERNAL_CLASS_MAP[name] = cls


def _get_class_from_name(class_name: str):
    """Find a class by name in the registry or by importing session_py.<lowercase name>."""

    if class_name in _EXTERNAL_CLASS_MAP:
        return _EXTERNAL_CLASS_MAP[class_name]

    try:
        mod = _CLASS_MODULE_MAP.get(class_name, class_name.lower())
        module = importlib.import_module(f"session_py.{mod}")

        return getattr(module, class_name, None)
    except (ImportError, AttributeError):
        return None


def _decode_typed(node: dict) -> Any:
    """Rebuild a geometry object from a dict with a "type" field, a Component for an unknown type with a guid and name, or return the dict."""

    try:
        class_name = node["type"].rsplit("/", 1)[-1]
        cls = _get_class_from_name(class_name)

        if cls is None and "guid" in node and "name" in node:
            from .objects import Component

            cls = Component

        if cls is None or not hasattr(cls, "__jsonload__"):
            return node

        return cls.__jsonload__(node)
    except Exception:
        return node


def file_decode_node(node: Any) -> Any:
    """Recursively rebuild geometry objects inside a decoded JSON node."""

    if isinstance(node, list):
        result = []

        for item in node:
            result.append(file_decode_node(item))

        return result

    if not isinstance(node, dict):
        return node

    if "type" in node:
        return _decode_typed(node)

    result = {}

    for key, value in node.items():
        result[key] = file_decode_node(value)

    return result


class GeometryFileEncoder(json.JSONEncoder):
    """JSON encoder that serializes geometry objects through __jsondump__."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "__jsondump__"):
            return obj.__jsondump__()

        if hasattr(obj, "__next__"):
            return list(obj)

        return super().default(obj)


class GeometryFileDecoder(json.JSONDecoder):
    """JSON decoder that rebuilds geometry objects from their "type" field."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: dict) -> Any:
        if "type" not in obj:
            return obj

        return _decode_typed(obj)
