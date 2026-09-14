from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# TreeNode
# ═══════════════════════════════════════════════════════════════════════════


class TreeNode:
    """A node of a tree; geometry nodes are named by their object's guid, group nodes by a label"""

    def __init__(self, name: str = "my_node"):
        self._guid = None
        self._parent = None
        self._children = []
        self.name = name
        self.color = None

    def has_guid(self) -> bool:
        return getattr(self, "_guid", None) is not None

    @property
    def guid(self) -> str:
        if getattr(self, "_guid", None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return not self._children

    def add(self, child: TreeNode) -> None:
        """Add a child node to this node"""
        if child is None:
            return
        child._parent = self
        self._children.append(child)

    def remove(self, child: TreeNode) -> TreeNode | None:
        """Remove a child node and return it (None if not found)"""
        for i in range(len(self._children)):
            if self._children[i] is not child:
                continue
            removed = self._children.pop(i)
            removed._parent = None
            return removed
        return None

    @property
    def parent(self) -> TreeNode | None:
        return self._parent

    @property
    def ancestors(self) -> list[TreeNode]:
        result = []
        current = self._parent
        while current is not None:
            result.append(current)
            current = current._parent
        return result

    @property
    def descendants(self) -> list[TreeNode]:
        result = []
        for child in self._children:
            result.append(child)
            result.extend(child.descendants)
        return result

    @property
    def children(self) -> list[TreeNode]:
        return self._children

    def traverse(
        self, strategy: str = "depthfirst", order: str = "preorder"
    ) -> list[TreeNode]:
        """Traverse from this node ("depthfirst"|"breadthfirst", "preorder"|"postorder")"""
        result = []
        if strategy == "depthfirst":
            if order != "preorder" and order != "postorder":
                raise ValueError(f"Unknown traversal order: {order}")
            if order == "preorder":
                result.append(self)
            for child in self._children:
                result.extend(child.traverse(strategy, order))
            if order == "postorder":
                result.append(self)
        elif strategy == "breadthfirst":
            queue = deque([self])
            while queue:
                current = queue.popleft()
                result.append(current)
                for child in current._children:
                    queue.append(child)
        else:
            raise ValueError(f"Unknown traversal strategy: {strategy}")
        return result

    def __eq__(self, other) -> bool:
        return isinstance(other, TreeNode) and self.guid == other.guid

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.guid)

    def __jsondump__(self) -> dict:
        children = []
        for child in self._children:
            children.append(child.__jsondump__())
        data = {"children": children}
        if self.color is not None:
            data["color"] = self.color.__jsondump__()
        data["guid"] = self.guid
        data["name"] = self.name
        data["type"] = "TreeNode"
        return data

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> TreeNode:
        from .color import Color

        node = cls(data["name"])
        node.guid = guid if guid is not None else data["guid"]
        color = data.get("color")
        if isinstance(color, dict):
            color = Color.__jsonload__(color)
        node.color = color
        for child_data in data["children"]:
            child = child_data
            if isinstance(child_data, dict):
                child = TreeNode.__jsonload__(child_data)
            node.add(child)
        return node

    def __str__(self) -> str:
        return f"TreeNode({self.name}, {self.guid}, {len(self._children)} children)"

    def __repr__(self) -> str:
        return self.__str__()


# ═══════════════════════════════════════════════════════════════════════════
# Tree
# ═══════════════════════════════════════════════════════════════════════════


class Tree:
    """A hierarchy of TreeNodes under one root"""

    def __init__(self, name: str = "my_tree"):
        self._guid = None
        self._root = None
        self.name = name

    def has_guid(self) -> bool:
        return getattr(self, "_guid", None) is not None

    @property
    def guid(self) -> str:
        if getattr(self, "_guid", None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    @property
    def root(self) -> TreeNode | None:
        return self._root

    def add(self, node: TreeNode, parent: TreeNode | None = None) -> None:
        """Add a node to the tree (parent=None adds as root)"""
        if node is None:
            raise ValueError("Cannot add null node")
        if parent is not None:
            parent.add(node)
            return
        if self._root is not None:
            raise ValueError("Tree already has a root node")
        self._root = node

    @property
    def nodes(self) -> list[TreeNode]:
        result = []
        if self._root is None:
            return result
        queue = deque([self._root])
        while queue:
            current = queue.popleft()
            result.append(current)
            for child in current._children:
                queue.append(child)
        return result

    def remove(self, node: TreeNode) -> TreeNode:
        """Remove a node and return it with its subtree intact"""
        if node is None:
            raise ValueError("Cannot remove null node")
        if node is self._root:
            self._root = None
            return node
        parent = node.parent
        if parent is None:
            raise ValueError("Node is not in this tree")
        return parent.remove(node)

    @property
    def leaves(self) -> list[TreeNode]:
        result = []
        for node in self.nodes:
            if node.is_leaf:
                result.append(node)
        return result

    def traverse(
        self, strategy: str = "depthfirst", order: str = "preorder"
    ) -> list[TreeNode]:
        """Traverse from root ("depthfirst"|"breadthfirst", "preorder"|"postorder")"""
        if self._root is None:
            return []
        return self._root.traverse(strategy, order)

    def get_node_by_name(self, node_name: str) -> TreeNode | None:
        """First node with the given name (None if not found)"""
        for node in self.nodes:
            if node.name == node_name:
                return node
        return None

    def get_nodes_by_name(self, node_name: str) -> list[TreeNode]:
        """All nodes with the given name"""
        result = []
        for node in self.nodes:
            if node.name == node_name:
                result.append(node)
        return result

    def find_node_by_guid(self, node_guid: str) -> TreeNode | None:
        """Node with the given guid (None if not found)"""
        for node in self.nodes:
            if node.guid == node_guid:
                return node
        return None

    def add_child_by_guid(self, parent_guid: str, child_guid: str) -> bool:
        """Reparent a child by guid; False when either node is missing or the child is the root"""
        parent = self.find_node_by_guid(parent_guid)
        child = self.find_node_by_guid(child_guid)
        if parent is None or child is None:
            return False
        current = child.parent
        if current is None:
            return False
        current.remove(child)
        parent.add(child)
        return True

    def get_children_guids(self, node_guid: str) -> list[str]:
        """Guids of the children of a node by guid (empty if not found)"""
        result = []
        node = self.find_node_by_guid(node_guid)
        if node is None:
            return result
        for child in node.children:
            result.append(child.guid)
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════════════════════════════

    def __jsondump__(self) -> dict:
        return {
            "guid": self.guid,
            "name": self.name,
            "root": self._root.__jsondump__() if self._root is not None else None,
            "type": "Tree",
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: str | None = None, name: str | None = None
    ) -> Tree:
        tree = cls(data["name"])
        tree.guid = guid if guid is not None else data["guid"]
        root = data["root"]
        if isinstance(root, dict):
            root = TreeNode.__jsonload__(root)
        if root is not None:
            tree.add(root)
        return tree

    def file_json_dumps(self) -> str:
        import json

        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Tree:
        import json

        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        import json

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Tree:
        import json

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    def pb_dumps(self) -> bytes:
        from .proto import tree_pb2

        proto = tree_pb2.Tree()
        if self.has_guid():
            proto.guid = self.guid
        proto.name = self.name
        if self._root is not None:
            proto.root.CopyFrom(_node_to_proto(self._root))
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> Tree:
        from .proto import tree_pb2

        proto = tree_pb2.Tree()
        proto.ParseFromString(data)
        tree = cls(proto.name)
        if proto.guid:
            tree.guid = proto.guid
        if proto.HasField("root"):
            tree.add(_proto_to_node(proto.root))
        return tree

    def pb_dump(self, filename: str | Path) -> None:
        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Tree:
        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    def __str__(self) -> str:
        return f"Tree: {self.name}"

    def __repr__(self) -> str:
        return self.__str__()


def _node_to_proto(node: TreeNode):
    from .proto import treenode_pb2

    proto = treenode_pb2.TreeNode()
    proto.guid = node.guid
    proto.name = node.name
    proto.parent_guid = ""
    if node.color is not None:
        proto.color.r = node.color.r
        proto.color.g = node.color.g
        proto.color.b = node.color.b
        proto.color.a = node.color.a
    for child in node.children:
        proto.children.append(_node_to_proto(child))
    return proto


def _proto_to_node(proto) -> TreeNode:
    from .color import Color

    node = TreeNode(proto.name)
    node.guid = proto.guid
    if proto.HasField("color") and proto.color.a > 0:
        node.color = Color(proto.color.r, proto.color.g, proto.color.b, proto.color.a)
    for child in proto.children:
        node.add(_proto_to_node(child))
    return node
