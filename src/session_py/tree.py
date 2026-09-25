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
    """A node of a tree; geometry nodes are named by their object's guid, group nodes by a label."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, name: str = "my_node"):
        """Construct a node with a name."""

        self._guid = None  # Lazy guid.
        self._parent = None  # Parent node, None for the root.
        self._children = []  # Child nodes in order.
        self.name = name  # Object guid or group label.
        self.color = None  # Display colour override.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    @property
    def is_root(self) -> bool:
        """Return whether this node has no parent."""
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        """Return whether this node has no children."""
        return not self._children

    @property
    def parent(self) -> TreeNode | None:
        """Return the parent node, or None when this is the root."""
        return self._parent

    @property
    def ancestors(self) -> list[TreeNode]:
        """Return all ancestors from the immediate parent up to the root."""

        result = []
        current = self._parent

        while current is not None:
            result.append(current)
            current = current._parent

        return result

    def descendants(self) -> list[TreeNode]:
        """Return all descendants of this node, depth-first."""

        result = self.traverse("depthfirst", "preorder")
        result.pop(0)

        return result

    @property
    def children(self) -> list[TreeNode]:
        """Return the direct children of this node."""
        return self._children

    # ═══════════════════════════════════════════════════════════════════════════
    # Mutators
    # ═══════════════════════════════════════════════════════════════════════════
    def add(self, child: TreeNode) -> None:
        """Add a child node to this node."""

        if child is None:
            return

        if child is self:
            return

        ancestor = self

        while ancestor is not None:
            if ancestor is child:
                return

            ancestor = ancestor.parent

        child._parent = self
        self._children.append(child)

    def remove(self, child: TreeNode) -> TreeNode | None:
        """Remove a child node and return it, or None when not found."""

        for i in range(len(self._children)):
            if self._children[i] is not child:
                continue

            removed = self._children.pop(i)
            removed._parent = None

            return removed

        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # Operators
    # ═══════════════════════════════════════════════════════════════════════════
    def __eq__(self, other) -> bool:
        """Compare by guid."""
        return isinstance(other, TreeNode) and self.guid == other.guid

    def __ne__(self, other) -> bool:
        """Compare by guid."""
        return not self == other

    def __hash__(self) -> int:
        """Hash by guid."""
        return hash(self.guid)

    # ═══════════════════════════════════════════════════════════════════════════
    # Traversal
    # ═══════════════════════════════════════════════════════════════════════════
    def traverse(
        self, strategy: str = "depthfirst", order: str = "preorder"
    ) -> list[TreeNode]:
        """Traverse from this node ("depthfirst"|"breadthfirst", "preorder"|"postorder")."""

        result = []

        if strategy == "depthfirst":
            if order != "preorder" and order != "postorder":
                raise ValueError(f"Unknown traversal order: {order}")

            stack = [self]

            while stack:
                current = stack.pop()
                result.append(current)

                if order == "preorder":
                    for i in range(len(current._children) - 1, -1, -1):
                        stack.append(current._children[i])
                else:
                    stack.extend(current._children)

            if order == "postorder":
                result.reverse()
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

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

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
        """Deserialize from a JSON object."""

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

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return the name and child count."""
        return f"TreeNode({self.name}, {len(self._children)} children)"

    def __repr__(self) -> str:
        """Return the name, guid and child count."""
        return f"TreeNode({self.name}, {self.guid}, {len(self._children)} children)"


# ═══════════════════════════════════════════════════════════════════════════
# Tree
# ═══════════════════════════════════════════════════════════════════════════
class Tree:
    """A hierarchy of TreeNodes under one root."""

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════
    def __init__(self, name: str = "my_tree"):
        """Construct an empty tree with a name."""

        self._guid = None  # Lazy guid.
        self._root = None  # Root node, None when empty.
        self.name = name  # Tree name.

    # ═══════════════════════════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════════════════════════
    def has_guid(self) -> bool:
        """Return whether the lazy guid has been created."""
        return self._guid is not None

    @property
    def guid(self) -> str:
        """Return the guid, creating it on first access."""

        if self._guid is None:
            self._guid = str(uuid.uuid4())

        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        """Set the guid."""
        self._guid = value

    @property
    def root(self) -> TreeNode | None:
        """Return the root node, or None when empty."""
        return self._root

    @property
    def nodes(self) -> list[TreeNode]:
        """Return all nodes in the tree, breadth-first from the root."""

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

    @property
    def leaves(self) -> list[TreeNode]:
        """Return all nodes without children."""

        result = []

        for node in self.nodes:
            if node.is_leaf:
                result.append(node)

        return result

    def get_node_by_name(self, node_name: str) -> TreeNode | None:
        """Return the first node with the given name, or None when not found."""

        for node in self.nodes:
            if node.name == node_name:
                return node

        return None

    def get_nodes_by_name(self, node_name: str) -> list[TreeNode]:
        """Return all nodes with the given name."""

        result = []

        for node in self.nodes:
            if node.name == node_name:
                result.append(node)

        return result

    def find_node_by_guid(self, node_guid: str) -> TreeNode | None:
        """Return the node with the given guid, or None when not found."""

        for node in self.nodes:
            if node.guid == node_guid:
                return node

        return None

    def get_children_guids(self, node_guid: str) -> list[str]:
        """Return the guids of the children of a node by guid, empty when not found."""

        result = []
        node = self.find_node_by_guid(node_guid)

        if node is None:
            return result

        for child in node.children:
            result.append(child.guid)

        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # Mutators
    # ═══════════════════════════════════════════════════════════════════════════
    def add(self, node: TreeNode, parent: TreeNode | None = None) -> None:
        """Add a node to the tree; a None parent adds it as the root."""

        if node is None:
            raise ValueError("Cannot add null node")

        if parent is not None:
            parent.add(node)

            return

        if self._root is not None:
            raise ValueError("Tree already has a root node")

        self._root = node

    def remove(self, node: TreeNode) -> TreeNode:
        """Remove a node and return it with its subtree intact."""

        if node is None:
            raise ValueError("Cannot remove null node")

        if node is self._root:
            self._root = None

            return node

        parent = node.parent

        if parent is None:
            raise ValueError("Node is not in this tree")

        return parent.remove(node)

    def add_child_by_guid(self, parent_guid: str, child_guid: str) -> bool:
        """Reparent a child by guid; false when either node is missing or the child is the root."""

        parent = self.find_node_by_guid(parent_guid)
        child = self.find_node_by_guid(child_guid)

        if parent is None or child is None:
            return False

        if parent is child:
            return False

        ancestor = parent

        while ancestor is not None:
            if ancestor is child:
                return False

            ancestor = ancestor.parent

        current = child.parent

        if current is None:
            return False

        current.remove(child)
        parent.add(child)

        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # Traversal
    # ═══════════════════════════════════════════════════════════════════════════
    def traverse(
        self, strategy: str = "depthfirst", order: str = "preorder"
    ) -> list[TreeNode]:
        """Traverse from the root ("depthfirst"|"breadthfirst", "preorder"|"postorder")."""

        if self._root is None:
            return []

        return self._root.traverse(strategy, order)

    # ═══════════════════════════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════════════════════════
    def __jsondump__(self) -> dict:
        """Serialize to a JSON object."""

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
        """Deserialize from a JSON object."""

        tree = cls(data["name"])
        tree.guid = guid if guid is not None else data["guid"]
        root = data["root"]

        if isinstance(root, dict):
            root = TreeNode.__jsonload__(root)

        if root is not None:
            tree.add(root)

        return tree

    def file_json_dumps(self) -> str:
        """Serialize to a JSON string."""

        import json

        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, json_string: str) -> Tree:
        """Deserialize from a JSON string."""

        import json

        return cls.__jsonload__(json.loads(json_string))

    def file_json_dump(self, filename: str | Path) -> None:
        """Write to a JSON file."""

        import json

        with open(filename, "w") as file:
            json.dump(self.__jsondump__(), file, indent=4)

    @classmethod
    def file_json_load(cls, filename: str | Path) -> Tree:
        """Read from a JSON file."""

        import json

        with open(filename) as file:
            return cls.__jsonload__(json.load(file))

    # ═══════════════════════════════════════════════════════════════════════════
    # Protobuf
    # ═══════════════════════════════════════════════════════════════════════════
    def pb_dumps(self) -> bytes:
        """Serialize to protobuf bytes."""

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
        """Deserialize from protobuf bytes."""

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
        """Write to a protobuf file."""

        with open(filename, "wb") as file:
            file.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filename: str | Path) -> Tree:
        """Read from a protobuf file."""

        with open(filename, "rb") as file:
            return cls.pb_loads(file.read())

    # ═══════════════════════════════════════════════════════════════════════════
    # String
    # ═══════════════════════════════════════════════════════════════════════════
    def __str__(self) -> str:
        """Return the node count and the hierarchy drawn with box-drawing connectors."""

        text = f"<Tree with {len(self.nodes)} nodes: {self.name}>\n"

        if self._root is not None:
            text += _draw_node(self._root, "", True)

        return text

    def __repr__(self) -> str:
        """Return the tree name and node count."""
        return f"Tree({self.name}, {len(self.nodes)} nodes)"


# ═══════════════════════════════════════════════════════════════════════════
# Node helpers
# ═══════════════════════════════════════════════════════════════════════════
def _draw_node(node: TreeNode, prefix: str, last: bool) -> str:
    """Draw one node and its subtree, the last child of every level closing its branch."""

    text = (
        prefix
        + ("\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 ")
        + str(node)
        + "\n"
    )
    nxt = prefix + ("    " if last else "\u2502   ")
    for i, child in enumerate(node.children):
        text += _draw_node(child, nxt, i + 1 == len(node.children))

    return text


def _node_to_proto(node: TreeNode):
    """Convert a node and its subtree to protobuf."""

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
    """Convert a protobuf node and its subtree to a TreeNode."""

    from .color import Color

    node = TreeNode(proto.name)
    node.guid = proto.guid

    if proto.HasField("color") and proto.color.a > 0:
        node.color = Color(proto.color.r, proto.color.g, proto.color.b, proto.color.a)

    for child in proto.children:
        node.add(_proto_to_node(child))

    return node
