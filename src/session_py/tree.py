from typing import Any
from typing import Iterator
from typing import Optional
from typing import Union
from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from pathlib import Path


class TreeNode:
    """A node of a tree data structure.

    TreeNodes can represent either:
    - Geometry nodes: name is set to the geometry's GUID for lookup
    - Organizational nodes: name is a descriptive string (e.g., "folder", "group")

    When adding geometry to a Session, the TreeNode.name is automatically set to
    the geometry.guid, allowing the tree hierarchy to reference geometry objects.

    Parameters
    ----------
    name : str, optional
        The name of the tree node. For geometry nodes, this should be the geometry's GUID.
        For organizational nodes, this can be any descriptive string.

    Attributes
    ----------
    name : str
        The name of the tree node. For geometry nodes, this is the geometry's GUID.
    guid : UUID
        The unique identifier of the tree node itself (distinct from geometry GUID).
    parent : :class:`TreeNode`
        The parent node of the tree node.
    children : list[:class:`TreeNode`]
        The children of the tree node.

    """

    def __init__(self, name: str = "my_node"):
        self.name = name
        self._guid = None
        self.color = None
        self._parent = None
        self._children = []
        self._tree = None

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def __str__(self):
        """String representation."""
        return f"TreeNode({self.name}"

    def __repr__(self):
        return f"TreeNode({self.name}, {self.guid}, {len(self.children)} children)"

    ###########################################################################################
    # JSON (polymorphic)
    ###########################################################################################

    def __jsondump__(self) -> dict:
        """Serialize to polymorphic JSON format with type field."""
        d = {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "children": [child.__jsondump__() for child in self.children],
        }
        if self.color is not None:
            d["color"] = self.color.__jsondump__()
        return d

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: Optional[str] = None, name: Optional[str] = None
    ) -> "TreeNode":
        """Deserialize from polymorphic JSON format."""
        from .color import Color
        node = cls(name=data["name"])
        node.guid = guid if guid is not None else data.get("guid", node.guid)
        if "color" in data and data["color"] is not None:
            node.color = Color.__jsonload__(data["color"])
        for child_data in data.get("children", []):
            # Children are polymorphic nodes themselves
            from .file_encoders import file_decode_node

            child_node = file_decode_node(child_data)
            node.add(child_node)
        return node

    ###########################################################################################
    # Details
    ###########################################################################################

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def is_leaf(self) -> bool:
        return not self._children

    @property
    def is_branch(self) -> bool:
        return not self.is_root and not self.is_leaf

    @property
    def parent(self) -> Optional["TreeNode"]:
        return self._parent

    @property
    def children(self) -> list:
        return self._children

    @property
    def tree(self) -> Optional["Tree"]:
        if self.is_root:
            return self._tree
        else:
            return self.parent.tree  # type: ignore

    def add(self, node: "TreeNode") -> None:
        """Add a child node to this node.

        Parameters
        ----------
        node : :class:`TreeNode`
            The node to add.

        """
        if not isinstance(node, TreeNode):
            raise TypeError("The node is not a TreeNode object.")
        if node not in self._children:
            self._children.append(node)
        node._parent = self

    def remove(self, node: "TreeNode") -> None:
        """Remove a child node from this node.

        Parameters
        ----------
        node : :class:`TreeNode`
            The node to remove.

        """
        self._children.remove(node)
        node._parent = None

    @property
    def ancestors(self) -> Iterator["TreeNode"]:
        this = self
        while this.parent:
            yield this.parent
            this = this.parent

    @property
    def descendants(self) -> Iterator["TreeNode"]:
        for child in self.children:
            yield child
            for descendant in child.descendants:
                yield descendant

    def traverse(self, strategy: str = "depthfirst", order: str = "preorder") -> Iterator["TreeNode"]:
        """Traverse the tree from this node.

        Parameters
        ----------
        strategy : {"depthfirst", "breadthfirst"}, optional
            The traversal strategy.
        order : {"preorder", "postorder"}, optional
            The traversal order.

        """
        if strategy == "depthfirst":
            if order == "preorder":
                yield self
                for child in self.children:
                    for node in child.traverse(strategy, order):
                        yield node
            elif order == "postorder":
                for child in self.children:
                    for node in child.traverse(strategy, order):
                        yield node
                yield self
            else:
                raise ValueError("Unknown traversal order: {}".format(order))
        elif strategy == "breadthfirst":
            queue = [self]
            while queue:
                node = queue.pop(0)
                yield node
                queue.extend(node.children)
        else:
            raise ValueError("Unknown traversal strategy: {}".format(strategy))


class Tree:
    """A hierarchical data structure with parent-child relationships.

    Parameters
    ----------
    name : str, optional
        The name of the tree. Defaults to "Tree".

    Attributes
    ----------
    guid : UUID
        The unique identifier of the tree.
    name : str
        The name of the tree.
    root : :class:`TreeNode`
        The root node of the tree.

    """

    def __init__(self, name: str = "my_tree"):
        self._guid = None
        self.name = name
        self._root = None

    @property
    def guid(self) -> str:
        if getattr(self, '_guid', None) is None:
            self._guid = str(uuid.uuid4())
        return self._guid

    @guid.setter
    def guid(self, value: str) -> None:
        self._guid = value

    def __str__(self):
        return "<Tree with {} nodes>".format(len(list(self.nodes)))

    def __repr__(self):
        return "<Tree with {} nodes>".format(len(list(self.nodes)))

    ###########################################################################################
    # JSON (polymorphic)
    ###########################################################################################

    def __jsondump__(self) -> dict:
        """Serialize to polymorphic JSON format with type field."""
        return {
            "type": f"{self.__class__.__name__}",
            "guid": self.guid,
            "name": self.name,
            "root": self.root.__jsondump__() if self.root else None,
        }

    @classmethod
    def __jsonload__(
        cls, data: dict, guid: Optional[str] = None, name: Optional[str] = None
    ) -> "Tree":
        """Deserialize from polymorphic JSON format."""
        tree = cls(name=data.get("name", "Tree"))
        tree.guid = guid if guid is not None else data.get("guid", tree.guid)
        if data.get("root"):
            from .file_encoders import file_decode_node

            root = file_decode_node(data["root"])
            tree.add(root)
        return tree

    def file_json_dumps(self) -> str:
        import json
        return json.dumps(self.__jsondump__())

    @classmethod
    def file_json_loads(cls, s: str) -> "Tree":
        import json
        return cls.__jsonload__(json.loads(s))

    def file_json_dump(self, filepath: Union[str, "Path"]) -> None:
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__jsondump__(), f, indent=2)

    @classmethod
    def file_json_load(cls, filepath: Union[str, "Path"]) -> "Tree":
        import json
        with open(filepath, 'r') as f:
            return cls.__jsonload__(json.load(f))

    def pb_dumps(self) -> bytes:
        from .proto import tree_pb2

        def fill_node(proto_node, node):
            proto_node.guid = node.guid
            proto_node.name = node.name
            proto_node.parent_guid = ""
            if node.color is not None:
                proto_node.color.r = node.color[0]
                proto_node.color.g = node.color[1]
                proto_node.color.b = node.color[2]
                proto_node.color.a = node.color[3]
            for child in node.children:
                fill_node(proto_node.children.add(), child)

        proto = tree_pb2.Tree()
        proto.guid = self.guid
        proto.name = self.name
        if self.root:
            fill_node(proto.root, self.root)
        return proto.SerializeToString()

    @classmethod
    def pb_loads(cls, data: bytes) -> "Tree":
        from .proto import tree_pb2

        proto = tree_pb2.Tree()
        proto.ParseFromString(data)

        def proto_to_node(proto_node):
            from .color import Color
            node = TreeNode(name=proto_node.name)
            node.guid = proto_node.guid
            if proto_node.HasField("color") and proto_node.color.a > 0:
                node.color = Color(proto_node.color.r, proto_node.color.g,
                                   proto_node.color.b, proto_node.color.a)
            for child_proto in proto_node.children:
                child = proto_to_node(child_proto)
                node.add(child)
            return node

        tree = cls(name=proto.name)
        tree.guid = proto.guid
        if proto.HasField('root'):
            root = proto_to_node(proto.root)
            tree._root = root
            root._tree = tree
        return tree

    def pb_dump(self, filepath: Union[str, "Path"]) -> None:
        with open(filepath, 'wb') as f:
            f.write(self.pb_dumps())

    @classmethod
    def pb_load(cls, filepath: Union[str, "Path"]) -> "Tree":
        with open(filepath, 'rb') as f:
            return cls.pb_loads(f.read())

    def find_node_by_guid(self, guid: str) -> Optional["TreeNode"]:
        for node in self.nodes:
            if node.guid == guid:
                return node
        return None

    ###########################################################################################
    # Details
    ###########################################################################################

    @property
    def root(self) -> Optional["TreeNode"]:
        return self._root

    def add(self, node: "TreeNode", parent: Optional["TreeNode"] = None) -> None:
        """Add a node to the tree.

        Parameters
        ----------
        node : :class:`TreeNode`
            The node to add.
        parent : :class:`TreeNode`, optional
            The parent node. If None, adds as root.

        """
        if not isinstance(node, TreeNode):
            raise TypeError("The node is not a TreeNode object.")

        if node.parent:
            raise ValueError(
                "The node already has a parent, remove it from that parent first."
            )

        if parent is None:
            # add the node as a root node
            if self.root is not None:
                raise ValueError("The tree already has a root node, remove it first.")

            self._root = node
            node._tree = self  # type: ignore

        else:
            # add the node as a child of the parent node
            if not isinstance(parent, TreeNode):
                raise TypeError("The parent node is not a TreeNode object.")

            if parent.tree is not self:
                raise ValueError("The parent node is not part of this tree.")

            parent.add(node)

    @property
    def nodes(self) -> Iterator["TreeNode"]:
        if self.root:
            for node in self.root.traverse():
                yield node

    def remove(self, node: "TreeNode") -> None:
        """Remove a node from the tree.

        Parameters
        ----------
        node : :class:`TreeNode`
            The node to remove.

        """
        if node == self.root:
            self._root = None
            node._tree = None
        else:
            node.parent.remove(node)

    @property
    def leaves(self) -> Iterator["TreeNode"]:
        for node in self.nodes:
            if node.is_leaf:
                yield node

    def traverse(self, strategy: str = "depthfirst", order: str = "preorder") -> Iterator["TreeNode"]:
        """
        Traverse the tree from the root node.

        Parameters
        ----------
        strategy : {"depthfirst", "breadthfirst"}, optional
            The traversal strategy.
        order : {"preorder", "postorder"}, optional
            The traversal order. This parameter is only used for depth-first traversal.

        Yields
        ------
        :class:`TreeNode`
            The next node in the traversal.

        Raises
        ------
        ValueError
            If the strategy is not ``"depthfirst"`` or ``"breadthfirst"``.
            If the order is not ``"preorder"`` or ``"postorder"``.

        """
        if self.root:
            for node in self.root.traverse(strategy=strategy, order=order):
                yield node

    def get_node_by_name(self, name: str) -> Optional["TreeNode"]:
        """Get a node by its name.

        Parameters
        ----------
        name : str
            The name of the node.

        """
        for node in self.nodes:
            if node.name == name:
                return node

    def get_nodes_by_name(self, name: str) -> list["TreeNode"]:
        """
        Get all nodes by their name.

        Parameters
        ----------
        name : str
            The name of the node.

        Returns
        -------
        list[:class:`TreeNode`]
            The nodes.

        """
        nodes = []
        for node in self.nodes:
            if node.name == name:
                nodes.append(node)
        return nodes

    def add_child_by_guid(self, parent_guid: uuid.UUID, child_guid: uuid.UUID) -> bool:
        """
        Add a parent-child relationship using GUIDs.

        Parameters
        ----------
        parent_guid : UUID
            The GUID of the parent node.
        child_guid : UUID
            The GUID of the child node.

        Returns
        -------
        bool
            True if the relationship was added, False if nodes not found.
        """
        parent_node = self.find_node_by_guid(parent_guid)
        child_node = self.find_node_by_guid(child_guid)

        if not parent_node or not child_node:
            return False

        # Remove child from its current parent if it has one
        if child_node.parent:
            child_node.parent.remove(child_node)

        # Add to new parent
        parent_node.add(child_node)
        return True

    def get_children_guids(self, guid: uuid.UUID) -> list[uuid.UUID]:
        """
        Get all children GUIDs of a node by its GUID.

        Parameters
        ----------
        guid : UUID
            The GUID of the parent node.

        Returns
        -------
        list[UUID]
            List of children GUIDs.
        """
        node = self.find_node_by_guid(guid)
        if not node:
            return []

        return [child.guid for child in node.children if isinstance(child, TreeNode)]

    def print_hierarchy(self) -> None:
        """Print the spatial hierarchy of the tree."""

        def _print(node, prefix="", last=True):
            connector = "└── " if last else "├── "
            print("{}{}{}".format(prefix, connector, node))
            prefix += "    " if last else "│   "
            for i, child in enumerate(node.children):
                _print(child, prefix, i == len(node.children) - 1)

        if self.root:
            _print(self.root)
        else:
            print("Empty tree")
