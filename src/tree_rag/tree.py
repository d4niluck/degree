from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TreeNode:
    title: str
    node_id: int
    pages: Optional[List[int]] = None
    nodes: Optional[List[int]] = None
    text: Optional[str] = None
    text_type: Optional[str] = None
    _tree: Optional["DocumentTree"] = field(default=None, repr=False, compare=False)

    @property
    def is_leaf(self) -> bool:
        return self.text is not None

    @property
    def preview_title(self) -> str:
        if self.is_leaf:
            return f"{self.title} [{len(self.text or '')} chars]"
        return self.title

    def preview(self, depth: Optional[int] = None) -> str:
        if self._tree is None:
            raise ValueError("TreeNode is not attached to a DocumentTree")

        lines = [self.preview_title]
        child_ids = self.nodes or []

        if depth == 0 or not child_ids:
            return "\n".join(lines)

        max_depth = None if depth is None else depth - 1
        for index, child_id in enumerate(child_ids):
            child = self._tree.get_node(child_id)
            is_last = index == len(child_ids) - 1
            lines.extend(child._preview_lines(prefix="", is_last=is_last, depth=max_depth))

        return "\n".join(lines)

    def _preview_lines(self, prefix: str, is_last: bool, depth: Optional[int]) -> List[str]:
        branch = "└── " if is_last else "├── "
        lines = [f"{prefix}{branch}{self.preview_title}"]

        child_ids = self.nodes or []
        if depth == 0 or not child_ids:
            return lines

        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        next_depth = None if depth is None else depth - 1
        for index, child_id in enumerate(child_ids):
            child = self._tree.get_node(child_id)
            child_is_last = index == len(child_ids) - 1
            lines.extend(child._preview_lines(child_prefix, child_is_last, next_depth))

        return lines

    def to_dict(self) -> dict:
        data = {
            "title": self.title,
            "node_id": self.node_id,
            "pages": self.pages,
            "nodes": self.nodes,
            "text": self.text,
            "text_type": self.text_type,
        }
        return {key: value for key, value in data.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        return cls(
            title=data["title"],
            node_id=data["node_id"],
            pages=data.get("pages"),
            nodes=data.get("nodes"),
            text=data.get("text"),
            text_type=data.get("text_type"),
        )


class DocumentTree:
    def __init__(self, root_id: int, nodes: Dict[int, TreeNode]):
        self.root_id = root_id
        self.nodes = nodes
        for node in self.nodes.values():
            node._tree = self

    @property
    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def get_node(self, node_id: int) -> TreeNode:
        return self.nodes[node_id]

    def collapse_single_leaf_children(self) -> None:
        def collapse(node_id: int) -> None:
            node = self.get_node(node_id)
            child_ids = node.nodes or []
            for child_id in child_ids:
                collapse(child_id)

            child_ids = node.nodes or []
            if len(child_ids) != 1 or node.text is not None:
                return

            child = self.get_node(child_ids[0])
            if not child.is_leaf:
                return

            node.text = child.text
            node.text_type = child.text_type
            node.nodes = None
            if child.pages:
                node.pages = child.pages
            self.nodes.pop(child.node_id, None)

        collapse(self.root_id)

    def to_nested_dict(self, node_id: Optional[int] = None) -> dict:
        current_id = self.root_id if node_id is None else node_id
        node = self.get_node(current_id)
        data = node.to_dict()

        child_ids = node.nodes or []
        if child_ids:
            data["nodes"] = [self.to_nested_dict(child_id) for child_id in child_ids]

        return data

    def to_flat_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    def save(self, path: str | Path, nested: bool = True) -> None:
        output_path = Path(path)
        payload = self.to_nested_dict() if nested else self.to_flat_dict()
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_flat_dict(cls, data: dict) -> "DocumentTree":
        nodes = {
            int(node_id): TreeNode.from_dict(node_data)
            for node_id, node_data in data["nodes"].items()
        }
        return cls(root_id=data["root_id"], nodes=nodes)
