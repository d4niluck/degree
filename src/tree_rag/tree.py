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
    def __init__(
        self,
        root_id: int,
        nodes: Dict[int, TreeNode],
        document_description: Optional[str] = None,
        source_name: Optional[str] = None,
    ):
        self.root_id = root_id
        self.nodes = nodes
        self.document_description = document_description
        self.source_name = source_name
        for node in self.nodes.values():
            node._tree = self

    @property
    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def get_node(self, node_id: int) -> TreeNode:
        return self.nodes[node_id]

    def get_leaf_nodes(self) -> List[TreeNode]:
        ordered: List[TreeNode] = []

        def walk(node_id: int) -> None:
            node = self.get_node(node_id)
            if node.is_leaf:
                ordered.append(node)
                return
            for child_id in node.nodes or []:
                walk(child_id)

        walk(self.root_id)
        return ordered

    def _leaf_idx_map(self) -> Dict[int, int]:
        return {
            node.node_id: index
            for index, node in enumerate(self.get_leaf_nodes(), start=1)
        }

    def max_leaf_idx(self) -> int:
        return len(self.get_leaf_nodes())

    def _node_path_titles(self, node_id: int) -> List[str]:
        path: List[str] = []

        def walk(current_id: int, trail: List[str]) -> bool:
            node = self.get_node(current_id)
            next_trail = trail + [node.title]
            if current_id == node_id:
                path.extend(next_trail)
                return True
            for child_id in node.nodes or []:
                if walk(child_id, next_trail):
                    return True
            return False

        walk(self.root_id, [])
        return path

    def llm_view(
        self,
        node_id: Optional[int] = None,
        labels: Optional[Dict[int, List[str]]] = None,
    ) -> str:
        current_id = self.root_id if node_id is None else node_id
        leaf_idx_map = self._leaf_idx_map()
        labels = labels or {}
        lines: List[str] = []

        def walk(target_id: int, depth: int) -> None:
            node = self.get_node(target_id)
            indent = "  " * depth
            if node.is_leaf:
                idx = leaf_idx_map[node.node_id]
                label_suffix = "".join(f"[{label}]" for label in labels.get(idx, []))
                lines.append(f"{indent}- [{idx}]{label_suffix} {node.title}")
                return

            lines.append(f"{indent}- {node.title}")
            for child_id in node.nodes or []:
                walk(child_id, depth + 1)

        walk(current_id, 0)
        return "\n".join(lines)

    def get_text_by_idx(self, idx: int) -> str:
        leaf_nodes = self.get_leaf_nodes()
        if idx < 1 or idx > len(leaf_nodes):
            raise IndexError(f"Leaf idx out of range: {idx}")
        node = leaf_nodes[idx - 1]
        path = self.get_leaf_path_by_idx(idx)
        return f"Path: {path}\nContent:\n{node.text or ''}"

    def get_leaf_path_by_idx(self, idx: int) -> str:
        leaf_nodes = self.get_leaf_nodes()
        if idx < 1 or idx > len(leaf_nodes):
            raise IndexError(f"Leaf idx out of range: {idx}")
        node = leaf_nodes[idx - 1]
        path_titles = self._node_path_titles(node.node_id)
        source_name = self.source_name or f"{self.root.title}.pdf"
        if path_titles and path_titles[0] == self.root.title:
            path_titles = path_titles[1:]
        return " / ".join([source_name, *path_titles])

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
            "document_description": self.document_description,
            "source_name": self.source_name,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    def save(self, path: str | Path, nested: bool = True) -> None:
        output_path = Path(path)
        if nested:
            payload = {
                "document_description": self.document_description,
                "source_name": self.source_name,
                "tree": self.to_nested_dict(),
            }
        else:
            payload = self.to_flat_dict()
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
        return cls(
            root_id=data["root_id"],
            nodes=nodes,
            document_description=data.get("document_description"),
            source_name=data.get("source_name"),
        )

    @classmethod
    def from_nested_dict(
        cls,
        data: dict,
        document_description: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> "DocumentTree":
        nodes: Dict[int, TreeNode] = {}

        def build(node_data: dict) -> int:
            node = TreeNode.from_dict(node_data)
            child_dicts = node_data.get("nodes") or []
            child_ids = [build(child_data) for child_data in child_dicts]
            node.nodes = child_ids or None
            nodes[node.node_id] = node
            return node.node_id

        root_id = build(data)
        return cls(
            root_id=root_id,
            nodes=nodes,
            document_description=document_description,
            source_name=source_name,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DocumentTree":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "tree" in payload:
            return cls.from_nested_dict(
                payload["tree"],
                document_description=payload.get("document_description"),
                source_name=payload.get("source_name"),
            )
        return cls.from_flat_dict(payload)
