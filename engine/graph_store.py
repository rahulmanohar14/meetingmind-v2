"""NetworkX knowledge graph store for extracted entities and relations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalise_name(name: str) -> str:
    text = name.lower().strip()
    text = _PUNCT.sub("", text)
    text = _LEADING_ARTICLE.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def build_graph(extractions: dict) -> nx.DiGraph:
    """Build a directed graph with entity resolution by normalised name+type."""
    entities = extractions.get("entities") or []
    relations = extractions.get("relations") or []

    # Map raw extraction ids -> canonical node ids after resolution.
    id_map: dict[str, str] = {}
    # canonical_id -> node payload being assembled
    nodes: dict[str, dict] = {}
    # (type, normalised_name) -> canonical_id
    key_to_canonical: dict[tuple[str, str], str] = {}

    for ent in entities:
        eid = ent["id"]
        etype = ent["type"]
        name = ent["name"]
        source_turn = ent["source_turn"]
        key = (etype, normalise_name(name))

        if key in key_to_canonical:
            canonical = key_to_canonical[key]
            nodes[canonical]["source_turns"].append(source_turn)
            # Prefer a cleaner display name if the stored one is shorter/worse?
            # Keep the first-seen name.
        else:
            canonical = eid
            # Collision: another entity already uses this id for a different key.
            if canonical in nodes:
                canonical = f"{etype.lower()}:{normalise_name(name)}"
            key_to_canonical[key] = canonical
            nodes[canonical] = {
                "type": etype,
                "name": name,
                "source_turns": [source_turn],
            }
        id_map[eid] = key_to_canonical[key]

    # Deduplicate source_turns while preserving order.
    for node in nodes.values():
        seen: set[str] = set()
        unique: list[str] = []
        for tid in node["source_turns"]:
            if tid not in seen:
                seen.add(tid)
                unique.append(tid)
        node["source_turns"] = unique

    g = nx.DiGraph()
    for nid, attrs in nodes.items():
        g.add_node(nid, **attrs)

    # Merge parallel edges of the same type between the same endpoints.
    edge_acc: dict[tuple[str, str, str], list[str]] = {}
    for rel in relations:
        src = id_map.get(rel["source"])
        tgt = id_map.get(rel["target"])
        if src is None or tgt is None:
            continue
        if src not in g or tgt not in g:
            continue
        rtype = rel["type"]
        key = (src, tgt, rtype)
        edge_acc.setdefault(key, []).append(rel["source_turn"])

    for (src, tgt, rtype), turns in edge_acc.items():
        seen: set[str] = set()
        unique: list[str] = []
        for tid in turns:
            if tid not in seen:
                seen.add(tid)
                unique.append(tid)
        g.add_edge(src, tgt, type=rtype, source_turns=unique)

    return g


def save_graph(g: nx.DiGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_graph.node_link_data(g)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_graph(path: str | Path) -> nx.DiGraph:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    g = json_graph.node_link_graph(data, directed=True)
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)
    return g


def find_entities(g: nx.DiGraph, query: str) -> list[str]:
    """Return entity ids by normalised substring match (either direction)."""
    q = normalise_name(query)
    if not q:
        return []
    hits: list[str] = []
    for nid, attrs in g.nodes(data=True):
        name = normalise_name(str(attrs.get("name", "")))
        nid_norm = normalise_name(str(nid))
        if name and (q in name or (len(name) >= 3 and name in q)):
            hits.append(nid)
        elif nid_norm and (q in nid_norm or (len(nid_norm) >= 3 and nid_norm in q)):
            hits.append(nid)
    return hits


def local_search(
    g: nx.DiGraph,
    query_entities: list[str],
    hops: int = 2,
) -> tuple[nx.DiGraph, list[str]]:
    """Traverse up to `hops` from query entities; return subgraph and fact strings."""
    seeds = [nid for nid in query_entities if nid in g]
    if not seeds or hops < 0:
        return nx.DiGraph(), []

    keep: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt: set[str] = set()
        for nid in frontier:
            nxt.update(g.successors(nid))
            nxt.update(g.predecessors(nid))
        nxt -= keep
        keep.update(nxt)
        frontier = nxt
        if not frontier:
            break

    sub = g.subgraph(keep).copy()
    facts: list[str] = []
    for src, tgt, attrs in sub.edges(data=True):
        src_name = sub.nodes[src].get("name", src)
        tgt_name = sub.nodes[tgt].get("name", tgt)
        rtype = attrs.get("type", "related_to")
        turns = attrs.get("source_turns") or []
        cite = turns[0] if turns else "unknown"
        facts.append(f"{src_name} {rtype} {tgt_name} ({cite})")
    return sub, facts
