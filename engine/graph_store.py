"""NetworkX knowledge graph store for extracted entities and relations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")

# Question scaffolding and generic meeting vocabulary. These tokens carry no
# entity signal, so they must not seed a traversal.
_STOPWORDS = frozenset(
    """
    a an the and or but of to for in on at by with from as is are was were be been
    being do does did done doing has have had can could will would shall should may
    might must what who whom whose when where why how which that this these those
    there here it its their our your my me we they he she him her his them us you i
    about into over under after before between during against without within if then
    than so such no not any some more most other another each both few all only own
    same too very just also still yet again once because while until up down out off
    team teams meeting meetings week weeks day days date dates work working plan plans
    status update updates thing things item items said say says tell said according
    anything something everything happen happens happened change changed changes
    """.split()
)


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


def graph_to_extractions(g: nx.DiGraph) -> dict:
    """Flatten a graph back into entities/relations for rebuild/merge."""
    entities: list[dict] = []
    relations: list[dict] = []
    for nid, attrs in g.nodes(data=True):
        turns = attrs.get("source_turns") or ["unknown"]
        for st in turns:
            entities.append(
                {
                    "id": str(nid),
                    "type": attrs.get("type", "Project"),
                    "name": attrs.get("name", str(nid)),
                    "source_turn": st,
                }
            )
    for src, tgt, attrs in g.edges(data=True):
        turns = attrs.get("source_turns") or ["unknown"]
        for st in turns:
            relations.append(
                {
                    "source": str(src),
                    "target": str(tgt),
                    "type": attrs.get("type", "depends_on"),
                    "source_turn": st,
                }
            )
    return {"entities": entities, "relations": relations}


def merge_extractions(g: nx.DiGraph, extractions: dict) -> nx.DiGraph:
    """Merge new extractions into an existing graph via entity resolution."""
    existing = graph_to_extractions(g)
    combined = {
        "entities": existing["entities"] + list(extractions.get("entities") or []),
        "relations": existing["relations"] + list(extractions.get("relations") or []),
    }
    return build_graph(combined)


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


def content_tokens(text: str) -> list[str]:
    """Normalised content tokens: no punctuation, no stopwords, no 1-char tokens."""
    return [
        tok
        for tok in normalise_name(text).split()
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _name_token_sets(g: nx.DiGraph) -> dict[str, set[str]]:
    tokens: dict[str, set[str]] = {}
    for nid, attrs in g.nodes(data=True):
        name_tokens = set(content_tokens(str(attrs.get("name", ""))))
        if not name_tokens:
            # Fall back to the node id so unnamed nodes stay reachable.
            name_tokens = set(content_tokens(str(nid)))
        if name_tokens:
            tokens[nid] = name_tokens
    return tokens


def find_entities(g: nx.DiGraph, query: str, limit: int = 8) -> list[str]:
    """Return seed entity ids ranked by IDF-weighted token overlap with the query.

    Phrase-level substring matching fails on this corpus because the extractor
    emits sentence-shaped names ("DataCorp vendor contract is still unsigned"),
    which never appear verbatim inside a user question. Token overlap matches
    those names; IDF weighting stops tokens that are spread across many nodes
    (such as "launch") from seeding every traversal on their own.
    """
    q_tokens = set(content_tokens(query))
    if not q_tokens:
        return []

    node_tokens = _name_token_sets(g)
    if not node_tokens:
        return []

    total = len(node_tokens)
    doc_freq: dict[str, int] = {}
    for tokens in node_tokens.values():
        for tok in tokens:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    scored: list[tuple[float, float, str]] = []
    for nid, tokens in node_tokens.items():
        overlap = q_tokens & tokens
        if not overlap:
            continue
        weight = sum(
            math.log(1.0 + total / (1.0 + doc_freq.get(tok, 0))) for tok in overlap
        )
        if weight <= 0.0:
            continue
        # Tie-break toward names the question covers most completely, so
        # "DataCorp vendor contract" outranks the bare token "contract".
        precision = len(overlap) / len(tokens)
        scored.append((weight + precision, precision, nid))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [nid for _weight, _precision, nid in scored[:limit]]


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
