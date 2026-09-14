"""LLM entity/relation extraction from transcript turns."""

from __future__ import annotations

import json

from engine.corpus import Turn, turn_id
from engine.llm import complete

BATCH_SIZE = 15

ENTITY_TYPES = ("Person", "Project", "Decision", "ActionItem", "Blocker")
RELATION_TYPES = ("owns", "blocks", "depends_on", "decided_in")

SYSTEM = (
    "You extract entities and relations from meeting transcript turns. "
    "Extract only what is explicitly stated. Do not invent entities or "
    "relationships. If nothing is explicit, return empty lists. "
    "Precision matters more than recall. "
    "Respond with a single valid JSON object only — no markdown, no commentary."
)


def extract_from_turns(turns: list[Turn]) -> dict:
    """Extract entities and relations from all turns in batches of 15."""
    entities: list[dict] = []
    relations: list[dict] = []

    for start in range(0, len(turns), BATCH_SIZE):
        batch = turns[start : start + BATCH_SIZE]
        payload = _extract_batch(batch)
        entities.extend(payload.get("entities") or [])
        relations.extend(payload.get("relations") or [])

    return {"entities": entities, "relations": relations}


def _extract_batch(turns: list[Turn]) -> dict:
    turn_lines = []
    valid_ids = []
    for t in turns:
        tid = turn_id(t)
        valid_ids.append(tid)
        turn_lines.append(
            f"- source_turn={tid} | speaker={t.speaker} | text={t.text}"
        )

    prompt = (
        "Extract entities and relations from these transcript turns.\n\n"
        f"Turns:\n{chr(10).join(turn_lines)}\n\n"
        "Entity types (use exactly these strings): "
        f"{', '.join(ENTITY_TYPES)}\n"
        "Relation types (use exactly these strings): "
        f"{', '.join(RELATION_TYPES)}\n\n"
        "Rules:\n"
        "- Only extract explicitly stated entities and relations.\n"
        "- Do not invent relationships. Prefer empty lists over guesses.\n"
        "- Every entity and every relation MUST include source_turn set to "
        "one of the turn ids listed above.\n"
        "- Entity id should be a short stable slug like 'person_alice' or "
        "'blocker_datacorp_contract'.\n"
        "- Relation source and target must be entity ids from this response.\n"
        "- Output must be valid JSON parseable by json.loads.\n"
        "- Do not include any text before or after the JSON object.\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "entities": [{"id": str, "type": str, "name": str, "source_turn": str}],\n'
        '  "relations": [{"source": str, "target": str, "type": str, "source_turn": str}]\n'
        "}"
    )

    raw = complete(prompt, system=SYSTEM, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Prompt revision path: one retry with an explicit repair instruction
        # (different prompt => different cache key; avoids a poisoned cache hit).
        repair_prompt = (
            prompt
            + "\n\nYour previous answer was invalid JSON. "
            "Return ONLY a valid JSON object matching the schema above."
        )
        raw = complete(repair_prompt, system=SYSTEM, json_mode=True)
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object from the model, got: {type(data)}")

    valid_id_set = set(valid_ids)
    entities = []
    for item in data.get("entities") or []:
        if not isinstance(item, dict):
            continue
        eid = item.get("id")
        etype = item.get("type")
        name = item.get("name")
        source_turn = item.get("source_turn")
        if not isinstance(eid, str) or not eid.strip():
            continue
        if etype not in ENTITY_TYPES:
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        if source_turn not in valid_id_set:
            continue
        entities.append(
            {
                "id": eid.strip(),
                "type": etype,
                "name": name.strip(),
                "source_turn": source_turn,
            }
        )

    entity_ids = {e["id"] for e in entities}
    relations = []
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        target = item.get("target")
        rtype = item.get("type")
        source_turn = item.get("source_turn")
        if source not in entity_ids or target not in entity_ids:
            continue
        if rtype not in RELATION_TYPES:
            continue
        if source_turn not in valid_id_set:
            continue
        relations.append(
            {
                "source": source,
                "target": target,
                "type": rtype,
                "source_turn": source_turn,
            }
        )

    return {"entities": entities, "relations": relations}
