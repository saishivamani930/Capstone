import re


NEGATABLE_LABELS = {
    "Sign_symptom",
    "Disease_disorder",
    "Medication",
    "Allergy",
    "Diagnostic_procedure",
    "Therapeutic_procedure",
    "Aggravating_factor",
}


NEGATION_PATTERNS = [
    # Examples:
    # no fever
    # without chest pain
    # denies diabetes
    # negative for infection
    re.compile(
        r"\b(?:"
        r"no|without|denies|denied|deny|"
        r"negative\s+for|free\s+of"
        r")\b"
        r"(?:\s+\w+){0,8}\s*$",
        re.IGNORECASE,
    ),

    # Examples:
    # do not have diabetes
    # am not allergic
    # has not experienced pain
    re.compile(
        r"\b(?:"
        r"do|does|did|have|has|had|"
        r"am|is|are|was|were|"
        r"can|could|would|should"
        r")\s+not\b"
        r"(?:\s+\w+){0,8}\s*$",
        re.IGNORECASE,
    ),

    # Examples:
    # don't have diabetes
    # isn't allergic
    # haven't experienced pain
    re.compile(
        r"\b(?:"
        r"don't|doesn't|didn't|"
        r"haven't|hasn't|hadn't|"
        r"isn't|aren't|wasn't|weren't|"
        r"can't|cannot|couldn't|wouldn't|shouldn't"
        r")\b"
        r"(?:\s+\w+){0,8}\s*$",
        re.IGNORECASE,
    ),

    # Examples:
    # I'm not allergic
    # Im not allergic
    # we're not experiencing pain
    re.compile(
        r"\b(?:"
        r"i['’]?m|"
        r"you['’]?re|"
        r"we['’]?re|"
        r"they['’]?re|"
        r"he['’]?s|"
        r"she['’]?s|"
        r"it['’]?s"
        r")\s+not\b"
        r"(?:\s+\w+){0,8}\s*$",
        re.IGNORECASE,
    ),

    # Example:
    # never had chest pain
    re.compile(
        r"\bnever\b"
        r"(?:\s+\w+){0,8}\s*$",
        re.IGNORECASE,
    ),
]


def get_relevant_prefix(
    text: str,
    entity_start: int,
) -> str:
    """
    Return only the text relevant to the entity's assertion.

    This prevents negation from one sentence or clause from
    incorrectly affecting another clinical entity.
    """
    prefix = text[:entity_start]

    # Negation must not cross sentence boundaries.
    sentence_boundaries = [
        prefix.rfind("."),
        prefix.rfind("?"),
        prefix.rfind("!"),
        prefix.rfind(";"),
    ]

    latest_boundary = max(sentence_boundaries)
    prefix = prefix[latest_boundary + 1:]

    # Contrast words usually begin a new assertion.
    contrast_matches = list(
        re.finditer(
            r"\b(?:but|however|although|though|yet)\b",
            prefix,
            re.IGNORECASE,
        )
    )

    if contrast_matches:
        prefix = prefix[
            contrast_matches[-1].end():
        ]

    # Keep only the latest assertion beginning with a subject.
    #
    # Example:
    # "I'm not allergic to penicillin I take aspirin"
    #
    # For penicillin:
    #     relevant prefix = "I'm not allergic to"
    #
    # For aspirin:
    #     relevant prefix = "I take"
    clause_pattern = re.compile(
        r"\b(?:i|we|you|he|she|they)\s+"
        r"(?:"
        r"am|are|is|was|were|"
        r"do|does|did|"
        r"have|has|had|"
        r"take|takes|took|taking|"
        r"feel|feels|felt|feeling|"
        r"experience|experiences|experienced|experiencing"
        r")\b"
        r"|"
        r"\b(?:"
        r"i['’]?m|"
        r"you['’]?re|"
        r"we['’]?re|"
        r"they['’]?re|"
        r"he['’]?s|"
        r"she['’]?s|"
        r"it['’]?s"
        r")\b",
        re.IGNORECASE,
    )

    clause_matches = list(
        clause_pattern.finditer(prefix)
    )

    if clause_matches:
        latest_clause = clause_matches[-1]
        prefix = prefix[latest_clause.start():]

    return prefix.strip()


def find_negation(
    text: str,
    entity: dict,
) -> dict:
    """
    Determine whether a single clinical entity is negated.
    """
    result = {
        **entity,
        "negated": False,
        "assertion": "present",
    }

    if entity.get("label") not in NEGATABLE_LABELS:
        return result

    entity_start = entity.get("start")

    if not isinstance(entity_start, int):
        return result

    prefix = get_relevant_prefix(
        text,
        entity_start,
    )

    # "Not only" does not indicate clinical absence.
    if re.search(
        r"\bnot\s+only\b",
        prefix,
        re.IGNORECASE,
    ):
        return result

    for pattern in NEGATION_PATTERNS:
        match = pattern.search(prefix)

        if match:
            result["negated"] = True
            result["assertion"] = "absent"
            result["negation_evidence"] = (
                match.group().strip()
            )
            break

    return result


def apply_negation(
    text: str,
    entities: list[dict],
) -> list[dict]:
    """
    Apply negation detection to every clinical entity.
    """
    return [
        find_negation(text, entity)
        for entity in entities
    ]