import re


AGGRAVATING_PATTERNS = [
    re.compile(
        r"\b(?:gets?|becomes?|is)\s+worse\s+"
        r"(?:when|while|with|during)\s+"
        r"(?P<factor>[^.,;]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:yes[,\s]+)?especially\s+"
        r"(?:when|while|with|during)\s+"
        r"(?P<factor>[^.,;]+)",
        re.IGNORECASE,
    ),
]


def extract_context_entities(text: str) -> list[dict]:
    entities = []

    for pattern in AGGRAVATING_PATTERNS:
        for match in pattern.finditer(text):
            factor = match.group("factor").strip()

            factor = re.sub(
                r"^(?:i|we|the patient)\s+",
                "",
                factor,
                flags=re.IGNORECASE,
            ).strip()

            entities.append({
                "text": factor,
                "label": "Aggravating_factor",
                "start": match.start("factor"),
                "end": match.end("factor"),
                "source": "rule",
                "evidence": match.group().strip(),
            })

    return entities