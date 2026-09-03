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
            raw_factor = match.group("factor").strip()

            factor = re.sub(
                r"^(?:i|we|the patient)\s+",
                "",
                raw_factor,
                flags=re.IGNORECASE,
            ).strip()

            # Truncate at clause breaks (e.g. "I ", "and ", "but ", "don't", "take ")
            clause_split = re.split(
                r"\s+(?:I|we|they|and|but|don't|does|doesn't|denies|not|take|taking)\b",
                factor,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            clean_factor = clause_split[0].strip()

            if clean_factor:
                factor_start = match.start("factor")
                factor_end = factor_start + len(clean_factor)
                entities.append({
                    "text": clean_factor,
                    "label": "Aggravating_factor",
                    "start": factor_start,
                    "end": factor_end,
                    "source": "rule",
                    "evidence": match.group().strip(),
                })

    return entities