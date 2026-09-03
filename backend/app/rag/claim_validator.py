import re


# --------------------------------------------------
# Normalization
# --------------------------------------------------

def normalize(text):
    text = text.lower()
    text = text.replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# BP extraction
# --------------------------------------------------

def extract_bp_conditions(text):

    patterns = [
        r"130\s*[–-]\s*139\s*mmHg",
        r"≥\s*140\s*mmHg",
        r"≥\s*90\s*mmHg",
        r"≥\s*160\s*mmHg",
        r"≥\s*100\s*mmHg"
    ]

    found = []

    for pattern in patterns:

        for match in re.findall(pattern, text):

            value = re.sub(
                r"\s+",
                " ",
                match
            ).strip()

            if value not in found:
                found.append(value)

    return found


# --------------------------------------------------
# Numerical validation
# --------------------------------------------------

def validate_numerical_claims(answer, evidence):

    answer_values = extract_bp_conditions(answer)
    evidence_normalized = normalize(evidence)

    results = []

    for value in answer_values:

        supported = (
            normalize(value)
            in evidence_normalized
        )

        results.append({
            "claim": value,
            "type": "NUMERICAL",
            "status": (
                "SUPPORTED"
                if supported
                else "UNSUPPORTED"
            )
        })

    return results


# --------------------------------------------------
# Timing validation
# --------------------------------------------------

def validate_timing_claims(answer, evidence):

    answer_n = normalize(answer)
    evidence_n = normalize(evidence)

    results = []

    # Four weeks
    if (
        "four weeks" in answer_n
        or "4 weeks" in answer_n
    ):

        supported = (
            "four weeks" in evidence_n
            or "4 weeks" in evidence_n
        )

        results.append({
            "claim": "Treatment should start no later than four weeks.",
            "type": "TIMING",
            "status": (
                "SUPPORTED"
                if supported
                else "UNSUPPORTED"
            )
        })


    # Without delay
    if (
        "without delay" in answer_n
        or "immediately" in answer_n
    ):

        supported = (
            "without delay" in evidence_n
            or "immediately" in evidence_n
        )

        results.append({
            "claim": "Treatment should start without delay under urgent conditions.",
            "type": "TIMING",
            "status": (
                "SUPPORTED"
                if supported
                else "UNSUPPORTED"
            )
        })


    return results


# --------------------------------------------------
# Condition validation
# --------------------------------------------------

def validate_condition_claims(answer, evidence):

    answer_n = normalize(answer)
    evidence_n = normalize(evidence)

    conditions = [
        "cardiovascular disease",
        "high cardiovascular risk",
        "diabetes mellitus",
        "chronic kidney disease",
        "end organ damage",
        "hypertension"
    ]

    results = []

    for condition in conditions:

        if condition in answer_n:

            supported = condition in evidence_n

            results.append({
                "claim": condition,
                "type": "CONDITION",
                "status": (
                    "SUPPORTED"
                    if supported
                    else "UNSUPPORTED"
                )
            })

    return results


# --------------------------------------------------
# Detect potentially unsupported causal claims
# --------------------------------------------------

def validate_causal_claims(answer, evidence):

    answer_n = normalize(answer)
    evidence_n = normalize(evidence)

    results = []

    causal_phrases = [
        "can help reduce",
        "helps reduce",
        "reduces",
        "prevents",
        "leads to",
        "causes",
        "results in"
    ]

    for phrase in causal_phrases:

        if phrase in answer_n:

            # We conservatively mark causal statements
            # as partially supported unless the same
            # relationship appears in the evidence.

            if phrase in evidence_n:

                status = "SUPPORTED"

            else:

                status = "PARTIALLY SUPPORTED"

            results.append({
                "claim": (
                    f"Causal statement containing "
                    f"'{phrase}'"
                ),
                "type": "CAUSAL",
                "status": status
            })

    return results


# --------------------------------------------------
# Main validator
# --------------------------------------------------

def validate_claims(answer, evidence):

    results = []

    results.extend(
        validate_numerical_claims(
            answer,
            evidence
        )
    )

    results.extend(
        validate_timing_claims(
            answer,
            evidence
        )
    )

    results.extend(
        validate_condition_claims(
            answer,
            evidence
        )
    )

    results.extend(
        validate_causal_claims(
            answer,
            evidence
        )
    )

    return results


# --------------------------------------------------
# Scores
# --------------------------------------------------

def calculate_scores(results):

    if not results:

        return {
            "supported": 0,
            "partial": 0,
            "unsupported": 0,
            "total": 0,
            "faithfulness": 0.0
        }


    supported = sum(
        1
        for r in results
        if r["status"] == "SUPPORTED"
    )

    partial = sum(
        1
        for r in results
        if r["status"] == "PARTIALLY SUPPORTED"
    )

    unsupported = sum(
        1
        for r in results
        if r["status"] == "UNSUPPORTED"
    )

    total = len(results)


    # Conservative score:
    # supported = 1
    # partial = 0.5
    # unsupported = 0

    score = (
        supported
        + (0.5 * partial)
    ) / total * 100


    return {
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "total": total,
        "faithfulness": score
    }


# --------------------------------------------------
# Display
# --------------------------------------------------

def display_validation(results):

    print(
        "\n" + "=" * 70
    )

    print(
        "CLAIM-LEVEL EVIDENCE VALIDATION"
    )

    print(
        "=" * 70
    )


    if not results:

        print("\nNo claims detected.")

        return


    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nClaim {i}:"
        )

        print(
            result["claim"]
        )


        if result["status"] == "SUPPORTED":

            print("✓ SUPPORTED")

        elif result["status"] == "PARTIALLY SUPPORTED":

            print("⚠ PARTIALLY SUPPORTED")

        else:

            print("✗ UNSUPPORTED")


    scores = calculate_scores(
        results
    )


    print(
        "\n" + "-" * 70
    )

    print(
        f"SUPPORTED: {scores['supported']}"
    )

    print(
        f"PARTIALLY SUPPORTED: {scores['partial']}"
    )

    print(
        f"UNSUPPORTED: {scores['unsupported']}"
    )

    print(
        f"TOTAL CLAIMS: {scores['total']}"
    )

    print(
        f"CLAIM FAITHFULNESS: "
        f"{scores['faithfulness']:.2f}%"
    )

    print(
        "-" * 70
    )
