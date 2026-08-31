from __future__ import annotations


def compute_confidence(extraction: dict) -> float:
    """Compute an average confidence score from a nested extraction dict.

    Each field may contain a ``{"value": ..., "confidence": 0.0-1.0}`` entry.
    The function collects every numeric ``confidence`` value across the dict
    (including nested structures) and returns their arithmetic mean.

    Returns ``0.0`` for empty or confidence-free extractions.
    """
    confidences: list[float] = []

    def _walk(obj: object) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key == "confidence" and isinstance(val, (int, float)):
                    confidences.append(float(val))
                elif key.endswith("_confidence") and isinstance(val, (int, float)):
                    confidences.append(float(val))
                else:
                    _walk(val)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(extraction)

    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)
