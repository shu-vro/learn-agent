import re
import json


def extract_json_payload(raw_content: object) -> dict[str, object]:
    if isinstance(raw_content, dict):
        return raw_content

    text = raw_content if isinstance(raw_content, str) else str(raw_content)
    stripped = text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1).strip()

    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        stripped = stripped[start_index : end_index + 1]

    # strict=False tolerates raw control characters (e.g. literal newlines/tabs
    # inside string values), which LLMs frequently emit and which would
    # otherwise raise JSONDecodeError.
    try:
        payload = json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        # Common secondary breakage: trailing commas before } or ].
        repaired = re.sub(r",(\s*[}\]])", r"\1", stripped)
        payload = json.loads(repaired, strict=False)

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object from the model.")
    return payload


__all__ = ["extract_json_payload"]
