import re

# Comprehensive list of urgency / emergency keywords in grievance contexts
URGENT_KEYWORDS = [
    r"emergenc(?:y|ies)",
    r"harass(?:ment|ing|ed)?",
    r"ragging",
    r"fire",
    r"medical",
    r"safety",
    r"danger(?:ous)?",
    r"urgent(?:ly)?",
    r"threat(?:ening)?",
    r"assault(?:ed)?",
    r"violence",
    r"electric(?:al)?\s+shock",
    r"sparks?",
    r"burn(?:ing)?",
    r"gas\s+leak",
    r"injur(?:y|ies|ed)",
    r"bleeding",
    r"ambulance",
]

# Pre-compile the pattern using word boundaries
URGENT_PATTERN = re.compile(
    r"\b(?:" + "|".join(URGENT_KEYWORDS) + r")\b",
    re.IGNORECASE
)

def get_priority_level(text):
    """
    Determines priority level of a complaint based on emergency/urgent keywords.
    Returns 'High Priority' if matched, otherwise 'Normal'.
    """
    if not text or not isinstance(text, str):
        return "Normal"

    if URGENT_PATTERN.search(text):
        return "High Priority"

    return "Normal"
