import re

# Weighted severity rules for numerical urgency scoring (0-100)
# Categories: High Priority (>= 70), Medium Priority (40-69), Low Priority (< 40)
URGENCY_RULES = [
    # Critical Life / Safety / Fire / Medical / Anti-Ragging Hazards
    (r"\b(?:fire|flames?|smoke|explosion|gas\s+leak|short\s+circuit|live\s+wire|electric(?:al)?\s+shock)\b", 70, "Fire / Electrical Safety Hazard"),
    (r"\b(?:suicid(?:e|al)|unconscious|bleeding|ambulance|poison(?:ing)?|severe\s+injur(?:y|ies)|medical\s+emergenc(?:y|ies))\b", 75, "Medical Emergency"),
    (r"\b(?:ragging|assault(?:ed)?|violence|weapon|harass(?:ment|ing|ed)?|threat(?:ening)?|stalking)\b", 65, "Personal Safety / Anti-Ragging"),
    (r"\b(?:emergenc(?:y|ies)|sos|critical|danger(?:ous)?|collapsed?|panic)\b", 55, "Declared Emergency"),
    (r"\b(?:sparks?|burn(?:ing)?|shock(?:ed)?|injur(?:y|ies|ed)|accident|hospital)\b", 50, "Safety / Injury Risk"),

    # High-Impact Security & Infrastructure Incidents
    (r"\b(?:stolen|theft|robbery|intruder|trespass(?:er)?|unauthorized\s+entry|opened\s+my\s+room|broke\s+into)\b", 52, "Security / Theft Incident"),
    (r"\b(?:flooding|water\s+flooding|lift\s+stuck|elevator\s+stuck|ceiling\s+(?:fell|collapse|plaster))\b", 50, "Severe Infrastructure Hazard"),
    (r"\b(?:urgent(?:ly)?|immediate(?:ly)?|asap|severe)\b", 40, "Time-Sensitive Urgency"),
    (r"\b(?:food\s+poisoning|cockroach|insect\s+in\s+food|hair\s+in\s+food|sour\s+milk)\b", 48, "Food Hygiene Risk"),
    (r"\b(?:exam\s+today|tomorrow\s+exam|hall\s+ticket|admit\s+card|exam\s+clash|answers?\s+vanished)\b", 45, "Urgent Academic / Exam Impact"),

    # Medium-Impact Campus Service Disruptions
    (r"\b(?:leak(?:ing|age)?|pipe\s+burst|no\s+water|water\s+supply|drainage|sewage|blocked\s+toilet|flush\s+not|power\s+cut|fuse)\b", 32, "Essential Facility Disruption"),
    (r"\b(?:wi[\s-]?fi|internet|network|portal|ums|lms|server|500\s+error|internal\s+error|cannot\s+login|password\s+reset)\b", 28, "Digital / IT Service Disruption"),
    (r"\b(?:charged\s+twice|double\s+payment|deducted\s+twice|refund|scholarship|fee\s+receipt|fine)\b", 28, "Financial / Billing Issue"),
    (r"\b(?:stale\s+food|uncooked|cold\s+food|unhygienic|dirty\s+water|water\s+cooler|ac\s+not|air\s+condition|fan\s+not|broken\s+door|broken\s+window|projector)\b", 25, "Campus Facility / Service Quality"),
    (r"\b(?:internal\s+marks|grade\s+card|marksheet|re-?evaluation|overdue\s+fine|library\s+fine|rude|misbehavior|misconduct)\b", 24, "Academic / Administrative Discrepancy"),
]

COMPILED_RULES = [
    (re.compile(pattern, re.IGNORECASE), weight, label)
    for pattern, weight, label in URGENCY_RULES
]


def calculate_urgency(text):
    """
    Computes a numerical urgency score (0-100) and categorizes into:
      - 'High Priority'   (score >= 70)
      - 'Medium Priority' (40 <= score < 70)
      - 'Low Priority'    (score < 40)
    Returns: (urgency_score: int, priority_level: str, reasons: list[str])
    """
    if not text or not isinstance(text, str) or not text.strip():
        return 15, "Low Priority", []

    score = 18  # baseline for any submitted grievance
    reasons = []

    for regex, weight, label in COMPILED_RULES:
        if regex.search(text):
            score += weight
            if label not in reasons:
                reasons.append(label)

    # Extra bump if multiple exclamation marks or ALL-CAPS urgency words appear
    if "!!" in text:
        score += 5

    score = max(10, min(98, int(score)))

    if score >= 70:
        priority = "High Priority"
    elif score >= 40:
        priority = "Medium Priority"
    else:
        priority = "Low Priority"

    return score, priority, reasons


def get_priority_level(text):
    """Backward-compatible helper returning High Priority / Medium Priority / Low Priority."""
    _, priority, _ = calculate_urgency(text)
    return priority
