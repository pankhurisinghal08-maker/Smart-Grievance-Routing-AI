"""
Smart Campus Grievance System - RAG Engine (Retrieval-Augmented Generation)
Standard University Terms, Policies, Acceptability Verification,
Semantic Clustering & One-Click Redressal Synthesizer.
"""

import os 
import re
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

# ==============================================================================
# 1. UNIVERSITY STANDARD POLICIES & TERMS KNOWLEDGE BASE
# ==============================================================================

CAMPUS_POLICIES = [
    {
        "id": "POL-IT-001",
        "department": "IT",
        "category": "Network & Wi-Fi Access",
        "title": "Campus Wi-Fi & Network Infrastructure Policy",
        "citation": "LPU IT Code 2026, Sec 4.1 (Campus Digital Connectivity)",
        "sla_hours": 12,
        "acceptable_scope": [
            "hostel room wifi weak signal", "lab internet disconnected", "wifi access point down",
            "cannot connect to campus network", "slow speed during class hours", "ip address conflict",
            "dns lookup failure", "captive portal login timeout", "router flashing red"
        ],
        "unacceptable_scope": [
            "request to bypass campus firewall", "request to unblock torrents or gambling",
            "requesting admin wifi passwords", "setting up unauthorized private routers",
            "sniffing campus network packets", "requesting proxy bypass for attendance"
        ],
        "default_advisory": (
            "Valid IT Grievance. Under Sec 4.1, campus network team guarantees 99.5% uptime. "
            "Please verify that MAC address randomization is disabled in your Wi-Fi device settings. "
            "An IT field technician will inspect the nearest block access point within 12 hours."
        ),
        "violation_reason": (
            "Grievance rejected under Sec 4.1.3: Circumventing institutional firewall rules, unauthorized "
            "proxy requests, or tampering with university network security credentials is strictly prohibited "
            "under IT Act 2000 and University Disciplinary Rules."
        )
    },
    {
        "id": "POL-IT-002",
        "department": "IT",
        "category": "Academic Portals & UMS / LMS",
        "title": "University Management System (UMS) & LMS Access Terms",
        "citation": "LPU Academic Computing Norms, Clause 5.2",
        "sla_hours": 24,
        "acceptable_scope": [
            "ums fee receipt download error", "lms assignment submission portal crashed",
            "portal shows 500 internal server error", "cannot view grade sheet on ums",
            "student email password reset", "biometric attendance sync mismatch",
            "course registration portal timed out"
        ],
        "unacceptable_scope": [
            "requesting another student's login credentials", "requesting to change attendance manually without mentor approval",
            "submitting fake medical certificates via tech ticket", "hacking portal database"
        ],
        "default_advisory": (
            "Valid Grievance. UMS Technical Support ticket logged. If submitting an urgent assignment, "
            "please note ticket ID in an email to your course instructor to claim technical amnesty."
        ),
        "violation_reason": (
            "Grievance rejected under Clause 5.2: Student credentials cannot be altered or accessed on behalf "
            "of another individual, and attendance ledger modifications require verified physical mentor sanction."
        )
    },
    {
        "id": "POL-MAINT-001",
        "department": "Maintenance",
        "category": "Civil & Plumbing Infrastructure",
        "title": "Hostel & Classroom Facilities Maintenance Regulations",
        "citation": "LPU Estate Maintenance Bylaws, Art. 8 (Civil & Sanitation)",
        "sla_hours": 24,
        "acceptable_scope": [
            "water leaking from washroom pipe", "tap broken sink flooding", "toilet flush not working",
            "no running water in hostel floor", "drinking water water cooler filter dirty",
            "drainage choke foul smell", "ceiling water seepage rain leakage",
            "toilet blocked", "blocked toilet", "clean toilet", "warden clean toilet", "choked toilet"
        ],
        "unacceptable_scope": [
            "requesting personal bathroom remodeling", "demanding luxury bathtub or jacuzzi in hostel",
            "demanding custom tile replacement", "intentional vandalism of university plumbing"
        ],
        "default_advisory": (
            "Valid Maintenance Grievance under Art. 8 (Civil & Sanitation). Facilities maintenance desk notified. "
            "Duty technician dispatched for inspection. Expected resolution within 24 hours."
        ),
        "violation_reason": (
            "Grievance rejected under Art. 8.4: Standard residential hostel accommodation follows uniform "
            "sanitation inventory; personalized luxury civil modifications are not permitted."
        )
    },
    {
        "id": "POL-MAINT-002",
        "department": "Maintenance",
        "category": "Electrical & HVAC Safety",
        "title": "Campus Electrical & HVAC Safety Standards",
        "citation": "LPU Electrical Safety Regulations & Fire Code 2026",
        "sla_hours": 6,
        "acceptable_scope": [
            "electrical sparks from switchboard", "ac cooling not working cooling fan broken",
            "short circuit smell burning wire", "tube light sparking in classroom",
            "power cut in hostel room fuse tripped", "elevator lift stuck emergency",
            "geyser hot water shock"
        ],
        "unacceptable_scope": [
            "requesting permission for unauthorized 2000W heating coils", "tampering with main substation breaker",
            "running cryptocurrency mining rig on room power", "illegal electrical wire tapping"
        ],
        "default_advisory": (
            "PRIORITY SAFETY GRIEVANCE under Fire Code 2026. Electrical substation alert initiated. "
            "Please maintain a safe 5-meter distance from the sparking fixture. Duty electrician dispatched."
        ),
        "violation_reason": (
            "Grievance rejected under Fire Code Sec 12: High-wattage heating coils, heavy unapproved "
            "appliances, or unauthorized wire tapping represent severe fire hazards subject to disciplinary fines."
        )
    },
    {
        "id": "POL-CAFE-001",
        "department": "Cafeteria",
        "category": "Food Safety & Mess Hygiene",
        "title": "University Food Safety, Hygiene & Canteen Redressal Standards",
        "citation": "FSSAI Compliance & LPU Food Charter, Section 3.4",
        "sla_hours": 12,
        "acceptable_scope": [
            "uncooked food served in mess", "foreign object insect hair found in lunch",
            "stale food bad smell cafeteria", "mess water dispenser dirty",
            "canteen vendor double charged qr payment failed", "mess staff not wearing gloves hairnet",
            "food poisoning symptoms reported by students"
        ],
        "unacceptable_scope": [
            "demanding free meal coupons without financial aid", "demanding outside uncontracted fast food chains",
            "refusing to pay canteen bill", "harassing cafeteria staff or throwing utensils"
        ],
        "default_advisory": (
            "Valid Food Hygiene Grievance under FSSAI Section 3.4. Food safety supervisor alerted. "
            "Sample from the current meal lot has been quarantined for microbiological evaluation. "
            "If transaction was debited twice, auto-reversal is triggered within 24-48 bank hours."
        ),
        "violation_reason": (
            "Grievance rejected under Food Charter Clause 6: Student mess operations are governed by "
            "democratic student mess committee menus; disorderly conduct or unsubstantiated demands for "
            "unapproved vendors fall outside redressal scope."
        )
    },
    {
        "id": "POL-FIN-001",
        "department": "Finance",
        "category": "Fee Assessment & Refunds",
        "title": "Student Financial Services, Scholarships & Refund Regulations",
        "citation": "LPU Financial Ordinances, Chapter IX (Fees & Refunds)",
        "sla_hours": 48,
        "acceptable_scope": [
            "fee payment deducted twice from bank account", "scholarship amount not credited to fee balance",
            "security caution money refund pending after no-dues clearance",
            "ums fee challan shows incorrect late fine", "educational loan disbursement verification delayed",
            "receipt not generated after successful net banking"
        ],
        "unacceptable_scope": [
            "demanding complete waiver of tuition fee without academic merit or scholarship",
            "demanding refund after non-refundable semester commencement cutoff",
            "requesting cash disbursement without audited bank account", "evading university mandatory fees"
        ],
        "default_advisory": (
            "Valid Financial Grievance under Chapter IX. Accounts reconciliation desk notified. "
            "Double-deductions are resolved within 48-72 business hours upon automated bank gateway reconciliation. "
            "Please ensure your bank UTR number is shared in case discussion."
        ),
        "violation_reason": (
            "Grievance rejected under Chapter IX, Art. 4: University tuition and institutional fee schedules "
            "are approved by Board of Governors. Blanket fee waivers cannot be granted through grievance redressing "
            "without formal scholarship board petition."
        )
    },
    {
        "id": "POL-SEC-001",
        "department": "Security",
        "category": "Campus Safety, Anti-Ragging & Property",
        "title": "Campus Safety, Anti-Ragging & Security Code",
        "citation": "UGC Anti-Ragging Directives & LPU Security Manual 2026",
        "sla_hours": 4,
        "acceptable_scope": [
            "ragging incident in hostel block", "verbal physical bullying harassment",
            "stolen laptop bicycle phone from library", "unauthorized stranger trespasser in corridor",
            "broken cctv camera near girls hostel", "streetlights dark on campus walkway at night",
            "lost id card access card blocked", "sos emergency threat",
            "someone opened my room without permission", "unauthorized entry into room", "unlocked my room", "room privacy", "someone opened room"
        ],
        "unacceptable_scope": [
            "complaint protesting standard security guard id verification at campus gates",
            "demanding security allow illegal substance smuggling", "frivolous prank emergency alarms",
            "refusal to submit vehicle parking sticker"
        ],
        "default_advisory": (
            "Valid Campus Security Grievance under Security Manual Sec 2. Duty proctor and campus security patrol notified to investigate unauthorized room access and review corridor CCTV surveillance."
        ),
        "violation_reason": (
            "Grievance rejected under Security Manual Sec 2: Institutional identity card verification, vehicle frisking, "
            "and campus boundary surveillance are statutory safety measures; obstruction or protests against "
            "legitimate security checks constitute disciplinary violations."
        )
    },
    {
        "id": "POL-LIB-001",
        "department": "Library",
        "category": "Learning Resources & Silent Study",
        "title": "Central Library Circulation & Digital Repository Guidelines",
        "citation": "LPU Central Library Manual, Part IV",
        "sla_hours": 24,
        "acceptable_scope": [
            "returned book showing pending fine on library card", "ieee acm digital library access expired",
            "noise loud talking in 4th floor quiet study zone", "air conditioning broken in library reading hall",
            "book lost replacement calculation error", "rfid turnstile gate reading error"
        ],
        "unacceptable_scope": [
            "requesting exemption from lost book replacement cost", "demanding to borrow reference-only rare books off-campus",
            "stealing library books or tearing journal pages", "sleeping continuously in study carrels overnight"
        ],
        "default_advisory": (
            "Valid Library Grievance under Part IV. Assistant Librarian notified. The circulation desk RFID "
            "return audit log will be verified within 24 hours to waive erroneously marked overdue fines."
        ),
        "violation_reason": (
            "Grievance rejected under Library Manual Sec 4.5: Standard replacement charges for damaged or lost "
            "university volumes are non-negotiable institutional property safeguards."
        )
    },
    {
        "id": "POL-EXAM-001",
        "department": "Examination",
        "category": "Controller of Examinations & Evaluations",
        "title": "University Examination Code & Evaluation Conduct",
        "citation": "LPU Examination Ordinances, Clause 14 (Redressal & Grievance)",
        "sla_hours": 36,
        "acceptable_scope": [
            "internal assessment marks missing from ums grade card", "hall ticket admit card download photo corrupted",
            "two end-term exams scheduled on same date and time slot", "re-evaluation result not declared after 3 weeks",
            "scribes accommodation needed for injured student", "wrong subject code listed in exam date sheet"
        ],
        "unacceptable_scope": [
            "demanding advance examination question papers", "protesting detention for attendance below 75% without medical sanction",
            "requesting cancellation of unfair means (UFM) disciplinary action without committee review",
            "demanding passing grades without taking exam"
        ],
        "default_advisory": (
            "Valid Examination Grievance under Clause 14. Controller of Examinations (COE) desk notified. "
            "Internal mark verification request forwarded to Department Subject Coordinator for roster audit."
        ),
        "violation_reason": (
            "Grievance rejected under Exam Ordinance Sec 14.8: Demands for unearned marks, advance paper disclosures, "
            "or circumventing mandatory attendance eligibility thresholds (75%) strictly contravene statutory university norms."
        )
    },
    {
        "id": "POL-DISC-001",
        "department": "Discipline",
        "category": "Student Code of Conduct, Staff Affairs & Proctorial Grievances",
        "title": "University Proctorial Guidelines, Staff Conduct & Dispute Redressal",
        "citation": "LPU Student Handbook 2026, Code of Conduct Sec 2",
        "sla_hours": 24,
        "acceptable_scope": [
            "roommate dispute sleep disruption", "unauthorized entry into room",
            "loud music played during exam study week", "defamation or cyber bullying in university group",
            "unresolved conflict in student club committee",
            "hospital staff is so much annoying", "hospital staff annoying", "hospital staff rude", "dispensary staff misbehavior",
            "medical clinic staff unprofessional", "campus staff rude arrogant annoying", "warden misconduct", "staff behavior complaint",
            "hospital staff", "dispensary staff", "clinic staff", "medical staff"
        ],
        "unacceptable_scope": [
            "demanding university ignore physical assault or drug possession", "requesting removal of suspension without inquiry",
            "filing malicious knowingly false counter-complaints to intimidate witnesses"
        ],
        "default_advisory": (
            "Valid Grievance under Code of Conduct & Proctorial Regulations. Proctorial board and department superintendent "
            "notified to review staff conduct and address service grievances. Inquiry and student assistance initiated within 24-48 hours."
        ),
        "violation_reason": (
            "Grievance rejected under Code of Conduct Sec 2.9: Malicious or retaliatory filings intended "
            "to obstruct formal investigations are subject to immediate proctorial disciplinary review."
        )
    }
]

# ==============================================================================
# 2. RAG VECTOR KNOWLEDGE BASE INITIALIZER
# ==============================================================================

class PolicyRAGKnowledgeBase:
    def __init__(self):
        self.policies = CAMPUS_POLICIES
        self.vectorizer = None
        self.policy_matrix = None
        self.doc_texts = []
        self._build_index()

    def _build_index(self):
        self.doc_texts = []
        for p in self.policies:
            combined = (
                f"{p['title']} {p['category']} {p['department']} "
                f"{' '.join(p['acceptable_scope'])} "
                f"{' '.join(p['unacceptable_scope'])} "
                f"{p['citation']} {p['default_advisory']}"
            )
            self.doc_texts.append(combined)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
        self.policy_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def retrieve_relevant_policy(self, query_text, department=None):
        if not query_text or not self.vectorizer:
            return self.policies[0], 0.0

        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.policy_matrix)[0]

        # If a department is specified, prioritize policies belonging to that department
        if department:
            dept_indices = [i for i, p in enumerate(self.policies) if p.get('department') == department]
            if dept_indices:
                best_sub_idx = dept_indices[int(np.argmax([sims[i] for i in dept_indices]))]
                return self.policies[best_sub_idx], float(sims[best_sub_idx])

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return self.policies[best_idx], best_score

# Singleton knowledge base
_rag_kb = PolicyRAGKnowledgeBase()

# ==============================================================================
# 3. ACCEPTABILITY & COMPLIANCE VERIFICATION ENGINE
# ==============================================================================

# Strong red-flag signals for prohibited / non-compliant requests
PROHIBITED_PATTERNS = [
    (r'\b(bypass|hack|crack|tamper)\s+(firewall|proxy|attendance|password|server)\b', "Attempt to bypass security or attendance proxy"),
    (r'\b(torrent|gambling|crypto\s*mining|porn|dark\s*web)\b', "Use of campus resources for prohibited or high-bandwidth unauthorized services"),
    (r'\b(leak|leakage|advance)\s+(paper|exam|question|solution)\b', "Solicitation of unauthorized academic examination material"),
    (r'\b(cancel|waive)\s+(attendance|suspension|ufm|punishment|penalty|fine)\s+(without|mandatory)\b', "Request to nullify statutory disciplinary action without due process"),
    (r'\b(free\s+degree|passing\s+marks|give\s+me\s+marks|increase\s+marks\s+without)\b', "Unsubstantiated request for academic grading manipulation"),
    (r'\b(alcohol|drugs|weed|contraband|hookah|substance)\b', "Prohibited contraband / narcotics reference"),
    (r'\b(jacuzzi|bathtub|luxury|personal\s+maid|private\s+chef)\b', "Unreasonable out-of-scope personal luxury demands in student accommodation"),
    (r'\b(fake|fraudulent|forged)\s+(certificate|receipt|medical)\b', "Submission or mention of fraudulent documentation")
]

def evaluate_complaint_acceptability(description, department=None):
    """
    Evaluates whether a student complaint adheres to university terms and policies.
    Returns a dict with:
      - is_acceptable: bool
      - status: 'Acceptable' | 'Policy Violation' | 'Conditional'
      - policy_score: float (0.0 to 100.0)
      - matched_policy: dict (title, id, citation, sla)
      - citation: str
      - verdict_title: str
      - advisory_response: str
      - violation_reasons: list of str
    """
    clean_desc = (description or "").strip().lower()
    if not clean_desc or len(clean_desc) < 8:
        return {
            "is_acceptable": False,
            "status": "Conditional",
            "policy_score": 40.0,
            "matched_policy": CAMPUS_POLICIES[0],
            "citation": "LPU Student Handbook 2026, Sec 1 (Submission Guidelines)",
            "verdict_title": "Insufficient Detail",
            "advisory_response": "Your complaint description is too brief. Please provide specific details (e.g. room/block number, error message, or date of incident) so our automated triage can route it properly.",
            "violation_reasons": ["Description is too short for meaningful verification."]
        }

    # 1. Check for blatant red flags / prohibited patterns
    for pat, label in PROHIBITED_PATTERNS:
        if re.search(pat, clean_desc, re.IGNORECASE):
            policy, _ = _rag_kb.retrieve_relevant_policy(clean_desc, department=department)
            return {
                "is_acceptable": False,
                "status": "Policy Violation",
                "policy_score": 15.0,
                "matched_policy": policy,
                "citation": policy.get("citation", "LPU Student Code of Conduct 2026"),
                "verdict_title": "Rejected: Standard Policy Violation",
                "advisory_response": policy.get("violation_reason", "This request falls outside the acceptable scope of university grievance redressal standards."),
                "violation_reasons": [label]
            }

    # 2. Retrieve most relevant campus policy via RAG vector similarity
    policy, relevance = _rag_kb.retrieve_relevant_policy(clean_desc, department=department)

    # 3. Check for exact negative matches against policy unacceptable scope
    unacceptable_hits = []
    for unacc in policy.get("unacceptable_scope", []):
        unacc_tokens = set(unacc.lower().split())
        desc_tokens = set(clean_desc.split())
        overlap = len(unacc_tokens & desc_tokens)
        if overlap >= 3 or (len(unacc_tokens) <= 3 and overlap >= 2):
            unacceptable_hits.append(unacc)

    if unacceptable_hits:
        return {
            "is_acceptable": False,
            "status": "Policy Violation",
            "policy_score": 25.0,
            "matched_policy": policy,
            "citation": policy["citation"],
            "verdict_title": f"Rejected Under {policy['id']}",
            "advisory_response": policy["violation_reason"],
            "violation_reasons": [f"Matches prohibited scope: '{unacc}'" for unacc in unacceptable_hits]
        }

    # 4. Check for positive scope alignment
    positive_hits = 0
    for acc in policy.get("acceptable_scope", []):
        acc_tokens = set(acc.lower().split())
        desc_tokens = set(clean_desc.split())
        overlap = len(acc_tokens & desc_tokens)
        if overlap >= 2:
            positive_hits += 1

    # Base score calculated from RAG relevance + positive matching
    calculated_score = min(98.5, max(65.0, (relevance * 50) + (positive_hits * 15) + 40))

    # Determine status
    if calculated_score >= 60.0:
        status = "Acceptable"
        verdict_title = "Acceptable Grievance (Compliant with University Terms)"
        advisory = policy.get("default_advisory")
    else:
        status = "Conditional"
        verdict_title = "Conditional Review: Supplementary Evidence Required"
        advisory = (
            f"Under {policy['citation']}, this complaint has been logged conditionally. "
            "Please provide supporting proof (such as transaction receipts, screenshot, or witness detail) "
            "in the case discussion thread to expedite processing."
        )

    return {
        "is_acceptable": (status == "Acceptable"),
        "status": status,
        "policy_score": round(calculated_score, 1),
        "matched_policy": policy,
        "citation": policy["citation"],
        "verdict_title": verdict_title,
        "advisory_response": advisory,
        "violation_reasons": []
    }

# ==============================================================================
# 4. SIMILAR GRIEVANCE CLUSTERING & SEMANTIC DEDUPLICATION
# ==============================================================================

def find_similar_complaints(query_text, all_existing_complaints, threshold=0.55):
    """
    Scans existing historical complaints and finds semantically similar records.
    Returns a tuple: (similar_list, cluster_id, is_duplicate)
    """
    if not query_text or not all_existing_complaints:
        return [], None, False

    clean_query = query_text.strip().lower()
    docs = [clean_query] + [c.get("description", "").lower() for c in all_existing_complaints]

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', min_df=1)
    try:
        matrix = vec.fit_transform(docs)
    except Exception:
        return [], None, False

    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]

    similar_results = []
    cluster_id = None
    is_duplicate = False

    for i, score in enumerate(sims):
        if score >= threshold:
            item = all_existing_complaints[i]
            sim_pct = round(float(score) * 100, 1)
            similar_results.append({
                "id": item.get("id"),
                "student_name": item.get("student_name") or "Student",
                "roll_number": item.get("roll_number") or "",
                "department": item.get("predicted_department"),
                "priority": item.get("priority_level"),
                "status": item.get("status"),
                "description": item.get("description", "")[:120] + "...",
                "similarity": sim_pct,
                "created_at": str(item.get("created_at") or "")
            })
            if score >= 0.82:
                is_duplicate = True

    # Sort descending by similarity
    similar_results.sort(key=lambda x: x["similarity"], reverse=True)

    if similar_results:
        # Formulate cluster ID based on the top match's department or earliest ID
        top_match = similar_results[0]
        dept = top_match.get("department") or "GEN"
        cluster_id = f"CLUST-{dept[:4].upper()}-{top_match['id']}"

    return similar_results[:8], cluster_id, is_duplicate

# ==============================================================================
# 5. RAG AI RESOLUTION DRAFT GENERATOR (FOR FACULTY & MASS BROADCAST)
# ==============================================================================

def generate_rag_resolution_draft(complaint_description, department, faculty_name="Assigned Officer", similar_count=0):
    """
    Synthesizes an intelligent, policy-backed resolution note / student reply
    incorporating standard departmental operating procedures.
    """
    policy, _ = _rag_kb.retrieve_relevant_policy(complaint_description)
    dept_name = department or policy.get("department", "University")

    # Domain specific technical solutions
    domain_actions = {
        "IT": (
            "1. Verified access point health and re-provisioned IP lease on campus gateway.\n"
            "2. Reset port configuration and validated bandwidth throughput.\n"
            "3. Checked server error logs and cleared authentication cache for affected student accounts."
        ),
        "Maintenance": (
            "1. Deputed field plumbing and sanitation team for physical inspection.\n"
            "2. Isolated faulty joint/valve, replaced worn washer, and restored normal drainage pressure.\n"
            "3. Conducted hygiene and safety sanitization around the affected facility floor."
        ),
        "Cafeteria": (
            "1. Inspected food preparation counter, temperature logs, and vendor hygiene compliance.\n"
            "2. Quarantined suspicious food lot and conducted supervisor review with the mess vendor.\n"
            "3. Initiated payment gateway reversal check for any duplicated UPI debit."
        ),
        "Finance": (
            "1. Audited ERP merchant ledger against bank settlement transaction records.\n"
            "2. Verified student roll number account reconciliation and cleared fee challan hold.\n"
            "3. Generated updated no-dues receipt on UMS student portal."
        ),
        "Security": (
            "1. Reviewed CCTV surveillance timeline and logged perimeter patrol report.\n"
            "2. Escalated incident report to Proctorial Board and deployed stationed security personnel.\n"
            "3. Contacted student for confidential verification and security reassurance."
        ),
        "Examination": (
            "1. Reconciled internal marks ledger against teacher award sheets.\n"
            "2. Corrected grade discrepancy in central database and re-issued updated hall ticket.\n"
            "3. Logged official verification entry with Controller of Examinations (COE)."
        ),
        "Library": (
            "1. Checked automated RFID drop box and shelf return history.\n"
            "2. Erroneously marked late fine waived on student library ledger.\n"
            "3. Restored borrowing privileges on digital card."
        ),
        "Discipline": (
            "1. Convened confidential counseling between concerned parties.\n"
            "2. Recorded mutual undertaking in compliance with University Code of Conduct.\n"
            "3. Enforced quiet-hours study regulations with floor warden monitoring."
        )
    }

    action_text = domain_actions.get(dept_name, (
        "1. Deputed departmental officer for on-site inspection.\n"
        "2. Addressed grievance under University Standard Operating Procedures.\n"
        "3. Completed verification and notified administration."
    ))

    mass_prefix = f" [Mass Redressal applied to {similar_count} students affected by this issue]" if similar_count > 1 else ""

    draft = (
        f"Official Redressal Summary - {dept_name} Desk{mass_prefix}\n\n"
        f"Dear Student,\n"
        f"Your grievance has been investigated under {policy['citation']}.\n\n"
        f"Action Taken by {faculty_name}:\n"
        f"{action_text}\n\n"
        f"The issue has been addressed in full accordance with university standards. "
        f"Please verify resolution and submit your satisfaction feedback to complete case closure."
    )
    return draft

# ==============================================================================
# 6. MULTI-FACTOR HAZARD RISK & SLA CALCULATOR
# ==============================================================================

def calculate_hazard_risk_score(description):
    """
    Computes a 0 to 100 Hazard & Risk Score based on safety, infrastructure,
    academic impact, and panic words.
    """
    text = (description or "").lower()
    score = 10  # baseline

    hazard_weights = [
        (r'\b(fire|sparks?|flames?|gas\s*leak|short\s*circuit|explosion)\b', 45, "Severe Fire / Explosion Hazard"),
        (r'\b(shock|electrocution|live\s*wire)\b', 40, "Electrical Hazard"),
        (r'\b(suicid|bleeding|unconscious|poison|hospital|ambulance)\b', 50, "Medical Emergency"),
        (r'\b(ragging|harass|assault|violence|threat|stalking)\b', 40, "Personal Safety / UGC Anti-Ragging"),
        (r'\b(flooding|ceiling\s*collapse|broken\s*glass|elevator\s*stuck)\b', 30, "Infrastructure Hazard"),
        (r'\b(exam\s*today|tomorrow\s*exam|hall\s*ticket|admit\s*card)\b', 25, "Urgent Academic Deadline"),
        (r'\b(theft|stolen|robbery|laptop\s*stolen)\b', 25, "Campus Security / Loss of Property")
    ]

    detected_reasons = []
    for pat, weight, label in hazard_weights:
        if re.search(pat, text):
            score += weight
            detected_reasons.append(label)

    score = min(100, max(5, score))
    return score, detected_reasons

def calculate_dynamic_sla(priority_level, department, created_at_dt=None):
    """
    Calculates target resolution SLA in hours and exact deadline timestamp.
    """
    if priority_level == 'High Priority':
        hours = 6
    elif department in ('IT', 'Cafeteria'):
        hours = 18
    elif department in ('Maintenance', 'Discipline'):
        hours = 24
    else:
        hours = 48

    base_time = created_at_dt or datetime.datetime.now()
    deadline = base_time + datetime.timedelta(hours=hours)
    return hours, deadline

def evaluate_sla_breach_risk(created_at, sla_hours, current_status):
    """
    Evaluates whether an active ticket is at risk of breaching SLA.
    Returns: ('Low' | 'Medium' | 'High' | 'Breached', hours_left_str)
    """
    if current_status in ('Closed', 'Resolved'):
        return 'Resolved', 'Resolved on time'

    if not created_at:
        return 'Low', 'Normal'

    try:
        if isinstance(created_at, str):
            # Parse common formats
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
                try:
                    dt = datetime.datetime.strptime(created_at, fmt)
                    break
                except ValueError:
                    pass
            else:
                return 'Low', 'Tracking'
        else:
            dt = created_at

        now = datetime.datetime.now()
        elapsed = (now - dt).total_seconds() / 3600.0
        remaining = sla_hours - elapsed

        if remaining <= 0:
            return 'Breached', f'Breached by {abs(round(remaining, 1))}h'
        elif remaining <= (sla_hours * 0.25):
            return 'High', f'{round(remaining, 1)}h remaining (Urgent)'
        elif remaining <= (sla_hours * 0.5):
            return 'Medium', f'{round(remaining, 1)}h remaining'
        else:
            return 'Low', f'{round(remaining, 1)}h remaining'
    except Exception:
        return 'Low', 'Normal'

# ==============================================================================
# 7. AUTOMATED TOPIC & HASHTAG GENERATOR
# ==============================================================================

def generate_topic_hashtags(description, department):
    """
    Extracts relevant campus sub-topics and hashtags for quick filtering.
    """
    text = (description or "").lower()
    tags = [f"#{department.replace(' ', '')}"]

    topic_rules = [
        (r'\b(wifi|internet|network|router|lan)\b', '#WiFi'),
        (r'\b(water|pipe|leak|tap|sink|toilet|flush)\b', '#Plumbing'),
        (r'\b(electricity|spark|light|fan|ac|switchboard|power)\b', '#Electrical'),
        (r'\b(mess|food|canteen|lunch|dinner|breakfast|hygiene)\b', '#FoodSafety'),
        (r'\b(fee|receipt|refund|scholarship|challan|payment)\b', '#FeeAccounts'),
        (r'\b(ragging|harass|safety|guard|cctv|bully)\b', '#CampusSafety'),
        (r'\b(exam|marks|grade|hall\s*ticket|admit\s*card)\b', '#Exams'),
        (r'\b(book|fine|library|journal)\b', '#Library'),
        (r'\b(hostel|room|bed|warden)\b', '#HostelLife'),
        (r'\b(emergency|urgent|sos|fire|critical)\b', '#UrgentSOS')
    ]

    for pat, tag in topic_rules:
        if re.search(pat, text):
            if tag not in tags:
                tags.append(tag)

    return " ".join(tags[:4])
