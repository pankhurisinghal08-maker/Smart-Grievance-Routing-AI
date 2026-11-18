"""
Smart Grievance Routing AI - RAG Engine & AI Student Assistant Chatbot
Retrieves official university terms, rules, policies, and admin-uploaded documents
stored in MySQL (`rag_documents` with TF-IDF embeddings) and powers:
  1. Policy compliance & advisory grounding for submitted grievances
  2. Conversational AI Student Assistance Chatbot (policy Q&A, live routing preview, ticket lookup)
"""

import json
import re
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from database.db import execute_query
from model.nlp_model import normalize_text, predict_department_with_confidence
from model.rules import calculate_urgency


# ==============================================================================
# 1. OFFICIAL UNIVERSITY TERMS, RULES & POLICIES KNOWLEDGE BASE
# ==============================================================================

DEFAULT_UNIVERSITY_POLICIES = [
    {
        "doc_code": "POL-IT-001",
        "department": "IT",
        "title": "Campus Wi-Fi, LAN & Network Infrastructure Policy",
        "citation": "LPU IT Code 2026, Sec 4.1 (Campus Digital Connectivity)",
        "sla_hours": 12,
        "content": (
            "Campus Wi-Fi, hostel internet, LAN ports, routers, and computer labs are maintained by the IT Department "
            "with a target resolution SLA of 12 hours. Valid grievances include weak hostel Wi-Fi signal, disconnected "
            "access points, slow internet speed, IP/DNS failures, captive portal login timeouts, and lab PC or projector "
            "malfunctions. Students should disable MAC address randomization when troubleshooting hostel Wi-Fi. "
            "Prohibited actions under Sec 4.1.3: bypassing campus firewall or proxy, unblocking torrents or gambling, "
            "setting up unauthorized private routers, packet sniffing, or requesting admin Wi-Fi credentials."
        ),
        "acceptable_scope": [
            "hostel wifi weak signal", "lab internet disconnected", "wifi access point down",
            "cannot connect campus network", "slow internet speed", "captive portal timeout",
            "router flashing red", "lab computer slow", "projector blue screen"
        ],
        "unacceptable_scope": [
            "bypass campus firewall", "unblock torrents gambling", "admin wifi passwords",
            "unauthorized private routers", "sniffing network packets", "proxy bypass attendance"
        ],
        "default_advisory": (
            "Valid IT Grievance under LPU IT Code Sec 4.1 (SLA: 12h). Please disable MAC address randomization "
            "on your device while the IT network technician inspects the nearest access point."
        ),
        "violation_reason": (
            "Rejected under LPU IT Code Sec 4.1.3: Bypassing institutional firewall rules, proxy circumvention, "
            "unauthorized torrenting, or tampering with university network credentials is strictly prohibited."
        ),
    },
    {
        "doc_code": "POL-IT-002",
        "department": "IT",
        "title": "UMS, LMS & Student Email Academic Portal Terms",
        "citation": "LPU Academic Computing Norms, Clause 5.2",
        "sla_hours": 24,
        "content": (
            "The University Management System (UMS), Learning Management System (LMS), and official student email "
            "portals are governed under Clause 5.2 (target SLA: 24 hours). Valid issues include 500 internal server "
            "errors, fee receipt download failures, LMS assignment submission glitches, grade sheet display errors, "
            "email password resets, and biometric attendance sync mismatches. If an LMS outage impacts an assignment "
            "deadline, note your Grievance Ticket ID and inform your course instructor for technical amnesty. "
            "Prohibited actions: requesting another student's login credentials, manual attendance alteration without "
            "mentor approval, uploading forged documents, or unauthorized database access."
        ),
        "acceptable_scope": [
            "ums fee receipt download error", "lms assignment portal crashed", "portal internal server error",
            "cannot view grade sheet ums", "student email password reset", "biometric attendance sync mismatch"
        ],
        "unacceptable_scope": [
            "another student login credentials", "change attendance manually without mentor",
            "fake medical certificates tech ticket", "hacking portal database"
        ],
        "default_advisory": (
            "Valid Portal Grievance under Clause 5.2 (SLA: 24h). UMS/LMS support has been notified; keep your "
            "ticket ID handy if an assignment or deadline is affected."
        ),
        "violation_reason": (
            "Rejected under Clause 5.2: Sharing or altering another student's credentials or modifying attendance "
            "records without verified mentor sanction violates Academic Computing Norms."
        ),
    },
    {
        "doc_code": "POL-MAINT-001",
        "department": "Maintenance",
        "title": "Hostel & Campus Civil, Plumbing & Facilities Maintenance Policy",
        "citation": "LPU Estate & Hostel Maintenance Bylaws, Art. 8",
        "sla_hours": 24,
        "content": (
            "Hostel rooms, washrooms, water coolers, corridors, classrooms, and campus furniture are maintained by "
            "Estate Maintenance under Art. 8 (target SLA: 24 hours). Valid grievances include leaking pipes, broken "
            "taps, clogged washrooms or non-working flushes, water supply outages, dirty water cooler filters, ceiling "
            "seepage, cracked window glass, broken door locks, broken benches/beds, and pest control requests. "
            "Prohibited requests under Art. 8.4: personalized luxury room remodeling, private bathtubs/jacuzzis, "
            "custom tile replacements, or claims arising from intentional vandalism of university property."
        ),
        "acceptable_scope": [
            "water leaking washroom pipe", "broken tap sink flooding", "toilet flush not working",
            "no running water hostel", "water cooler filter dirty", "ceiling seepage plaster",
            "broken door lock window glass", "broken bench chair bed"
        ],
        "unacceptable_scope": [
            "personal bathroom remodeling", "luxury bathtub jacuzzi hostel",
            "custom tile replacement", "intentional vandalism plumbing"
        ],
        "default_advisory": (
            "Valid Maintenance Grievance under Estate Bylaws Art. 8 (SLA: 24h). The facilities team will inspect "
            "and repair the reported fixture promptly."
        ),
        "violation_reason": (
            "Rejected under Estate Bylaws Art. 8.4: Personalized luxury civil modifications fall outside uniform "
            "hostel accommodation standards."
        ),
    },
    {
        "doc_code": "POL-MAINT-002",
        "department": "Maintenance",
        "title": "Campus Electrical, HVAC & Fire Safety Code",
        "citation": "LPU Electrical Safety & Fire Code 2026, Sec 12",
        "sla_hours": 6,
        "content": (
            "Electrical safety hazards and HVAC failures are handled under Fire Code Sec 12 with an emergency SLA "
            "of 6 hours for hazards. Valid reports include sparking switchboards, burning wire smells, short circuits, "
            "electric shocks from geysers/appliances, stuck elevators/lifts, flickering lights, wobbling ceiling fans, "
            "power cuts, and AC leakage or cooling failure. Maintain safe distance from any sparking fixture. "
            "Prohibited actions under Sec 12: unauthorized high-wattage heating coils (2000W+), tampering with main "
            "breakers, cryptocurrency mining rigs on hostel power, or illegal wire tapping."
        ),
        "acceptable_scope": [
            "electrical sparks switchboard", "short circuit burning wire", "ac cooling not working",
            "fan wobbling light flickering", "power cut fuse tripped", "elevator lift stuck",
            "geyser electric shock"
        ],
        "unacceptable_scope": [
            "unauthorized heating coils", "tampering substation breaker",
            "cryptocurrency mining rig power", "illegal wire tapping"
        ],
        "default_advisory": (
            "Safety & Electrical Grievance logged under Fire Code Sec 12 (Priority SLA: 6h). Please stay clear "
            "of any sparking or wet electrical fixture while the duty electrician responds."
        ),
        "violation_reason": (
            "Rejected under Fire Code Sec 12: High-wattage unapproved appliances, crypto rigs, or unauthorized "
            "electrical tapping pose severe fire hazards and are strictly prohibited."
        ),
    },
    {
        "doc_code": "POL-CAFE-001",
        "department": "Cafeteria",
        "title": "Cafeteria, Mess Hygiene & Food Safety Charter",
        "citation": "FSSAI Compliance & LPU Food Charter, Sec 3.4",
        "sla_hours": 12,
        "content": (
            "All campus mess halls, canteens, and food stalls operate under FSSAI norms and LPU Food Charter Sec 3.4 "
            "(target SLA: 12 hours). Valid grievances include stale or uncooked food, foreign objects/insects/hair in "
            "meals, food poisoning symptoms, unhygienic utensils or water dispensers, staff not wearing gloves/hairnets, "
            "unannounced price hikes, and double QR/UPI billing (auto-reconciled within 24-48 bank hours). "
            "Prohibited requests: demanding free meal coupons without financial aid sanction, refusing to pay valid "
            "bills, or disorderly conduct toward cafeteria staff."
        ),
        "acceptable_scope": [
            "uncooked stale food mess", "hair insect cockroach food", "canteen double charged qr payment",
            "food poisoning symptoms", "mess hygiene water dispenser", "canteen rates price hike"
        ],
        "unacceptable_scope": [
            "free meal coupons without financial aid", "refusing pay canteen bill",
            "harassing cafeteria staff throwing utensils"
        ],
        "default_advisory": (
            "Valid Food Safety Grievance under FSSAI & Food Charter Sec 3.4 (SLA: 12h). Food supervisor alerted; "
            "duplicate QR/UPI debits are reconciled within 24-48 banking hours."
        ),
        "violation_reason": (
            "Rejected under Food Charter Clause 6: Unsubstantiated free-meal demands or disorderly conduct fall "
            "outside formal grievance redressal scope."
        ),
    },
    {
        "doc_code": "POL-FIN-001",
        "department": "Finance",
        "title": "Tuition Fees, Scholarships, Refunds & Accounts Ordinance",
        "citation": "LPU Financial Ordinances, Chapter IX (Fees & Refunds)",
        "sla_hours": 48,
        "content": (
            "Student fee assessment, receipts, scholarships, educational loans, and caution money refunds are governed "
            "by LPU Financial Ordinances Chapter IX (target SLA: 48 hours). Valid grievances include double bank "
            "deductions (share your bank UTR number in case replies), scholarship credits missing from ledger, delayed "
            "caution deposit refunds after no-dues clearance, incorrect late fines, and missing fee receipts. "
            "Prohibited requests under Ch. IX Art. 4: blanket tuition fee waivers without formal scholarship board "
            "approval, refunds after semester commencement non-refundable cutoff, or unaudited cash disbursements."
        ),
        "acceptable_scope": [
            "fee payment deducted twice bank", "scholarship not credited balance",
            "caution money security deposit refund", "incorrect late fine challan",
            "education loan disbursement verification", "receipt not generated net banking"
        ],
        "unacceptable_scope": [
            "complete waiver tuition fee without scholarship", "refund after non refundable cutoff",
            "cash disbursement without bank account", "evading mandatory university fees"
        ],
        "default_advisory": (
            "Valid Financial Grievance under Financial Ordinances Ch. IX (SLA: 48h). Accounts reconciliation desk "
            "notified; please share your bank UTR/transaction ID in the case reply thread if payment-related."
        ),
        "violation_reason": (
            "Rejected under Financial Ordinances Ch. IX Art. 4: Blanket fee waivers or post-cutoff refunds cannot "
            "be granted outside Board of Governors scholarship and refund rules."
        ),
    },
    {
        "doc_code": "POL-SEC-001",
        "department": "Security",
        "title": "Campus Security, Anti-Ragging & Property Protection Code",
        "citation": "UGC Anti-Ragging Directives & LPU Security Manual, Sec 2",
        "sla_hours": 4,
        "content": (
            "Lovely Professional University enforces zero tolerance for ragging, harassment, theft, and unauthorized "
            "campus/hostel entry under UGC Anti-Ragging Directives and Security Manual Sec 2 (emergency SLA: 4 hours). "
            "Valid grievances include ragging, bullying, stolen laptops/phones/bicycles, unauthorized entry into hostel "
            "rooms, broken CCTV cameras, dark campus streetlights, gate pass issues, lost ID cards, and SOS threats. "
            "Prohibited actions under Sec 2: protesting mandatory ID card verification at campus gates, obstructing "
            "security checks, filing prank SOS alarms, or bringing prohibited contraband onto campus."
        ),
        "acceptable_scope": [
            "ragging incident hostel", "stolen laptop phone bicycle", "unauthorized entry opened room",
            "broken cctv camera", "dark streetlights walkway", "lost id card blocked", "sos emergency threat"
        ],
        "unacceptable_scope": [
            "protesting security guard id verification", "allow illegal substance smuggling",
            "frivolous prank emergency alarms", "refusal vehicle parking sticker"
        ],
        "default_advisory": (
            "Priority Security Grievance under UGC Directives & Security Manual Sec 2 (SLA: 4h). Campus security "
            "and duty proctor have been alerted to investigate and review CCTV records."
        ),
        "violation_reason": (
            "Rejected under Security Manual Sec 2: Mandatory ID verification, gate frisking, and anti-contraband "
            "rules are statutory campus safety controls."
        ),
    },
    {
        "doc_code": "POL-LIB-001",
        "department": "Library",
        "title": "Central Library Circulation, Fines & Study Zone Guidelines",
        "citation": "LPU Central Library Manual, Part IV",
        "sla_hours": 24,
        "content": (
            "Central Library book lending, RFID returns, digital databases (IEEE/ACM/Springer), and silent reading "
            "halls follow Library Manual Part IV (target SLA: 24 hours). Valid grievances include returned books still "
            "showing overdue fines, expired digital library credentials, noise disturbances in quiet study zones, "
            "broken reading hall AC/PCs, catalog/RFID gate errors, and book availability requests. "
            "Prohibited requests under Sec 4.5: waiving replacement charges for lost/damaged books, taking rare "
            "reference-only volumes off-campus, or tearing/stealing library materials."
        ),
        "acceptable_scope": [
            "returned book overdue fine", "digital library ieee access expired",
            "noise talking quiet study zone", "library reading hall broken", "rfid gate error",
            "library books catalog availability"
        ],
        "unacceptable_scope": [
            "exemption lost book replacement cost", "borrow reference rare books off campus",
            "stealing library books tearing pages"
        ],
        "default_advisory": (
            "Valid Library Grievance under Library Manual Part IV (SLA: 24h). The circulation desk will audit "
            "RFID return logs and update your library account."
        ),
        "violation_reason": (
            "Rejected under Library Manual Sec 4.5: Replacement charges for lost/damaged university books and "
            "reference-only circulation rules are non-negotiable."
        ),
    },
    {
        "doc_code": "POL-EXAM-001",
        "department": "Examination",
        "title": "Examination Ordinances, 75% Attendance Rule & Evaluation Redressal",
        "citation": "LPU Examination Ordinances, Clause 14",
        "sla_hours": 36,
        "content": (
            "Examinations, admit cards, internal marks, re-evaluation, and eligibility rules are governed by LPU "
            "Examination Ordinances Clause 14 (target SLA: 36 hours). Minimum 75% attendance is mandatory to appear "
            "in end-term examinations (medical condonation requires formal verified approval). Valid grievances include "
            "missing internal marks on UMS, corrupted admit card/hall ticket, two exams clashing in the same slot, "
            "delayed re-evaluation results, wrong subject codes on date sheet, and scribe accommodation for injured "
            "students. Prohibited requests under Sec 14.8: requesting advance/leaked question papers, bypassing the "
            "75% attendance detention rule without medical sanction, canceling UFM action without committee hearing, "
            "or demanding unearned passing marks."
        ),
        "acceptable_scope": [
            "internal marks missing ums grade card", "hall ticket admit card download error",
            "two exams same date slot clash", "re evaluation result delayed",
            "scribe accommodation injured student", "wrong subject code date sheet"
        ],
        "unacceptable_scope": [
            "advance examination question papers", "detention attendance below 75 without medical",
            "cancellation unfair means ufm without committee", "passing grades without taking exam"
        ],
        "default_advisory": (
            "Valid Examination Grievance under Examination Ordinances Clause 14 (SLA: 36h). Forwarded to the "
            "Controller of Examinations desk for roster verification."
        ),
        "violation_reason": (
            "Rejected under Examination Ordinances Sec 14.8: Demands for leaked exam papers, unearned marks, or "
            "bypassing mandatory 75% attendance eligibility strictly contravene university statutes."
        ),
    },
    {
        "doc_code": "POL-DISC-001",
        "department": "Discipline",
        "title": "Student Code of Conduct, Staff Behavior & Proctorial Guidelines",
        "citation": "LPU Student Handbook 2026, Code of Conduct Sec 2",
        "sla_hours": 24,
        "content": (
            "Campus harmony, hostel decorum, roommate conflicts, and staff/student conduct fall under Code of Conduct "
            "Sec 2 (target SLA: 24 hours). Valid grievances include roommate disputes, loud music during quiet/exam "
            "hours, rude or unprofessional staff behavior, cyberbullying in campus groups, and club/committee disputes. "
            "Prohibited actions under Sec 2.9: demanding the university ignore physical assault, alcohol/drugs/smoking "
            "violations, canceling active suspensions without proctorial inquiry, or filing knowingly false complaints."
        ),
        "acceptable_scope": [
            "roommate dispute sleep disruption", "loud music exam study week",
            "cyber bullying defamation university group", "rude staff misbehavior unprofessional",
            "student club committee conflict"
        ],
        "unacceptable_scope": [
            "ignore physical assault drug possession", "removal suspension without inquiry",
            "malicious knowingly false counter complaints"
        ],
        "default_advisory": (
            "Valid Conduct Grievance under Student Handbook Sec 2 (SLA: 24h). Proctorial officer notified to "
            "review the incident confidentially and initiate mediation/action."
        ),
        "violation_reason": (
            "Rejected under Code of Conduct Sec 2.9: Requests to bypass disciplinary inquiries or conceal "
            "substance/assault violations cannot be accepted."
        ),
    },
    {
        "doc_code": "POL-GRIEV-001",
        "department": "General",
        "title": "Student Grievance Filing, Escalation & Sentiment-Verified Closure Procedure",
        "citation": "LPU Student Grievance Redressal Charter, Sec 1",
        "sla_hours": 24,
        "content": (
            "How the Student Grievance System works: (1) Students log in using their University Registration Number "
            "and submit their complaint under 'Log Request'. (2) Our AI NLP classifier automatically predicts the "
            "right department with confidence %, assigns a numerical Urgency Score (0-100) into High (>=70), Medium "
            "(40-69), or Low (<40) priority, checks university policies via RAG, and routes instantly to the officer "
            "with the lowest active workload. (3) Students can track live progress, reply in the case thread, or "
            "escalate urgent unresolved tickets. (4) Once the officer completes the work, the status becomes "
            "'Completed - Awaiting Feedback'. (5) Final case closure is strictly enforced by Central Administration "
            "ONLY after the student submits Positive satisfaction feedback verified by our AI Sentiment Classifier; "
            "Negative feedback blocks closure and triggers re-routing."
        ),
        "acceptable_scope": [
            "how to submit grievance", "track complaint status", "how closure works",
            "escalation process", "feedback sentiment verification"
        ],
        "unacceptable_scope": [],
        "default_advisory": (
            "Your grievance is tracked end-to-end under LPU Grievance Charter Sec 1. Final closure occurs only "
            "after you verify resolution with positive satisfaction feedback."
        ),
        "violation_reason": "",
    },
]

_POLICY_BY_CODE = {p["doc_code"]: p for p in DEFAULT_UNIVERSITY_POLICIES}

PROHIBITED_PATTERNS = [
    (r'\b(bypass|hack|crack|tamper)\s+(firewall|proxy|attendance|password|server)\b',
     "Attempt to bypass campus security, firewall, or attendance rules"),
    (r'\b(torrent|torrents|gambling|crypto\s*mining|dark\s*web)\b',
     "Unauthorized use of campus network/power for prohibited services"),
    (r'\b(leak|leakage|advance)\s+(?:the\s+)?(?:end[\s-]?term\s+)?(?:exam(?:ination)?\s+)?(paper|question|solution)\b',
     "Solicitation of confidential examination material"),
    (r'\b(cancel|waive)\s+(attendance|suspension|ufm|punishment)\s+(without|mandatory)\b',
     "Request to bypass statutory disciplinary or attendance procedures"),
    (r'\b(free\s+degree|give\s+me\s+passing\s+marks|increase\s+marks\s+without)\b',
     "Request for unearned academic grading manipulation"),
    (r'\b(jacuzzi|luxury\s+bathtub|personal\s+maid|private\s+chef)\b',
     "Out-of-scope personal luxury modification request"),
    (r'\b(fake|fraudulent|forged)\s+(certificate|receipt|medical)\b',
     "Mention or submission of forged/fraudulent documentation"),
]

_EXTRA_STOPWORDS = {
    'please', 'want', 'need', 'request', 'requesting', 'demanding', 'demand',
    'campus', 'university', 'student', 'college', 'lpu', 'hostel', 'problem', 'issue'
}


def _meaningful_tokens(text):
    tokens = re.findall(r'[a-z0-9]+', normalize_text(text))
    return {t for t in tokens if t not in ENGLISH_STOP_WORDS and t not in _EXTRA_STOPWORDS and len(t) > 1}


# ==============================================================================
# 2. RAG VECTOR INDEX (BACKED BY MYSQL `rag_documents` + EMBEDDINGS)
# ==============================================================================

class UniversityRAGIndex:
    def __init__(self):
        self.documents = []
        self.vectorizer = None
        self.doc_matrix = None
        self._seeded = False

    def ensure_seeded(self):
        if self._seeded:
            return
        try:
            existing = execute_query("SELECT doc_code FROM rag_documents", fetchall=True) or []
            existing_codes = {r["doc_code"] for r in existing}
            for p in DEFAULT_UNIVERSITY_POLICIES:
                if p["doc_code"] not in existing_codes:
                    execute_query(
                        """INSERT INTO rag_documents (doc_code, title, department, citation, content, source_type)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (p["doc_code"], p["title"], p["department"], p["citation"], p["content"], "policy"),
                        commit=True
                    )
            self._seeded = True
        except Exception as e:
            print(f"[RAG Warning] Could not seed rag_documents in MySQL: {e}")

    def rebuild_index(self, persist_embeddings=False):
        self.ensure_seeded()
        try:
            rows = execute_query(
                "SELECT id, doc_code, title, department, citation, content, source_type, created_at FROM rag_documents ORDER BY id ASC",
                fetchall=True
            ) or []
        except Exception as e:
            print(f"[RAG Warning] Falling back to in-memory default policies ({e})")
            rows = [
                {
                    "id": i + 1,
                    "doc_code": p["doc_code"],
                    "title": p["title"],
                    "department": p["department"],
                    "citation": p["citation"],
                    "content": p["content"],
                    "source_type": "policy",
                }
                for i, p in enumerate(DEFAULT_UNIVERSITY_POLICIES)
            ]

        self.documents = rows
        corpus = [
            f"{d['title']} {d['department']} {d['citation']} {normalize_text(d['content'])}"
            for d in self.documents
        ]
        if not corpus:
            self.vectorizer = None
            self.doc_matrix = None
            return

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True)
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

        if persist_embeddings:
            feature_names = self.vectorizer.get_feature_names_out()
            for idx, doc in enumerate(self.documents):
                row_vec = self.doc_matrix[idx]
                top_pairs = sorted(
                    zip(row_vec.indices, row_vec.data),
                    key=lambda x: x[1],
                    reverse=True
                )[:40]
                emb = {str(feature_names[i]): round(float(val), 4) for i, val in top_pairs}
                try:
                    execute_query(
                        "UPDATE rag_documents SET embedding_json = %s WHERE id = %s",
                        (json.dumps(emb), doc["id"]),
                        commit=True
                    )
                except Exception:
                    pass

    def _ensure_ready(self):
        if self.vectorizer is None or not self.documents:
            self.rebuild_index(persist_embeddings=True)

    def retrieve(self, query_text, department=None, top_k=3):
        self._ensure_ready()
        if not query_text or not self.vectorizer or not self.documents:
            return []

        q_vec = self.vectorizer.transform([normalize_text(query_text)])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]

        scored = []
        for idx, score in enumerate(sims):
            doc = self.documents[idx]
            boosted = float(score)
            if department and doc.get("department") == department:
                boosted += 0.18
            scored.append((boosted, float(score), doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for boosted, raw_sim, doc in scored[:top_k]:
            item = dict(doc)
            item["similarity"] = round(raw_sim, 4)
            item["score"] = round(boosted, 4)
            meta = _POLICY_BY_CODE.get(doc.get("doc_code"), {})
            item["sla_hours"] = meta.get("sla_hours", 24)
            item["default_advisory"] = meta.get("default_advisory", "")
            item["violation_reason"] = meta.get("violation_reason", "")
            item["acceptable_scope"] = meta.get("acceptable_scope", [])
            item["unacceptable_scope"] = meta.get("unacceptable_scope", [])
            results.append(item)
        return results


_rag_index = UniversityRAGIndex()


def list_policy_documents():
    """Return all university policies and uploaded documents stored in MySQL."""
    _rag_index._ensure_ready()
    return list(_rag_index.documents)


def add_policy_document(title, department, citation, content):
    """Store an uploaded policy document in MySQL `rag_documents` and refresh TF-IDF embeddings."""
    doc_code = f"UPLOAD-{uuid.uuid4().hex[:8].upper()}"
    doc_id = execute_query(
        """INSERT INTO rag_documents (doc_code, title, department, citation, content, source_type)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (doc_code, title[:200], department[:100], citation[:200], content[:15000], "uploaded"),
        commit=True
    )
    _rag_index.rebuild_index(persist_embeddings=True)
    return doc_id, doc_code


def delete_policy_document(doc_id):
    """Delete an admin-uploaded policy document (default core policies stay protected)."""
    row = execute_query("SELECT id, source_type FROM rag_documents WHERE id = %s", (doc_id,), fetchone=True)
    if not row or row.get("source_type") != "uploaded":
        return False
    execute_query("DELETE FROM rag_documents WHERE id = %s", (doc_id,), commit=True)
    _rag_index.rebuild_index(persist_embeddings=True)
    return True


def retrieve_policies(query_text, department=None, top_k=3):
    return _rag_index.retrieve(query_text, department=department, top_k=top_k)


# ==============================================================================
# 3. GRIEVANCE POLICY COMPLIANCE & RAG GROUNDING
# ==============================================================================

def evaluate_complaint_policy(description, department=None):
    """
    Evaluates a grievance against official university terms, rules, and uploaded documents.
    Returns:
      {
        status: 'Acceptable' | 'Conditional' | 'Policy Violation',
        policy_score: float (0-100),
        citation: str,
        title: str,
        sla_hours: int,
        advisory: str,
        violation_reasons: list[str]
      }
    """
    clean_desc = (description or "").strip()
    if len(clean_desc) < 8:
        return {
            "status": "Conditional",
            "policy_score": 45.0,
            "citation": "LPU Student Grievance Charter, Sec 1",
            "title": "Submission Detail Guidelines",
            "sla_hours": 24,
            "advisory": "Please include specific details (location/block, date, or error message) for faster redressal.",
            "violation_reasons": ["Description is very brief."]
        }

    top_docs = retrieve_policies(clean_desc, department=department, top_k=2)
    best_doc = top_docs[0] if top_docs else {
        "title": "LPU Standard Student Grievance Policy",
        "citation": "LPU Student Grievance Charter, Sec 1",
        "sla_hours": 24,
        "similarity": 0.0,
        "default_advisory": "Grievance logged under University Student Redressal Charter.",
        "violation_reason": "This request falls outside university grievance redressal policy.",
        "acceptable_scope": [],
        "unacceptable_scope": [],
        "content": ""
    }

    # 1. Explicit policy violation pattern check
    for pattern, label in PROHIBITED_PATTERNS:
        if re.search(pattern, clean_desc, re.IGNORECASE):
            return {
                "status": "Policy Violation",
                "policy_score": 18.0,
                "citation": best_doc["citation"],
                "title": best_doc["title"],
                "sla_hours": best_doc.get("sla_hours", 24),
                "advisory": best_doc.get("violation_reason") or (
                    f"Flagged under {best_doc['citation']}: {label}. Please review university terms."
                ),
                "violation_reasons": [label]
            }

    # 2. Meaningful token overlap check against unacceptable scope
    desc_tokens = _meaningful_tokens(clean_desc)
    unacceptable_hits = []
    for unacc in best_doc.get("unacceptable_scope", []):
        u_tokens = _meaningful_tokens(unacc)
        if not u_tokens:
            continue
        overlap = len(u_tokens & desc_tokens)
        if overlap >= 3 or (len(u_tokens) <= 3 and overlap >= 2):
            unacceptable_hits.append(unacc)

    if unacceptable_hits:
        return {
            "status": "Policy Violation",
            "policy_score": 25.0,
            "citation": best_doc["citation"],
            "title": best_doc["title"],
            "sla_hours": best_doc.get("sla_hours", 24),
            "advisory": best_doc.get("violation_reason") or "Request conflicts with university policy terms.",
            "violation_reasons": [f"Conflicts with prohibited rule: '{u}'" for u in unacceptable_hits]
        }

    # 3. Positive alignment & uploaded document grounding
    pos_hits = 0
    for acc in best_doc.get("acceptable_scope", []):
        a_tokens = _meaningful_tokens(acc)
        if a_tokens and len(a_tokens & desc_tokens) >= 2:
            pos_hits += 1

    sim = float(best_doc.get("similarity", 0.0))
    score = min(98.0, max(68.0, round(58.0 + (sim * 60.0) + (pos_hits * 12.0), 1)))

    if best_doc.get("default_advisory"):
        advisory = best_doc["default_advisory"]
    else:
        # Synthesize concise excerpt from uploaded policy document
        excerpt = best_doc.get("content", "").strip()
        advisory = f"Matched university document '{best_doc['title']}' ({best_doc['citation']}): {excerpt[:220]}"

    return {
        "status": "Acceptable",
        "policy_score": score,
        "citation": best_doc["citation"],
        "title": best_doc["title"],
        "sla_hours": best_doc.get("sla_hours", 24),
        "advisory": advisory,
        "violation_reasons": []
    }


# ==============================================================================
# 4. AI STUDENT ASSISTANCE CHATBOT (LOCAL RAG + ML ROUTER + LIVE TICKET STATUS)
# ==============================================================================

_GREETING_RE = re.compile(r'^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|namaste|help|what\s+can\s+you\s+do)\b', re.IGNORECASE)
_STATUS_RE = re.compile(r'\b(my\s+ticket|my\s+complaint|my\s+grievance|ticket\s*#?\s*\d+|complaint\s*#?\s*\d+|check\s+status|track\s+my|how\s+many\s+complaints)\b', re.IGNORECASE)
_POLICY_QUERY_RE = re.compile(
    r'\b(policy|policies|rule|rules|regulation|terms|procedure|process|how\s+do\s+i|how\s+to|what\s+is\s+the|'
    r'attendance|75%|sla|deadline|refund\s+policy|anti[\s-]?ragging|closure|allowed|prohibited|guideline|bylaw)\b',
    re.IGNORECASE
)
_COMPLAINT_SIGNAL_RE = re.compile(
    r'\b(not\s+working|broken|leaking|leak|sparks|fire|stolen|theft|charged\s+twice|deducted\s+twice|'
    r'stale|cockroach|hair\s+in|missing\s+marks|clash|rude|ragging|harass|disconnected|slow\s+wifi|'
    r'500\s+error|internal\s+error|overdue\s+fine|no\s+water|power\s+cut|lift\s+stuck|file\s+a\s+complaint)\b',
    re.IGNORECASE
)


def _extract_relevant_sentences(content, query, max_sentences=3):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content or "") if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    q_tokens = _meaningful_tokens(query)
    scored = []
    for idx, sent in enumerate(sentences):
        s_tokens = _meaningful_tokens(sent)
        overlap = len(q_tokens & s_tokens)
        scored.append((overlap, -idx, sent))
    scored.sort(reverse=True)
    chosen = [s for _, _, s in scored[:max_sentences]]
    return " ".join(chosen)


def generate_chatbot_response(user_message, student_id=None, student_name=None):
    """
    Conversational AI Student Assistant powered by local RAG, ML classifier,
    Urgency Scorer (0-100), and live MySQL ticket lookup.
    """
    msg = (user_message or "").strip()
    if not msg:
        return {
            "reply": "Please type a question about university rules/policies, your grievance status, or describe an issue you need help with.",
            "intent": "empty",
            "sources": [],
            "analysis": None,
            "suggested_grievance": None
        }

    # 1. Greeting / Help intent
    if _GREETING_RE.search(msg) and len(msg.split()) <= 7:
        name_part = f" {student_name.split()[0]}" if student_name else ""
        return {
            "reply": (
                f"Hello{name_part}! I am the LPU AI Student Assistant. I can help you:\n"
                "• Answer questions on university rules, terms & policies (attendance, hostel, Wi-Fi, fees/refunds, exams, library, anti-ragging)\n"
                "• Check the live status of your submitted grievances\n"
                "• Preview which department, urgency score (0-100), and policy applies before you submit a complaint\n\n"
                "Ask me anything or describe your campus issue!"
            ),
            "intent": "greeting",
            "sources": [],
            "analysis": None,
            "suggested_grievance": None
        }

    # 2. Live Ticket Status Lookup (if student is logged in)
    if _STATUS_RE.search(msg):
        if not student_id:
            return {
                "reply": "Please sign in to the Student Portal with your Registration Number so I can check your personal grievance tickets.",
                "intent": "status",
                "sources": [],
                "analysis": None,
                "suggested_grievance": None
            }

        ticket_match = re.search(r'#?\b(\d+)\b', msg)
        if ticket_match:
            tid = int(ticket_match.group(1))
            row = execute_query(
                """SELECT c.id, c.predicted_department, c.priority_level, c.urgency_score, c.status,
                          c.completion_note, c.created_at, f.name AS faculty_name
                   FROM complaints c
                   LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
                   WHERE c.id = %s AND c.student_id = %s""",
                (tid, student_id),
                fetchone=True
            )
            if not row:
                return {
                    "reply": f"I could not find Grievance Ticket #{tid} under your account. Try asking 'check my tickets' to view your recent submissions.",
                    "intent": "status",
                    "sources": [],
                    "analysis": None,
                    "suggested_grievance": None
                }
            officer = row.get("faculty_name") or f"{row['predicted_department']} Desk"
            urg = row.get("urgency_score") or 25
            note_part = f"\nResolution Remarks: {row['completion_note']}" if row.get("completion_note") else ""
            return {
                "reply": (
                    f"Ticket #{row['id']} — {row['predicted_department']} Department\n"
                    f"• Status: {row['status']}\n"
                    f"• Priority: {row['priority_level']} (Urgency Score: {urg}/100)\n"
                    f"• Assigned Officer: {officer}\n"
                    f"• Lodged: {row['created_at']}{note_part}"
                ),
                "intent": "status",
                "sources": [],
                "analysis": None,
                "suggested_grievance": None
            }

        rows = execute_query(
            """SELECT c.id, c.predicted_department, c.priority_level, c.urgency_score, c.status, f.name AS faculty_name
               FROM complaints c
               LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
               WHERE c.student_id = %s
               ORDER BY c.id DESC LIMIT 5""",
            (student_id,),
            fetchall=True
        ) or []
        if not rows:
            return {
                "reply": "You haven't logged any grievances yet. Describe your issue here or use the 'Log Request' tab to submit one!",
                "intent": "status",
                "sources": [],
                "analysis": None,
                "suggested_grievance": None
            }
        lines = [f"Here are your latest {len(rows)} grievance ticket(s):"]
        for r in rows:
            urg = r.get("urgency_score") or 25
            officer = r.get("faculty_name") or r["predicted_department"]
            lines.append(f"• #{r['id']} [{r['predicted_department']}] — {r['status']} | {r['priority_level']} ({urg}/100) | Officer: {officer}")
        lines.append("\nYou can also ask 'status of ticket #<id>' for full details.")
        return {
            "reply": "\n".join(lines),
            "intent": "status",
            "sources": [],
            "analysis": None,
            "suggested_grievance": None
        }

    # 3. Decide between Complaint Analysis vs University Policy/Rules Q&A
    is_policy_question = bool(_POLICY_QUERY_RE.search(msg)) or msg.endswith('?')
    is_complaint_desc = bool(_COMPLAINT_SIGNAL_RE.search(msg))

    # 3A. Complaint Pre-Analysis + RAG Policy Grounding
    if is_complaint_desc and not is_policy_question:
        pred = predict_department_with_confidence(msg)
        urg_score, priority, urg_reasons = calculate_urgency(msg)
        policy_eval = evaluate_complaint_policy(msg, department=pred["department"])

        alt_text = ""
        if pred["alternatives"]:
            alt_parts = [f"{a['department']} ({a['confidence']}%)" for a in pred["alternatives"]]
            alt_text = f" | Alternatives: {', '.join(alt_parts)}"

        if policy_eval["status"] == "Policy Violation":
            reply = (
                f"Policy Alert ({policy_eval['citation']}):\n"
                f"{policy_eval['advisory']}\n\n"
                f"• Predicted Department: {pred['department']} ({pred['confidence']}%)\n"
                f"• Policy Verdict: {policy_eval['status']}"
            )
            suggested = None
        else:
            reply = (
                f"I analyzed your issue using our AI Router & University Policy RAG:\n"
                f"• Predicted Department: {pred['department']} ({pred['confidence']}% confidence{alt_text})\n"
                f"• Urgency Score: {urg_score}/100 — {priority}\n"
                f"• Applicable Policy: {policy_eval['citation']} (Est. SLA: {policy_eval['sla_hours']}h)\n"
                f"• Guidance: {policy_eval['advisory']}\n\n"
                f"Click 'Fill in Grievance Form' below to lodge this ticket immediately."
            )
            suggested = msg

        return {
            "reply": reply,
            "intent": "grievance_analysis",
            "sources": [{"title": policy_eval["title"], "citation": policy_eval["citation"]}],
            "analysis": {
                "department": pred["department"],
                "confidence": pred["confidence"],
                "alternatives": pred["alternatives"],
                "urgency_score": urg_score,
                "priority": priority,
                "policy_status": policy_eval["status"],
                "citation": policy_eval["citation"],
                "sla_hours": policy_eval["sla_hours"]
            },
            "suggested_grievance": suggested
        }

    # 3B. University Rules, Policies & Uploaded Documents RAG Q&A
    docs = retrieve_policies(msg, top_k=2)
    if docs and docs[0]["similarity"] >= 0.04:
        primary = docs[0]
        excerpt = _extract_relevant_sentences(primary["content"], msg, max_sentences=3)
        sources = [{"title": d["title"], "citation": d["citation"]} for d in docs if d["similarity"] >= 0.04]

        sla_line = f"\n• Standard Resolution SLA: {primary['sla_hours']} hours ({primary['department']} Department)" if primary.get("sla_hours") else ""
        reply = (
            f"According to {primary['title']} ({primary['citation']}):\n\n"
            f"{excerpt}{sla_line}\n\n"
            f"Need to file an official grievance for this? Just describe your specific problem or use the 'Log Request' tab."
        )
        return {
            "reply": reply,
            "intent": "policy_rag",
            "sources": sources,
            "analysis": None,
            "suggested_grievance": None
        }

    # 4. General fallback: analyze as potential issue and guide student
    pred = predict_department_with_confidence(msg)
    urg_score, priority, _ = calculate_urgency(msg)
    policy_eval = evaluate_complaint_policy(msg, department=pred["department"])
    return {
        "reply": (
            f"Based on your query, the closest campus desk is **{pred['department']}** "
            f"(Confidence: {pred['confidence']}%, Urgency Score: {urg_score}/100 — {priority}).\n"
            f"• Reference Policy: {policy_eval['citation']}\n"
            f"• Guidance: {policy_eval['advisory']}\n\n"
            f"You can ask about specific university rules (e.g., attendance, hostel, Wi-Fi, fees, exams, library, anti-ragging) or click below to lodge this as a grievance."
        ),
        "intent": "general",
        "sources": [{"title": policy_eval["title"], "citation": policy_eval["citation"]}],
        "analysis": {
            "department": pred["department"],
            "confidence": pred["confidence"],
            "urgency_score": urg_score,
            "priority": priority,
            "citation": policy_eval["citation"]
        },
        "suggested_grievance": msg if len(msg) >= 15 else None
    }
