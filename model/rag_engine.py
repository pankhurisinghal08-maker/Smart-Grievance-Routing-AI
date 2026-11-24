"""
Smart Grievance Routing AI - Faculty-Grade ChromaDB RAG Engine & AI Student Assistant
Powered by ChromaDB Vector Database (Dense Embeddings) & Google Gemini AI API
"""

import json
import os
import re
import uuid
import numpy as np

# ChromaDB vector store imports
import chromadb
from chromadb.utils import embedding_functions

from database.db import execute_query
from model.nlp_model import normalize_text, predict_department_with_confidence
from model.rules import calculate_urgency

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "database", "chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

_gemini_client = None
_gemini_cached_key = None


def _get_gemini_client():
    """Return an initialized google-genai Client if GEMINI_API_KEY is configured, else None."""
    global _gemini_client, _gemini_cached_key
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key or api_key.lower().startswith("your_"):
        return None
    if _gemini_client is not None and _gemini_cached_key == api_key:
        return _gemini_client
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=api_key)
        _gemini_cached_key = api_key
        return _gemini_client
    except Exception as e:
        print(f"[Gemini Init Warning] Could not initialize google-genai client: {e}")
        return None


def is_gemini_configured():
    return _get_gemini_client() is not None


# ==============================================================================
# 1. OFFICIAL UNIVERSITY TERMS, RULES & POLICIES KNOWLEDGE BASE (EXPANDED LOCAL DATA)
# ==============================================================================

DEFAULT_UNIVERSITY_POLICIES = [
    {
        "doc_code": "POL-EXAM-001",
        "department": "Examination",
        "title": "75% Attendance Requirement, Medical Condonation & Exam Eligibility",
        "citation": "LPU Examination Ordinances, Clause 14 & Prospectus Part C",
        "sla_hours": 36,
        "content": (
            "Attendance Policy & Exam Eligibility: Students are required to maintain a minimum of 75% attendance in "
            "each registered course (theory and practical) to be eligible to appear in the End-Term Examinations. "
            "A condonation of up to 5% (relaxation down to 70%) may be granted on verified medical grounds or genuine "
            "exigencies, subject to timely submission of medical certificates to the Head of Department / Mentor within "
            "3 working days of resumption. Students whose attendance falls below 65% are strictly detained (awarded 'F' "
            "grade) and must re-register the course during backlog terms. Duty Leaves (OD) for university-sanctioned events "
            "(sports, NCC, technical hackathons, cultural fests) must be endorsed by the faculty coordinator on UMS."
        ),
        "acceptable_scope": [
            "attendance requirement aggregate percentage", "medical leave condonation 75 percent", "detention below 65 percent",
            "duty leave od approval", "eligible for end term exams attendance", "biometric attendance shortage", "mentor medical proof"
        ],
        "unacceptable_scope": ["waive attendance without medical certificate", "bypass 75 percent detention illegally", "fake medical certificate"],
        "default_advisory": (
            "Under LPU Examination Ordinances Clause 14, 75% minimum attendance is required per course. Medical condonation "
            "(up to 5%) requires valid medical documentation submitted to your mentor within 3 days. Below 65% triggers detention."
        ),
        "violation_reason": "Bypassing the statutory 75% attendance rule without official medical/mentor verification violates university academic ordinances."
    },
    {
        "doc_code": "POL-EXAM-002",
        "department": "Examination",
        "title": "Date Sheet Slot Clashes, Admit Card Correction & Hall Ticket Clearance",
        "citation": "LPU Examination Guidelines & Admit Card Protocols",
        "sla_hours": 24,
        "content": (
            "Admit Cards (Hall Tickets) & Exam Scheduling: Admit cards are released on UMS 7 to 10 days before term exams. "
            "Clearance of semester tuition fees and meeting minimum attendance criteria are prerequisites for downloading admit cards. "
            "If two examinations clash in the same morning/afternoon slot on the official date sheet, students must report via UMS "
            "Log Request within 48 hours of date sheet publication for slot rescheduling. Admit card errors (incorrect subject code, "
            "misprinted candidate name or photo mismatch) are corrected by the Controller of Examinations (COE) desk within 24 hours."
        ),
        "acceptable_scope": [
            "two exams same date slot clash", "hall ticket admit card download error", "wrong subject code date sheet",
            "photo missing admit card", "fee cleared admit card blocked", "exam timing slot overlap", "ums date sheet clash"
        ],
        "unacceptable_scope": ["postpone exam for personal vacation", "request advance exam question paper", "skip exam without medical"],
        "default_advisory": (
            "Valid Examination Scheduling Grievance. Forwarded to the Controller of Examinations desk (SLA: 24h). For exam clashes, "
            "attach your date sheet screenshot and subject codes."
        ),
        "violation_reason": "Examination rescheduling is permitted solely for official timetable clashes or verified medical emergencies."
    },
    {
        "doc_code": "POL-EXAM-003",
        "department": "Examination",
        "title": "Evaluation Redressal, Re-evaluation Turnaround & Script Scrutiny",
        "citation": "LPU Evaluation Redressal Code & Ordinance Clause 16",
        "sla_hours": 48,
        "content": (
            "Evaluation & Re-evaluation Norms: Students dissatisfied with their evaluated grade or marks can apply for re-evaluation "
            "or physical/digital scrutiny of answer scripts through UMS within 15 working days from result declaration. The re-evaluation "
            "is conducted by an independent external evaluator. If the revised score shows a variance of 10% or more, the re-evaluation "
            "application fee is refunded to the student's fee ledger. Missing internal assessment or continuous evaluation (CA) marks "
            "must be escalated to the course instructor before theory exam commencement."
        ),
        "acceptable_scope": [
            "re evaluation result delayed", "internal marks missing ums grade card", "scrutiny answer sheet marks totaling error",
            "grade revision result declared", "recheck exam paper fee refund", "continuous assessment marks mismatch instructor"
        ],
        "unacceptable_scope": ["demand passing marks without re-evaluation", "bribe or coerce evaluator", "alter evaluated grade sheet"],
        "default_advisory": (
            "Evaluation grievance logged under LPU Evaluation Redressal Code (SLA: 48h). Re-evaluation requests must be filed within "
            "15 days of result declaration via UMS."
        ),
        "violation_reason": "Arbitrary mark increases without formal re-evaluation or evaluation scrutiny violate academic grading standards."
    },
    {
        "doc_code": "POL-EXAM-004",
        "department": "Examination",
        "title": "Scribe Allocation & Compensatory Extra Time for Differently-Abled/Injured Students",
        "citation": "LPU Disability & Inclusive Examination Policy",
        "sla_hours": 12,
        "content": (
            "Inclusive Examination & Scribe Accommodation: Differently-abled students or candidates suffering from acute physical injury "
            "(fractured writing arm, severe trauma) are entitled to an authorized examination scribe and compensatory extra time (20 minutes "
            "per hour of examination). Application must be submitted via UMS with verified government disability certificate or Chief Medical "
            "Officer (CMO) hospital documentation at least 3 days prior to the exam. Scribe qualification must be one academic tier junior to candidate."
        ),
        "acceptable_scope": [
            "scribe accommodation injured student", "extra time fractured hand exam", "differently abled student assistance",
            "compensatory time admit card", "scribe approval medical emergency"
        ],
        "unacceptable_scope": ["unauthorized external adult scribe", "requesting scribe without medical proof"],
        "default_advisory": (
            "Priority Medical Examination Accommodation logged (SLA: 12h). COE and DSW disability cell notified to assign an authorized scribe."
        ),
        "violation_reason": "Scribe allocation requires verified medical disability certificates under statutory university examination bylaws."
    },
    {
        "doc_code": "POL-ACAD-001",
        "department": "Examination",
        "title": "Class Schedule, Timetable Clashes & Section Allocation Rules",
        "citation": "LPU Academic Operations Regulations, Sec 3",
        "sla_hours": 24,
        "content": (
            "Timetable & Section Management: Academic timetables are published on UMS prior to semester commencement. Section change "
            "requests are entertained only within the first 5 instructional days on genuine grounds such as elective lecture overlap or "
            "official university transport batch clashes, subject to seat availability. Unscheduled faculty absenteeism or persistent class "
            "cancellations must be reported to the School Academic Coordinator for substitute lecture arrangement within 48 hours."
        ),
        "acceptable_scope": [
            "class timetable not updated ums", "lecture schedule clash two subjects", "section change request elective clash",
            "faculty absent persistent class canceled", "lab session timing overlap lecture", "classroom locked room changed"
        ],
        "unacceptable_scope": ["arbitrary section change to be with friends", "demand cancellation of mandatory lectures"],
        "default_advisory": (
            "Academic Operations ticket registered (SLA: 24h). School Academic Coordinator notified to inspect timetable/section allocation."
        ),
        "violation_reason": "Section changes are governed by academic batch balancing and cannot be granted for personal social convenience."
    },
    {
        "doc_code": "POL-ACAD-002",
        "department": "Examination",
        "title": "Grading System, SGPA/CGPA Calculation & Supplementary/Backlog Norms",
        "citation": "LPU Academic Curriculum Framework & Student Success Policy",
        "sla_hours": 36,
        "content": (
            "Grading & Backlog Clearance: LPU follows a 10-point relative and absolute grading scale (O, A+, A, B+, B, C, D, E/Reappear, F/Fail). "
            "A minimum of 40% aggregate in continuous assessment plus end-term evaluation is required to pass a course. Courses with 'E' or 'F' "
            "grades can be cleared by registering for Reappear / Supplementary Examinations offered during Autumn, Spring, and Summer terms. "
            "Make-up examinations are permitted solely for students who missed exams due to officially certified hospitalization or bereavement."
        ),
        "acceptable_scope": [
            "backlog registration supplementary exam", "make up examination hospitalization", "cgpa calculation sgpa discrepancy",
            "relative grading cut off query", "summer term course registration backlog", "pass criteria 40 percent aggregate"
        ],
        "unacceptable_scope": ["waive backlog without taking exam", "demand artificial cgpa boost", "bypass degree minimum credit rules"],
        "default_advisory": (
            "Academic Evaluation guidance under Curriculum Framework (SLA: 36h). Supplementary/Reappear registration windows are announced "
            "on UMS each term."
        ),
        "violation_reason": "Awarding passing grades or credit waivers without required evaluation examinations is strictly prohibited."
    },
    {
        "doc_code": "POL-DISC-001",
        "department": "Discipline",
        "title": "Student Code of Conduct, Campus Decorum & Proctorial Regulations",
        "citation": "LPU Student Handbook 2026, Code of Conduct Sec 2",
        "sla_hours": 24,
        "content": (
            "Student Decorum & Discipline: Students must observe exemplary conduct, mutual respect, and academic integrity. Mandatory wearing "
            "of the University Student ID Card visibly around the neck is required at all campus gates and academic blocks. Prohibited offenses "
            "include classroom disruption, unauthorized photography/videography of peers or faculty, cyberbullying in campus groups, "
            "destruction of university property, and possession of alcohol/tobacco/contraband. Violations trigger Proctorial Inquiry hearings."
        ),
        "acceptable_scope": [
            "roommate dispute sleep disruption", "loud music exam study week", "cyber bullying defamation university group",
            "rude staff misbehavior unprofessional", "student club committee conflict", "lost id card gate entry blocked"
        ],
        "unacceptable_scope": ["conceal physical violence substance abuse", "demand proctor ignore active suspension", "malicious false complaints"],
        "default_advisory": (
            "Proctorial Grievance logged under Code of Conduct Sec 2 (SLA: 24h). Proctorial officer alerted for confidential mediation and review."
        ),
        "violation_reason": "Acts endangering campus harmony, substance violations, or violent misconduct cannot be bypassed without formal proctorial inquiry."
    },
    {
        "doc_code": "POL-DISC-002",
        "department": "Discipline",
        "title": "Faculty Conduct, Teacher-Student Professionalism & Mentoring Norms",
        "citation": "LPU Faculty & Staff Code of Ethics & Academic Grievance Cell",
        "sla_hours": 24,
        "content": (
            "Faculty Conduct & Student Relationship: Faculty members are expected to maintain the highest standards of professionalism, "
            "fairness, and respectful pedagogical demeanor. Students have the right to report rude language, verbal humiliation, biased internal "
            "assessment, unauthorized cancellation of mentoring sessions, or refusal to resolve academic doubts. Complaints are submitted "
            "confidentially to the Head of Department and Academic Grievance Cell; students are strictly protected against academic retaliation."
        ),
        "acceptable_scope": [
            "faculty rude misbehavior classroom", "teacher unprofessional language humiliation", "biased internal assessment faculty grievance",
            "faculty absent mentoring meeting", "retaliation fear complaint teacher", "faculty discrimination classroom"
        ],
        "unacceptable_scope": ["frivolous defamation against teacher for strict grading", "blackmail faculty for unearned marks"],
        "default_advisory": (
            "Faculty Conduct Grievance logged with Academic Grievance Cell (SLA: 24h). The matter is forwarded confidentially to the Dean / HOD "
            "for inquiry with full whistleblower protection."
        ),
        "violation_reason": "All staff-student disputes must be processed via formal confidential academic inquiry to ensure fairness."
    },
    {
        "doc_code": "POL-SEC-001",
        "department": "Security",
        "title": "Zero-Tolerance Anti-Ragging Policy & 24/7 Emergency Security Code",
        "citation": "UGC Anti-Ragging Directives & LPU Security Manual, Sec 2",
        "sla_hours": 4,
        "content": (
            "Zero Tolerance on Ragging: Lovely Professional University strictly enforces zero tolerance for ragging, bullying, or harassment "
            "in compliance with Supreme Court and UGC directives. Emergency resolution SLA: 4 hours. Any incident of ragging results in immediate "
            "police FIR, hostel expulsion, and academic suspension. 24/7 National Anti-Ragging Helpline: 1800-180-5522. Campus Emergency Control: "
            "01824-444500. Campus security officers patrol hostels and academic blocks 24/7."
        ),
        "acceptable_scope": [
            "ragging incident hostel", "bullying harassment campus", "stolen laptop phone bicycle", "unauthorized entry hostel room",
            "broken cctv camera walkway", "dark streetlights night security", "sos emergency threat guard assistance"
        ],
        "unacceptable_scope": ["protesting security guard id check", "smuggling contraband campus", "false emergency prank alarm"],
        "default_advisory": (
            "EMERGENCY SECURITY / ANTI-RAGGING TICKET (SLA: 4h). Duty Proctor and Campus Security alerted immediately. For live threats, "
            "call 01824-444500 or 1800-180-5522 right now."
        ),
        "violation_reason": "Ragging and violent harassment are criminal offenses under Indian Penal Code and UGC statutory directives."
    },
    {
        "doc_code": "POL-SEC-002",
        "department": "Security",
        "title": "Internal Complaints Committee (ICC) & Gender Sensitization Charter",
        "citation": "POSH Act 2013 & LPU Internal Complaints Committee Regulations",
        "sla_hours": 6,
        "content": (
            "Gender Safety & Prevention of Sexual Harassment: LPU guarantees a safe campus for all students and staff. The Internal Complaints "
            "Committee (ICC) handles complaints of sexual harassment, stalking, unwelcome physical contact, or inappropriate digital communications. "
            "Contact: icc@lpu.co.in or +91-8283921144. Formal inquiry is initiated within statutory timeframes with interim safety measures "
            "(restraining orders, hostel reassignment) provided to the complainant."
        ),
        "acceptable_scope": [
            "sexual harassment stalker campus", "inappropriate message faculty student", "unwelcome physical contact hostel",
            "gender discrimination complaint icc", "safety escort late night campus"
        ],
        "unacceptable_scope": ["retaliatory fake complaints"],
        "default_advisory": (
            "ICC Grievance logged with highest priority (SLA: 6h). ICC Presiding Officer notified. Strict confidentiality and safety guaranteed."
        ),
        "violation_reason": "Matters of harassment are statutory legal inquiries governed by the POSH Act 2013."
    },
    {
        "doc_code": "POL-FIN-001",
        "department": "Finance",
        "title": "Tuition Fees, Payment Gateways & Late Fine Surcharges",
        "citation": "LPU Financial Ordinances, Chapter IX (Fees & Refunds)",
        "sla_hours": 48,
        "content": (
            "Tuition Fees & Payments: Semester fees must be cleared within scheduled deadlines via the UMS Payment Gateway (UPI, Net Banking, "
            "Credit/Debit cards, Education Loan Challans). Late fee fines accrue after the due date at prescribed daily rates. Double debit "
            "turnaround: If fees are deducted from the bank account but not updated on UMS, the transaction is auto-reconciled within 24 to 48 "
            "banking hours upon sharing the bank UTR transaction ID in the case thread."
        ),
        "acceptable_scope": [
            "fee payment deducted twice bank", "receipt not generated net banking", "incorrect late fine challan",
            "education loan disbursement verification", "semester fee payment gateway failure", "bank utr transaction status"
        ],
        "unacceptable_scope": ["demand blanket fee waiver without scholarship", "evade mandatory university fee dues"],
        "default_advisory": (
            "Finance Desk notified under Financial Ordinances Ch. IX (SLA: 48h). Please post your bank UTR number and transaction date "
            "in the case reply for automated payment reconciliation."
        ),
        "violation_reason": "Tuition fees and statutory late fine policies are regulated by the Board of Governors."
    },
    {
        "doc_code": "POL-FIN-002",
        "department": "Finance",
        "title": "Scholarship Retention, Ledger Adjustment & Caution Money Refunds",
        "citation": "LPU Scholarship Ordinances & Caution Deposit Bylaws",
        "sla_hours": 48,
        "content": (
            "Scholarships & Caution Deposit Refunds: Merit scholarships (LPUNEST, Board percentage, Sports/Cultural achievements) require "
            "maintaining the prescribed minimum CGPA and clean disciplinary record without debarment. Scholarship amounts are adjusted "
            "directly in the semester fee ledger. Caution money / security deposit is 100% refundable upon graduation or migration after "
            "completing online No-Dues clearance across department libraries, labs, and hostels."
        ),
        "acceptable_scope": [
            "scholarship not credited balance", "caution money security deposit refund", "lpunest scholarship slab renewal",
            "no dues certificate refund processing", "sports scholarship fee ledger adjustment"
        ],
        "unacceptable_scope": ["scholarship without meeting eligibility criteria", "cash refund without bank account"],
        "default_advisory": (
            "Scholarship & Refund Desk notified (SLA: 48h). Accounts department will audit your fee ledger and credit verified scholarship funds."
        ),
        "violation_reason": "Scholarship disbursement requires verified academic and disciplinary eligibility."
    },
    {
        "doc_code": "POL-IT-001",
        "department": "IT",
        "title": "Campus Wi-Fi, LAN & Network Infrastructure Policy",
        "citation": "LPU IT Code 2026, Sec 4.1 (Campus Digital Connectivity)",
        "sla_hours": 12,
        "content": (
            "Campus Connectivity: Wi-Fi access points, hostel LAN ports, and computing laboratories are maintained by IT with a target SLA "
            "of 12 hours. Valid complaints include weak hostel Wi-Fi signal, access point outages, slow download speeds, captive portal login "
            "timeouts, and lab projector malfunctions. Students must disable MAC address randomization on devices to maintain seamless captive "
            "portal authentication. Prohibited: bypassing campus firewall/proxy, setting up private Wi-Fi routers, or packet sniffing."
        ),
        "acceptable_scope": [
            "hostel wifi weak signal", "wifi access point down", "cannot connect campus network", "slow internet speed",
            "captive portal timeout", "router flashing red", "lab computer slow", "projector blue screen"
        ],
        "unacceptable_scope": ["bypass campus firewall proxy", "unblock torrents gambling", "unauthorized private routers"],
        "default_advisory": (
            "IT Infrastructure Grievance logged under IT Code Sec 4.1 (SLA: 12h). Disable MAC address randomization on your device while "
            "the network team inspects the nearest access point."
        ),
        "violation_reason": "Bypassing university firewall or setting up unauthorized routers is strictly prohibited under campus cybersecurity policy."
    },
    {
        "doc_code": "POL-IT-002",
        "department": "IT",
        "title": "UMS, LMS & Student Email Academic Portal Norms",
        "citation": "LPU Academic Computing Norms, Clause 5.2",
        "sla_hours": 24,
        "content": (
            "UMS & LMS Academic Systems: Governed under Clause 5.2 (target SLA: 24 hours). Valid reports include 500 internal server errors, "
            "LMS assignment submission timeouts, fee receipt download crashes, grade card display discrepancies, student email password resets, "
            "and biometric attendance sync mismatches. If an LMS server glitch impacts an assignment deadline, note your Ticket ID and submit "
            "to your course instructor for deadline amnesty."
        ),
        "acceptable_scope": [
            "ums fee receipt download error", "lms assignment portal crashed", "portal internal server error 500",
            "cannot view grade sheet ums", "student email password reset", "biometric attendance sync mismatch"
        ],
        "unacceptable_scope": ["hacking portal database", "unauthorized grade alteration", "stealing other student credentials"],
        "default_advisory": (
            "Portal Grievance logged under Clause 5.2 (SLA: 24h). UMS/LMS engineering desk alerted; quote your Ticket ID to instructors if deadlines are affected."
        ),
        "violation_reason": "Attempts to breach university databases or tamper with academic records trigger disciplinary and cyber law action."
    },
    {
        "doc_code": "POL-MAINT-001",
        "department": "Maintenance",
        "title": "Hostel & Campus Civil, Plumbing & Facilities Maintenance Policy",
        "citation": "LPU Estate & Hostel Maintenance Bylaws, Art. 8",
        "sla_hours": 24,
        "content": (
            "Civil & Plumbing Maintenance: Hostel rooms, corridors, classrooms, washrooms, and water coolers are maintained by Estate "
            "Maintenance (target SLA: 24 hours). Valid issues include pipe leakages, broken faucets, blocked washrooms, non-functional flushes, "
            "water supply shortages, dirty water cooler filters, ceiling seepage, broken window panes, broken door locks, and furniture repairs. "
            "Prohibited: personal luxury room remodeling or willful damage to university property."
        ),
        "acceptable_scope": [
            "water leaking washroom pipe", "broken tap sink flooding", "toilet flush not working", "no running water hostel",
            "water cooler filter dirty", "ceiling seepage plaster", "broken door lock window glass", "broken bench chair bed"
        ],
        "unacceptable_scope": ["personal luxury remodeling", "intentional property vandalism"],
        "default_advisory": (
            "Maintenance Grievance logged under Estate Bylaws Art. 8 (SLA: 24h). Duty plumber / civil technician assigned to inspect the fixture."
        ),
        "violation_reason": "Custom luxury civil alterations fall outside standard university hostel accommodation norms."
    },
    {
        "doc_code": "POL-MAINT-002",
        "department": "Maintenance",
        "title": "Campus Electrical, HVAC & Fire Safety Emergency Code",
        "citation": "LPU Electrical Safety & Fire Code 2026, Sec 12",
        "sla_hours": 6,
        "content": (
            "Electrical & Fire Safety: Emergency SLA: 6 hours. Handled with top priority. Valid reports include sparking switchboards, burning "
            "wire smells, short circuits, electric shocks from geysers/appliances, elevator / lift stoppages with trapped passengers, severe power cuts, "
            "and AC cooling failure in summer. Stay clear of wet sparking areas. Prohibited: unauthorized heating coils (2000W+), cryptocurrency mining, "
            "or tampering with circuit breakers."
        ),
        "acceptable_scope": [
            "electrical sparks switchboard", "short circuit burning wire", "ac cooling not working", "fan wobbling light flickering",
            "power cut fuse tripped", "elevator lift stuck", "geyser electric shock"
        ],
        "unacceptable_scope": ["unauthorized heavy heating coils", "tampering substation breakers"],
        "default_advisory": (
            "ELECTRICAL EMERGENCY LOGGED (SLA: 6h). Stay clear of wet or sparking fixtures. Duty electrician dispatched immediately."
        ),
        "violation_reason": "High-wattage unapproved appliances or tampering with breakers pose severe fire hazards."
    },
    {
        "doc_code": "POL-CAFE-001",
        "department": "Cafeteria",
        "title": "Cafeteria, Mess Hygiene & Food Safety Charter",
        "citation": "FSSAI Compliance & LPU Food Charter, Sec 3.4",
        "sla_hours": 12,
        "content": (
            "Food Safety & Mess Hygiene: Campus mess halls, canteens, and food kiosks operate under FSSAI standards and LPU Food Charter "
            "(target SLA: 12 hours). Valid grievances include stale or undercooked food, foreign objects/insects in meals, food poisoning symptoms, "
            "unhygienic utensils, food handlers without hairnets/gloves, unannounced price surges, and duplicate QR/UPI payments (auto-reconciled "
            "in 24-48 hours)."
        ),
        "acceptable_scope": [
            "uncooked stale food mess", "hair insect cockroach food", "canteen double charged qr payment",
            "food poisoning symptoms", "mess hygiene water dispenser", "canteen rates price hike"
        ],
        "unacceptable_scope": ["demanding free food without financial aid", "harassing cafeteria staff"],
        "default_advisory": (
            "Food Safety Grievance logged under FSSAI Charter Sec 3.4 (SLA: 12h). Mess supervisor alerted for food quality inspection."
        ),
        "violation_reason": "Disorderly conduct or unsubstantiated free food demands fall outside formal grievance redressal."
    },
    {
        "doc_code": "POL-LIB-001",
        "department": "Library",
        "title": "Central Library Circulation, Fines & Digital Research Databases",
        "citation": "LPU Central Library Manual, Part IV",
        "sla_hours": 24,
        "content": (
            "Central Library Services: Book lending, RFID automated returns, digital research portals (IEEE Xplore, ScienceDirect, ACM, Springer), "
            "and quiet study zones follow Library Manual Part IV (target SLA: 24 hours). Valid grievances: returned books still showing overdue fines, "
            "expired institutional database credentials, air conditioning failure in reading halls, and RFID self-kiosk errors."
        ),
        "acceptable_scope": [
            "returned book overdue fine", "digital library ieee access expired", "noise talking quiet study zone",
            "library reading hall broken", "rfid gate error", "library books catalog availability"
        ],
        "unacceptable_scope": ["refusal to pay lost book replacement fee", "tearing or damaging library books"],
        "default_advisory": (
            "Library Circulation Desk alerted under Library Manual Part IV (SLA: 24h). RFID logs will be audited to clear overdue records."
        ),
        "violation_reason": "Replacement fees for lost or defaced library volumes are statutory university requirements."
    },
    {
        "doc_code": "POL-HOSTEL-001",
        "department": "Hostel",
        "title": "Residential Living, Hostel Room Allotment, Curfew Hours & Outpass Rules",
        "citation": "LPU Residence Manual & Hostel Bylaws",
        "sla_hours": 24,
        "content": (
            "Hostel Life & Outpass Guidelines: Students residing in university hostels must adhere to designated curfew hours (9:30 PM for boys, "
            "8:30 PM for girls, unless extended for academic lab work). E-leave and outpass requests must be applied through the UMS Mobile App "
            "at least 6 hours in advance with parent consent confirmation. Room change requests are entertained during the designated window at the "
            "beginning of each semester upon clearance of dues."
        ),
        "acceptable_scope": [
            "hostel e leave outpass gate pass", "hostel room allotment accommodation", "hostel curfew late entry approval",
            "room change request semester beginning", "hostel warden grievance outpass delay", "hostel mess facility change"
        ],
        "unacceptable_scope": ["staying out overnight without outpass", "bringing unauthorized overnight guests"],
        "default_advisory": (
            "Hostel Administration notified (SLA: 24h). Submit e-leave through UMS Mobile App with parental approval for outpass clearance."
        ),
        "violation_reason": "Unauthorized overnight hostel absence violates campus safety and parental welfare agreements."
    },
    {
        "doc_code": "POL-GRIEV-001",
        "department": "General",
        "title": "Student Grievance Architecture: AI Routing, Urgency Scoring & 3-Tier Escalation",
        "citation": "LPU Student Grievance Redressal Charter, Sec 1",
        "sla_hours": 24,
        "content": (
            "How the Grievance System Works: (1) Students log complaints under 'Log Request'. (2) The AI NLP classifier predicts the appropriate "
            "department with confidence %, assigns an Urgency Score (0-100) into High (>=70), Medium (40-69), or Low (<40) priority, and routes "
            "to the least-loaded departmental officer. (3) 3-Tier Escalation: Level 1 (Faculty Officer), Level 2 (Head of Department), Level 3 "
            "(Central Grievance Redressal Cell / University Ombudsperson). (4) Once completed, case closure is verified by AI Sentiment Analysis "
            "of student feedback; negative feedback reopens the case."
        ),
        "acceptable_scope": [
            "how to submit grievance", "track complaint status", "how closure works", "escalation process", "feedback sentiment verification"
        ],
        "unacceptable_scope": [],
        "default_advisory": (
            "Track your ticket in real time under Log Status. Final case closure is verified through AI sentiment analysis of your feedback."
        ),
        "violation_reason": ""
    }
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
    (r'\b(fake|fraudulent|forged)\s+(certificate|receipt|medical)\b',
     "Mention or submission of forged/fraudulent documentation"),
]

_POLICY_DEPARTMENT_HINTS = [
    ("Examination", r'\b(exam|exams|examination|attendance|75%|admit\s*card|hall\s*ticket|re[\s-]?evaluation|date\s*sheet|backlog|reappear|marks|cgpa|sgpa|scribe)\b'),
    ("Discipline", r'\b(discipline|proctor|conduct|faculty\s*behavior|rude\s*teacher|misbehavior|harassment|ragging|bully|uniform|dress\s*code)\b'),
    ("Finance", r'\b(fee|fees|scholarship|refund|receipt|caution\s*money|utr|bank|challan|instalment|ledger)\b'),
    ("IT", r'\b(wifi|wi-fi|internet|ums|lms|portal|login|captive\s*portal|mac\s*address|lan|router|projector)\b'),
    ("Maintenance", r'\b(leak|leaking|pipe|washroom|toilet|water\s*cooler|flush|switchboard|sparks|short\s*circuit|geyser|lift|elevator|ac\s*not|fan)\b'),
    ("Cafeteria", r'\b(food|mess|canteen|hygiene|stale|meal|lunch|dinner|qr\s*payment|fssai)\b'),
    ("Library", r'\b(library|book|books|overdue|fine|rfid|ieee|acm|study\s*hall|reading\s*room)\b'),
    ("Security", r'\b(security|guard|gate\s*pass|id\s*card|theft|stolen|ragging|emergency|sos)\b'),
]


def _meaningful_tokens(text):
    tokens = re.findall(r'[a-z0-9]+', normalize_text(text or ""))
    stop = {"please", "want", "need", "request", "issue", "problem", "lpu", "university", "student"}
    return {t for t in tokens if t not in stop and len(t) > 1}


# ==============================================================================
# 2. CHROMADB VECTOR STORE FOR UNIVERSITY RAG POLICIES & VECTORIZATION
# ==============================================================================

class ChromaUniversityRAGIndex:
    """
    Persistent ChromaDB vector database index for university policies and circulars.
    Guarantees dense vectorization (384-d embeddings) using ONNX all-MiniLM-L6-v2,
    semantic vector cosine search, and seamless synchronization with MySQL.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self.ef = None
        self._seeded = False
        self._init_chroma()

    def _init_chroma(self):
        try:
            self.ef = embedding_functions.DefaultEmbeddingFunction()
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)
            self.collection = self.client.get_or_create_collection(
                name="lpu_university_policies",
                metadata={"hnsw:space": "cosine"},
                embedding_function=self.ef
            )
            print(f"[ChromaDB] Initialized persistent vector store at {CHROMA_DIR} (Items: {self.collection.count()})")
        except Exception as e:
            print(f"[ChromaDB Init Warning] {e}")

    def ensure_seeded_and_vectorized(self, force=False):
        """
        Seeds all DEFAULT_UNIVERSITY_POLICIES into MySQL and ChromaDB,
        and ensures true dense vectorization is saved in MySQL `rag_documents.embedding_json`.
        """
        if self._seeded and not force:
            return

        try:
            # If ChromaDB collection is already populated with all policies, mark seeded
            if not force and self.collection and self.collection.count() >= len(DEFAULT_UNIVERSITY_POLICIES):
                self._seeded = True
                return
        except Exception:
            pass

        # 1. Ensure MySQL has all default policies
        try:
            existing = execute_query(
                "SELECT id, doc_code, title, department, citation, content, embedding_json, source_type FROM rag_documents",
                fetchall=True
            ) or []
            existing_by_code = {r["doc_code"]: r for r in existing}

            for p in DEFAULT_UNIVERSITY_POLICIES:
                row = existing_by_code.get(p["doc_code"])
                if row and row.get("source_type") == "policy":
                    # Update content if changed
                    if row.get("content") != p["content"] or row.get("title") != p["title"]:
                        execute_query(
                            """UPDATE rag_documents
                               SET title = %s, department = %s, citation = %s, content = %s
                               WHERE id = %s""",
                            (p["title"], p["department"], p["citation"], p["content"], row["id"]),
                            commit=True
                        )
                elif not row:
                    execute_query(
                        """INSERT INTO rag_documents (doc_code, title, department, citation, content, source_type)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (p["doc_code"], p["title"], p["department"], p["citation"], p["content"], "policy"),
                        commit=True
                    )
        except Exception as e:
            print(f"[RAG Seed Warning] MySQL seeding error: {e}")

        # 2. Vectorize all MySQL documents into ChromaDB & store dense vector JSON in MySQL
        try:
            all_rows = execute_query(
                "SELECT id, doc_code, title, department, citation, content, embedding_json, source_type FROM rag_documents",
                fetchall=True
            ) or []

            if not all_rows:
                all_rows = [
                    {
                        "id": i + 1,
                        "doc_code": p["doc_code"],
                        "title": p["title"],
                        "department": p["department"],
                        "citation": p["citation"],
                        "content": p["content"],
                        "source_type": "policy",
                        "embedding_json": None
                    }
                    for i, p in enumerate(DEFAULT_UNIVERSITY_POLICIES)
                ]

            if self.collection:
                # Prepare batch vectors
                doc_ids = []
                documents_text = []
                metadatas = []
                update_mysql_needed = []

                for r in all_rows:
                    doc_id = r["doc_code"]
                    text_blob = f"{r['title']} | Department: {r['department']} | Citation: {r['citation']}\n{r['content']}"
                    doc_ids.append(doc_id)
                    documents_text.append(text_blob)
                    metadatas.append({
                        "doc_code": r["doc_code"],
                        "title": r["title"][:200],
                        "department": r["department"][:100],
                        "citation": r["citation"][:200],
                        "source_type": r.get("source_type", "policy"),
                        "sla_hours": int(_POLICY_BY_CODE.get(r["doc_code"], {}).get("sla_hours", 24))
                    })
                    # Check if MySQL needs vectorization update
                    emb_val = r.get("embedding_json") or ""
                    if "dimensions" not in emb_val or "all-MiniLM-L6-v2" not in emb_val:
                        update_mysql_needed.append(r)

                # Upsert into ChromaDB
                self.collection.upsert(
                    ids=doc_ids,
                    documents=documents_text,
                    metadatas=metadatas
                )
                print(f"[ChromaDB] Vectorized & indexed {len(doc_ids)} policy documents into vector store.")

                # If MySQL needs vectorization update, compute dense vectors and persist
                if update_mysql_needed and self.ef:
                    for r in update_mysql_needed:
                        text_blob = f"{r['title']} | Department: {r['department']} | Citation: {r['citation']}\n{r['content']}"
                        vector = self.ef([text_blob])[0]
                        vec_json = json.dumps({
                            "model": "all-MiniLM-L6-v2",
                            "dimensions": len(vector),
                            "embeddings": [round(float(v), 5) for v in vector]
                        })
                        try:
                            execute_query(
                                "UPDATE rag_documents SET embedding_json = %s WHERE doc_code = %s",
                                (vec_json, r["doc_code"]),
                                commit=True
                            )
                        except Exception:
                            pass
                    print(f"[Vectorization] Updated dense vector embeddings in MySQL for {len(update_mysql_needed)} documents.")

            self._seeded = True
        except Exception as e:
            print(f"[ChromaDB Vectorization Warning] {e}")

    def retrieve(self, query_text, department=None, top_k=3):
        """
        Query ChromaDB vector database using dense vector cosine similarity.
        Returns top matching policy documents with semantic similarity scores and metadata.
        """
        self.ensure_seeded_and_vectorized()
        clean_q = (query_text or "").strip()
        if not clean_q or not self.collection:
            return []

        try:
            total_count = self.collection.count()
            if total_count == 0:
                return []

            n_results = min(max(top_k * 2, 6), total_count)
            query_res = self.collection.query(
                query_texts=[clean_q],
                n_results=n_results
            )

            results = []
            ids = query_res.get("ids", [[]])[0]
            docs = query_res.get("documents", [[]])[0]
            metadatas = query_res.get("metadatas", [[]])[0]
            distances = query_res.get("distances", [[]])[0]

            for i in range(len(ids)):
                meta = metadatas[i] if i < len(metadatas) else {}
                dist = float(distances[i]) if i < len(distances) else 0.5
                # ChromaDB cosine distance: similarity = 1 - distance
                raw_sim = max(0.0, min(1.0, round(1.0 - dist, 4)))
                boosted = raw_sim
                if department and meta.get("department") == department:
                    boosted = min(1.0, boosted + 0.15)

                doc_code = meta.get("doc_code", ids[i])
                meta_extra = _POLICY_BY_CODE.get(doc_code, {})

                results.append({
                    "id": i + 1,
                    "doc_code": doc_code,
                    "title": meta.get("title", doc_code),
                    "department": meta.get("department", "General"),
                    "citation": meta.get("citation", "LPU University Regulations"),
                    "content": docs[i] if i < len(docs) else "",
                    "source_type": meta.get("source_type", "policy"),
                    "sla_hours": meta.get("sla_hours") or meta_extra.get("sla_hours", 24),
                    "similarity": raw_sim,
                    "score": round(boosted, 4),
                    "default_advisory": meta_extra.get("default_advisory", ""),
                    "violation_reason": meta_extra.get("violation_reason", ""),
                    "acceptable_scope": meta_extra.get("acceptable_scope", []),
                    "unacceptable_scope": meta_extra.get("unacceptable_scope", [])
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"[ChromaDB Query Error] {e}")
            return []

    def add_document(self, title, department, citation, content):
        """Vectorize a newly uploaded document, insert into ChromaDB and MySQL."""
        doc_code = f"UPLOAD-{uuid.uuid4().hex[:8].upper()}"
        text_blob = f"{title} | Department: {department} | Citation: {citation}\n{content}"

        vec_json = None
        if self.ef:
            vector = self.ef([text_blob])[0]
            vec_json = json.dumps({
                "model": "all-MiniLM-L6-v2",
                "dimensions": len(vector),
                "embeddings": [round(float(v), 5) for v in vector]
            })

        doc_id = execute_query(
            """INSERT INTO rag_documents (doc_code, title, department, citation, content, embedding_json, source_type)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (doc_code, title[:200], department[:100], citation[:200], content[:15000], vec_json, "uploaded"),
            commit=True
        )

        if self.collection:
            self.collection.upsert(
                ids=[doc_code],
                documents=[text_blob],
                metadatas=[{
                    "doc_code": doc_code,
                    "title": title[:200],
                    "department": department[:100],
                    "citation": citation[:200],
                    "source_type": "uploaded",
                    "sla_hours": 24
                }]
            )
            print(f"[ChromaDB] Successfully vectorized and added uploaded document: {doc_code}")
        return doc_id, doc_code

    def delete_document(self, doc_id):
        """Delete an uploaded document from both MySQL and ChromaDB."""
        row = execute_query("SELECT id, doc_code, source_type FROM rag_documents WHERE id = %s", (doc_id,), fetchone=True)
        if not row or row.get("source_type") != "uploaded":
            return False

        doc_code = row["doc_code"]
        execute_query("DELETE FROM rag_documents WHERE id = %s", (doc_id,), commit=True)

        if self.collection and doc_code:
            try:
                self.collection.delete(ids=[doc_code])
                print(f"[ChromaDB] Deleted document {doc_code} from vector store.")
            except Exception as e:
                print(f"[ChromaDB Delete Warning] {e}")
        return True


_chroma_rag = ChromaUniversityRAGIndex()


def list_policy_documents():
    """Return all university policies and uploaded documents stored in MySQL."""
    _chroma_rag.ensure_seeded_and_vectorized()
    return execute_query(
        "SELECT id, doc_code, title, department, citation, content, source_type, created_at FROM rag_documents ORDER BY id ASC",
        fetchall=True
    ) or []


def add_policy_document(title, department, citation, content):
    return _chroma_rag.add_document(title, department, citation, content)


def delete_policy_document(doc_id):
    return _chroma_rag.delete_document(doc_id)


def retrieve_policies(query_text, department=None, top_k=3):
    return _chroma_rag.retrieve(query_text, department=department, top_k=top_k)


# ==============================================================================
# 3. GRIEVANCE POLICY COMPLIANCE & RAG GROUNDING
# ==============================================================================

def evaluate_complaint_policy(description, department=None):
    """
    Evaluates a grievance against official university terms, rules, and ChromaDB documents.
    """
    clean_desc = (description or "").strip()
    if len(clean_desc) < 8:
        return {
            "status": "Conditional",
            "policy_score": 45.0,
            "citation": "LPU Student Grievance Charter, Sec 1",
            "title": "Submission Detail Guidelines",
            "sla_hours": 24,
            "advisory": "Please include specific details (course code, block/location, or date) for faster redressal.",
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

    # 1. Prohibited pattern check
    for pattern, label in PROHIBITED_PATTERNS:
        if re.search(pattern, clean_desc, re.IGNORECASE):
            return {
                "status": "Policy Violation",
                "policy_score": 18.0,
                "citation": best_doc["citation"],
                "title": best_doc["title"],
                "sla_hours": best_doc.get("sla_hours", 24),
                "advisory": best_doc.get("violation_reason") or (
                    f"Flagged under {best_doc['citation']}: {label}. Please review university regulations."
                ),
                "violation_reasons": [label]
            }

    # 2. Token overlap with unacceptable scope
    desc_tokens = _meaningful_tokens(clean_desc)
    unacceptable_hits = []
    for unacc in best_doc.get("unacceptable_scope", []):
        u_tokens = _meaningful_tokens(unacc)
        if u_tokens and len(u_tokens & desc_tokens) >= 2:
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

    sim = float(best_doc.get("similarity", 0.0))
    score = min(98.0, max(70.0, round(65.0 + (sim * 40.0), 1)))
    advisory = best_doc.get("default_advisory") or (
        f"Grievance evaluated under {best_doc['citation']} (SLA: {best_doc.get('sla_hours', 24)}h). Forwarded to department queue."
    )

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
# 4. ADVANCED AI STUDENT ASSISTANT CHATBOT (GEMINI + CHROMADB RAG)
# ==============================================================================

_GREETING_RE = re.compile(
    r'^\s*(hi+|hello+|hey+|hlo+|yo|sup|good\s+(morning|afternoon|evening|night)|namaste|how\s+are\s+you|how\s+r\s+u|'
    r'what\'?s\s+up|thanks?|thank\s+you|bye|goodbye|who\s+are\s+you|help|what\s+can\s+you\s+do)\b',
    re.IGNORECASE
)
_STATUS_RE = re.compile(
    r'\b(my\s+ticket|my\s+complaint|my\s+grievance|ticket\s*#?\s*\d+|complaint\s*#?\s*\d+|check\s+status|track\s+my|how\s+many\s+complaints)\b',
    re.IGNORECASE
)
_COMPLAINT_SIGNAL_RE = re.compile(
    r'\b(not\s+working|broken|leaking|leak|sparks|fire|stolen|theft|charged\s+twice|deducted\s+twice|'
    r'stale|cockroach|hair\s+in|missing\s+marks|exam\s+clash|rude\s+staff|rude\s+faculty|ragging|harass|disconnected|slow\s+wifi|'
    r'500\s+error|internal\s+error|overdue\s+fine|no\s+water|power\s+cut|lift\s+stuck|file\s+a\s+complaint|'
    r'water\s+cooler|washroom|flush\s+not|ac\s+not|fan\s+not|short\s+circuit|is\s+down|are\s+down|not\s+connecting)\b',
    re.IGNORECASE
)


def _build_rag_context_for_gemini(user_message, local_result, student_id=None, student_name=None):
    """Assemble grounded ChromaDB vector documents, ML triage, and live MySQL tickets for Gemini."""
    sections = []
    intent = local_result.get("intent", "general_chat")

    if student_name:
        sections.append(f"Student Name: {student_name}")

    sections.append(f"Query Intent: {intent}")

    # 1. Retrieved policies from ChromaDB vector database
    docs = retrieve_policies(user_message, top_k=3)
    if docs:
        pol_lines = [
            f"- [{d['department']}] {d['title']} ({d['citation']}): {d['content'][:500]} (SLA: {d.get('sla_hours', 24)}h)"
            for d in docs
        ]
        sections.append("Retrieved Official University Policies (ChromaDB Vector Store):\n" + "\n".join(pol_lines))

    # 2. AI Triage Analysis (if grievance)
    analysis = local_result.get("analysis")
    if analysis:
        alt_str = ""
        if analysis.get("alternatives"):
            alt_str = " | Alternatives: " + ", ".join(f"{a['department']} ({a['confidence']}%)" for a in analysis["alternatives"])
        sections.append(
            "AI Grievance Classification:\n"
            f"- Recommended Department: {analysis['department']} ({analysis['confidence']}% confidence{alt_str})\n"
            f"- Urgency Score: {analysis['urgency_score']}/100 ({analysis['priority']})\n"
            f"- Policy Status: {analysis.get('policy_status', 'Acceptable')} (Citation: {analysis.get('citation', '')})"
        )

    # 3. Live MySQL Tickets for this student
    if student_id:
        try:
            rows = execute_query(
                """SELECT c.id, c.predicted_department, c.priority_level, c.urgency_score, c.status,
                          c.completion_note, c.created_at, f.name AS faculty_name
                   FROM complaints c
                   LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
                   WHERE c.student_id = %s
                   ORDER BY c.id DESC LIMIT 5""",
                (student_id,),
                fetchall=True
            ) or []
            if rows:
                t_lines = []
                for r in rows:
                    officer = r.get("faculty_name") or f"{r['predicted_department']} Department"
                    note = f" | Resolution: {r['completion_note'][:100]}" if r.get("completion_note") else ""
                    t_lines.append(
                        f"- Ticket #{r['id']}: Dept={r['predicted_department']}, Status={r['status']}, "
                        f"Priority={r['priority_level']} ({r.get('urgency_score', 25)}/100), Officer={officer}{note}"
                    )
                sections.append("Student's Active / Recent Tickets in MySQL Database:\n" + "\n".join(t_lines))
            else:
                sections.append("Student's Active Tickets: None currently registered.")
        except Exception:
            pass

    return "\n\n".join(sections)


def _call_gemini_rag(user_message, context_block, intent="general_chat"):
    """
    Call Gemini using official google-genai SDK with multi-model fallback:
    gemini-3.5-flash -> gemini-3.5-flash-lite -> gemini-3.8-flash -> gemini-3.1-flash-lite.
    """
    client = _get_gemini_client()
    if client is None:
        return None, None

    candidate_models = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-flash-lite"]

    system_instruction = (
        "You are the official, highly advanced LPU AI Student Assistant & Academic Advisor, developed by the "
        "Department of Computer Science and Engineering (AI & ML) for Lovely Professional University. "
        "You have direct access to official university regulations via the ChromaDB vector database and live student "
        "grievance records in MySQL.\n\n"
        "Instructions:\n"
        "1. For greetings, casual chat, and general/out-of-box questions ('hey', 'how are you', study advice, coding help, jokes), "
        "respond warmly, intelligently, and engagingly as a helpful campus AI companion.\n"
        "2. For university rules, policies, attendance, examinations, fee refunds, hostel regulations, and faculty conduct, "
        "provide clear, authoritative, and structured answers grounded strictly in the ChromaDB context. Always cite the exact "
        "document code and official policy citation.\n"
        "3. When student asks about their tickets, report their live ticket details accurately from the context.\n"
        "4. When student describes an issue or grievance, summarize the predicted department, urgency score (0-100), "
        "resolution SLA, and encourage them to click 'Fill in Grievance Form' below.\n"
        "5. Format your output with clear Markdown: bold headings, bullet points, and practical action steps."
    )

    prompt = f"[CAMPUS CONTEXT & CHROMADB VECTOR RETRIEVAL]\n{context_block}\n\n[STUDENT INQUIRY]\n{user_message}"

    for model_name in candidate_models:
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                max_output_tokens=750
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            if response and response.text:
                return response.text.strip(), f"gemini-{model_name}"
        except Exception as e:
            err_msg = str(e)
            print(f"[Gemini Model Warning] '{model_name}' failed: {err_msg[:90]}")
            continue

    return None, None


def _generate_local_chatbot_response(user_message, student_id=None, student_name=None):
    """
    Fast, offline deterministic response using ChromaDB vector retrieval, ML classification,
    and MySQL ticket database.
    """
    msg = (user_message or "").strip()
    if not msg:
        return {
            "reply": "Please ask a question regarding university policies, fees, exams, attendance, or describe a campus problem.",
            "intent": "empty",
            "sources": [],
            "analysis": None,
            "suggested_grievance": None
        }

    # 1. Greeting
    if _GREETING_RE.search(msg) and len(msg.split()) <= 6:
        name_str = f" {student_name.split()[0]}" if student_name else ""
        return {
            "reply": (
                f"Hello{name_str}! 👋 I am your **LPU AI Student Assistant**, powered by ChromaDB vector intelligence and Gemini AI.\n\n"
                "How can I assist you today?\n"
                "• 📚 **University Policies:** 75% attendance rule, fees/scholarships, exam date sheet clashes, hostel bylaws\n"
                "• 🎫 **Live Tickets:** Ask *'check my ticket status'* to view real-time resolution progress\n"
                "• ⚡ **Campus Issues:** Describe any problem for instant department routing and urgency scoring (0-100)"
            ),
            "intent": "greeting",
            "sources": [],
            "analysis": None,
            "suggested_grievance": None
        }

    # 2. Live Ticket Status
    if _STATUS_RE.search(msg):
        if not student_id:
            return {
                "reply": "Please sign in with your University Registration Number so I can look up your active grievance tickets.",
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
                    "reply": f"Ticket #{tid} was not found under your student account. Type *'my tickets'* to see all your complaints.",
                    "intent": "status",
                    "sources": [],
                    "analysis": None,
                    "suggested_grievance": None
                }
            officer = row.get("faculty_name") or f"{row['predicted_department']} Desk"
            note = f"\n• **Resolution Note:** {row['completion_note']}" if row.get("completion_note") else ""
            return {
                "reply": (
                    f"**Grievance Ticket #{row['id']} Status**\n"
                    f"• **Department:** {row['predicted_department']}\n"
                    f"• **Current Status:** {row['status']}\n"
                    f"• **Priority & Urgency:** {row['priority_level']} ({row.get('urgency_score', 25)}/100)\n"
                    f"• **Assigned Officer:** {officer}\n"
                    f"• **Submitted:** {row['created_at']}{note}"
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
                "reply": "You have not registered any grievance tickets yet. Describe an issue here to submit one!",
                "intent": "status",
                "sources": [],
                "analysis": None,
                "suggested_grievance": None
            }

        lines = [f"Found **{len(rows)}** recent grievance ticket(s) under your account:"]
        for r in rows:
            officer = r.get("faculty_name") or r["predicted_department"]
            lines.append(f"• **#{r['id']}** [{r['predicted_department']}] — *{r['status']}* | Priority: {r['priority_level']} | Officer: {officer}")
        lines.append("\nType *'status of ticket #<id>'* for detailed case remarks.")
        return {
            "reply": "\n".join(lines),
            "intent": "status",
            "sources": [],
            "analysis": None,
            "suggested_grievance": None
        }

    # 3. Complaint Triage
    is_complaint = bool(_COMPLAINT_SIGNAL_RE.search(msg)) and not msg.endswith("?")
    if is_complaint:
        pred = predict_department_with_confidence(msg)
        urg_score, priority, urg_reasons = calculate_urgency(msg)
        policy_eval = evaluate_complaint_policy(msg, department=pred["department"])

        alt_text = ""
        if pred["alternatives"]:
            alt_text = " | Alternatives: " + ", ".join(f"{a['department']} ({a['confidence']}%)" for a in pred["alternatives"])

        if policy_eval["status"] == "Policy Violation":
            reply = (
                f"⚠️ **University Policy Alert ({policy_eval['citation']}):**\n"
                f"{policy_eval['advisory']}\n\n"
                f"• **Predicted Department:** {pred['department']}\n"
                f"• **Policy Verdict:** Flagged Violation"
            )
            suggested = None
        else:
            reply = (
                f"I analyzed your issue using **ChromaDB Vector Intelligence & AI Triage**:\n\n"
                f"• **Recommended Department:** **{pred['department']}** ({pred['confidence']}% confidence{alt_text})\n"
                f"• **Urgency Assessment:** **{priority}** (Urgency Score: {urg_score}/100)\n"
                f"• **Governing Policy:** {policy_eval['citation']} (Target Resolution SLA: **{policy_eval['sla_hours']} hours**)\n"
                f"• **Advisory:** {policy_eval['advisory']}\n\n"
                f"💡 Click **'Fill in Grievance Form'** below to register this complaint immediately with priority dispatch."
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

    # 4. ChromaDB Semantic Vector Policy Query
    docs = retrieve_policies(msg, top_k=2)
    if docs and docs[0]["similarity"] >= 0.25:
        primary = docs[0]
        sources = [{"title": d["title"], "citation": d["citation"]} for d in docs if d["similarity"] >= 0.25]
        reply = (
            f"**{primary['title']}**\n"
            f"*Citation: {primary['citation']}* | Standard SLA: **{primary['sla_hours']} hours**\n\n"
            f"{primary['content']}\n\n"
            f"Need to register an official ticket for this? Describe your issue or click the **Log Request** tab."
        )
        return {
            "reply": reply,
            "intent": "policy_rag",
            "sources": sources,
            "analysis": None,
            "suggested_grievance": None
        }

    # 5. Default General Response
    name_str = f" {student_name.split()[0]}" if student_name else ""
    return {
        "reply": (
            f"Hello{name_str}! I'm here to assist you with any campus query, academic policy, examination rules, "
            f"attendance guidelines, fee queries, or grievance resolution.\n\n"
            f"Feel free to ask a specific question or describe an issue you are experiencing."
        ),
        "intent": "general_chat",
        "sources": [],
        "analysis": None,
        "suggested_grievance": None
    }


def generate_chatbot_response(user_message, student_id=None, student_name=None, previous_interaction_id=None):
    """
    Hybrid ChromaDB RAG + Google Gemini AI Student Assistant.
    Runs dense vector retrieval via ChromaDB, parses live MySQL state,
    and queries Gemini API with structured institutional context.
    """
    msg = (user_message or "").strip()
    local_result = _generate_local_chatbot_response(
        msg,
        student_id=student_id,
        student_name=student_name
    )

    if not msg:
        local_result["powered_by"] = "local-rag"
        return local_result

    # If Gemini is configured, synthesize with Gemini AI
    if is_gemini_configured():
        context_block = _build_rag_context_for_gemini(
            msg,
            local_result,
            student_id=student_id,
            student_name=student_name
        )
        gemini_text, model_tag = _call_gemini_rag(
            msg,
            context_block,
            intent=local_result.get("intent", "general_chat")
        )
        if gemini_text:
            local_result["reply"] = gemini_text
            local_result["powered_by"] = "gemini-chromadb-rag"
            local_result["model"] = model_tag
            return local_result

    local_result["powered_by"] = "chromadb-local-rag"
    return local_result
