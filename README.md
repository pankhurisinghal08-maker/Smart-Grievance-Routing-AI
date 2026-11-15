# Student Grievance System (Lovely Professional University)
### AI-Powered RAG Campus Redressal, Policy Compliance & Intelligent Routing Platform

An enterprise campus grievance redressal and AI routing platform augmented with a **Retrieval-Augmented Generation (RAG)** policy guardrail engine. The system evaluates student grievances in real-time against university bylaws and standard operating terms, clusters overlapping complaints across students for **One-Click Group Redressal**, predicts departments using NLP text classification, balances faculty workloads, and enforces sentiment-verified case closure.

---

## 🚀 Advanced RAG & 20+ Key Features

### 1. Core RAG Policy & Acceptability Guardrail
- **University Terms & Policy Verification**: Evaluates incoming student grievances against official campus bylaws, student code of conduct, hostel rules, exam ordinances, and IT acceptable use policies.
- **Acceptability Scoring & Compliance Verdict**: Computes an Acceptability Score (0-100%) and categorizes grievances into `Acceptable`, `Policy Violation / Rejected`, or `Conditional Review (Evidence Required)`.
- **Statutory Bylaw Citations**: Automatically cites specific policy regulations (e.g., `LPU IT Code 2026, Sec 4.1`, `Estate Maintenance Bylaws Art. 8`, `FSSAI Section 3.4`).
- **Automated Policy Rejection & Advisory**: If non-compliant (e.g., firewall proxy circumvention, exam paper disclosures, luxury personal modification demands), the AI guardrail auto-rejects the ticket and issues formal policy guidance and allowed recourse.

### 2. Multi-Student Semantic Deduplication & Group Redressal (RAG Clustering)
- **Vector-Based Semantic Similarity Clustering**: Detects overlapping complaints across all students using TF-IDF + Cosine Similarity ($\ge 65\%$ threshold), grouping them under shared cluster IDs (e.g., `CLUST-IT-1`, `CLUST-MAIN-2`).
- **One-Click Mass Redressal for Faculty & Admin**: When multiple students report the same issue (e.g., Wi-Fi outage in Block 34, mess water cooler breakdown), faculty can update status and broadcast a unified resolution message to all affected students in a single click!
- **Incident Outbreak Detection Banner**: High-density clusters automatically trigger real-time alert banners on the Faculty and Admin dashboards to highlight campus infrastructure emergencies.

### 3. Faculty AI Resolution Co-Pilot
- **RAG Resolution Synthesis**: With one click (`Draft with RAG`), faculty can generate empathetic, technically accurate resolution drafts grounded in university standard operating procedures (SOPs).

### 4. Real-Time Interactive Live Pre-Check Assistant
- **As-You-Type AJAX Policy Validator**: As students type their grievance, an interactive assistant analyzes compliance, estimates resolution SLA (hours), checks for active campus clusters, and computes a Hazard Risk Score before submission.

### 5. Campus Safety, Urgency & SLA Intelligence
- **Multi-Factor Hazard Risk Scoring (0-100)**: Evaluates physical safety hazards, electrical shocks, fire risks, medical emergencies, and academic deadlines.
- **Dynamic SLA Deadline & Breach Risk Predictor**: Calculates dynamic resolution targets (e.g., 6h for high-priority hazards, 18h for IT network, 24h for maintenance) and alerts officers when approaching breach risk.

### 6. Student Self-Service & Accessibility
- **Searchable University Terms & Policy Explorer**: Interactive modal in Student View to search through all university policies, SLAs, and acceptable grievance scopes.
- **Voice-to-Text Speech Dictation**: Hands-free dictation using the Web Speech API (`SpeechRecognition`).
- **Campus Active Outage Bulletin System**: Live alert banner informing students about ongoing campus maintenance (e.g., scheduled Wi-Fi upgrades).
- **Formal Case Escalation Workflow**: Unsatisfied students can formally escalate unresolved tickets to High Priority with direct notification to the Central Administration Directorate.

### 7. Governance, Auditing & Export
- **Executive RAG Redressal KPI Dashboard**: Tracks Policy Compliance Rate (%), Blocked Policy Violations, Active Outbreak Clusters, and Average Hazard Indices.
- **One-Click CSV Grievance Dossier Export**: Comprehensive report download with compliance verdicts, cluster IDs, SLA breach risks, sentiment, and faculty workloads.
- **Dark / Light Theme Toggle**: Persistent university-styled dark mode across all views.
- **Sentiment-Verified Case Closure**: Final administrative closure is permitted only after positive student satisfaction feedback is verified by the integrated NLP sentiment classifier.

---

## 🏛️ System Architecture & RAG Workflow

```
[Student Grievance Input] ──> [Live AJAX As-You-Type Policy Pre-Check]
          │
          ▼
[Submission: RAG Policy Guardrail Engine]
          │
     ┌────┴─────────────────────────────┐
     ▼                                  ▼
[Policy Violation Detected]    [Acceptable / Compliant]
     │                                  │
     ▼                                  ▼
[Status: Rejected]             [Semantic Clustering & Deduplication Engine]
[Auto Policy Advisory Posted]           │
                                        ├─> [Cluster with Similar Active Student Tickets]
                                        │
                                        ▼
                               [AI NLP Classifier & Urgency Hazard Scoring]
                                        │
                                        ▼
                               [Dynamic SLA Calculator (6h - 48h)]
                                        │
                                        ▼
                               [Lowest Active Workload Assignment] ──> [Assigned Faculty Desk]
                                                                                │
                                                   ┌────────────────────────────┴────────────────────────────┐
                                                   ▼                                                         ▼
                                      [Single-Ticket Resolution]                              [One-Click Group Redressal]
                                      (RAG AI Draft Assistant)                                (Broadcast to All Clustered Students)
                                                   │                                                         │
                                                   └────────────────────────────┬────────────────────────────┘
                                                                                ▼
                                                                  [Completed - Awaiting Feedback]
                                                                                │
                                                                                ▼
                                                                    [Student Submits Feedback]
                                                                                │
                                                                                ▼
                                                                    [NLP Sentiment Classifier]
                                                                                │
                                                        ┌───────────────────────┴───────────────────────┐
                                                        ▼                                               ▼
                                              [Positive Feedback]                             [Negative Feedback]
                                                        │                                               │
                                                        ▼                                               ▼
                                         [Admin Verified Closure Allowed]                 [Admin Closure Blocked / Rerouted]
```

---

## 🛠️ Quick Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
```env
DB_TYPE=auto
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=Password
MYSQL_DATABASE=grievance_system

SECRET_KEY=long_random_secret_key
FLASK_DEBUG=False
PORT=5000
ADMIN_USERNAME=Pankhuri
ADMIN_PASSWORD=Pankhuri1007
```
*(Note: If MySQL is not running locally, the system automatically and gracefully falls back to local SQLite `database/grievance_system.db` without crashing).*

### 3. Run Automated Tests
```bash
python test_rag_pipeline.py
```

### 4. Start the Application
```bash
python app.py
```
Open `http://127.0.0.1:5000/login` in your browser.

---

## 🔐 Portal Access Roles

- **Unified Portal Gateway**: `http://127.0.0.1:5000/login`
  - **Student View**: Log in with Registration Number (e.g. `DEMO001` or any student registration number).
  - **Assigned Teachers Workspace**: Select faculty officer profile (e.g. `Parth Singhal - IT Desk`).
  - **Admin View**: Sign in with administrator credentials (`Pankhuri` / `Pankhuri1007`).
