# Student Grievance System (Lovely Professional University)

An intelligent, enterprise campus grievance redressal and AI routing platform. The system classifies grievances using NLP text models, evaluates urgency levels through rule-based triage, load-balances assignments across departmental faculty officers, and enforces verified resolution workflows through natural language student feedback sentiment analysis.

---

## Key Features

- **Automated AI Department Classification**: NLP TF-IDF and Multinomial Naive Bayes model trained on campus grievance logs to predict and route tickets to the correct department (`IT`, `Maintenance`, `Cafeteria`, `Security`, `Finance`, `Library`, `Discipline`, `Examination`).
- **Urgent Triage & Priority Evaluation**: Intelligent regex and keyword heuristics to immediately elevate critical issues (e.g. electrical fire, water leaks, medical emergencies, server outages) to `High Priority`.
- **Dynamic Faculty Workload Balancing**: Intelligently routes incoming grievances to the departmental officer with the lowest active ticket count.
- **Unified Single-Gateway Authentication**: A single portal entry point (`/login`) with role tabs for **Student View**, **Assigned Task Desk**, and **Admin View**.
- **Role-Based Access Control (RBAC)**:
  - **Students**: Lodge grievances, monitor status updates, converse in case discussion threads, and submit resolution satisfaction feedback.
  - **Faculty Officers**: Access dedicated departmental workspaces, post official updates to students, reassign tickets to peer officers, and mark jobs completed with resolution remarks. Faculty are restricted from unilaterally closing tickets.
  - **Central Administration**: Complete oversight of campus operations, collapsible faculty roster and workload statistics, re-routing controls, and exclusive authority to close tickets.
- **Sentiment-Verified Grievance Closure**:
  - A grievance cannot be closed until the student submits satisfaction feedback.
  - An integrated NLP sentiment classifier (`Positive` vs `Negative`) verifies the student's remarks.
  - Only when positive feedback is verified can Central Administration permanently close the grievance. Negative feedback blocks closure and alerts administration to take remedial action.
- **Clean MySQL Integration**: Direct connection with MySQL relational database storing university student records (with registration numbers) and faculty personnel profiles.

---

## System Architecture & Workflow

```
[Student Grievance Submission]
             │
             ▼
[AI NLP Department Classifier] + [Urgency Triage Evaluator]
             │
             ▼
[Lowest Active Workload Assignment] ──> [Assigned Faculty Desk]
                                                 │
                                                 ▼
                                     [Faculty Executes Resolution]
                                                 │
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

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Database Environment (`.env`)
Configure your MySQL database credentials in `.env`:
```env
DB_TYPE=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=Password
MYSQL_DATABASE=grievance_system

FLASK_DEBUG=False
PORT=5000
ADMIN_USERNAME=Pankhuri
ADMIN_PASSWORD=Pankhuri1007
```

### 3. Initialize & Train AI Models (Auto-trained on startup)
```bash
python train.py
```

### 4. Run the Application
```bash
python app.py
```

---

## Portal Access & Roles

- **Unified Portal Gateway**: `http://127.0.0.1:5000/login`
  - **Student View**: Log in with your University Registration Number (e.g. `12509701`).
  - **Assigned Desk**: Select your assigned faculty officer profile (e.g. `Parth Patel - IT Desk`).
  - **Admin View**: Sign in with administrator credentials:
    - **Username**: `Pankhuri`
    - **Password**: `Pankhuri1007`
