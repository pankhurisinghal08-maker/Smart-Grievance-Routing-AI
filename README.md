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
  - **Assigned Faculty Officers**: Work only on their assigned grievances, move work from Pending to In Progress, record a resolution summary that requests student feedback, and send private internal notes to Administration. They cannot reassign cases or message students directly.
  - **Central Administration**: View campus-wide queues and workload reports, assign or reassign officers, send student-facing updates, review negative feedback, and exclusively close completed grievances after positive student feedback.
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
MYSQL_HOST=your-mysql-host
MYSQL_PORT=3306
MYSQL_USER=your-mysql-user
MYSQL_PASSWORD=your-private-password
MYSQL_DATABASE=grievance_system

SECRET_KEY=replace_with_a_long_random_value
FLASK_DEBUG=False
PORT=5000
ADMIN_USERNAME=your-admin-username
ADMIN_PASSWORD=your-strong-admin-password
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
  - **Admin View**: Sign in with the administrator credentials configured in `.env`.

---

## Deploy the Department Classifier API for Free

The standalone classifier API does not connect to MySQL. Create a free Render Web Service from this repository's `codex/deploy-department-model` branch and choose `render.yaml` as its Blueprint file. Render installs the dependencies, trains the classifier from `dataset/complaints_train.csv`, and serves the prediction API. The free service can sleep while idle, so its first request after a quiet period may take longer.

- `GET /health` checks that the model is ready.
- `POST /predict` accepts JSON such as `{"description":"The lab Wi-Fi is not working"}` and returns a predicted department.

For the full grievance portal, configure a cloud-accessible MySQL database separately. A MySQL server running only on your computer is not reachable by a cloud service; do not expose your PC's MySQL port to the public internet.
