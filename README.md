# Smart Grievance Routing AI

A campus Student Grievance System and AI routing platform that classifies grievances using an NLP TF-IDF / Naive Bayes model, assesses priority levels based on urgency rules, and intelligently routes each case to assigned faculty and departmental officers.

## Features

- **Automated AI Department Prediction**: Natural language processing model trained on campus grievance datasets to classify tickets across departments (IT, Maintenance, Cafeteria, Security, Finance, Library, Discipline, Examination).
- **Rule-based Urgent Triage**: Emergency pattern matching to flag high-priority complaints instantly.
- **Dynamic Load Balancing**: Routes incoming complaints to the faculty member with the lowest active workload within that department.
- **Two-way Correspondence**: Official discussion threads between students, faculty officers, and administration.
- **Resolution & Sign-off**: Faculty marks grievances resolved with official remarks; students review and provide satisfaction feedback to verify and close tickets.
- **MySQL Integration**: Persistent storage in MySQL with real student accounts and faculty profiles.

---

## Quick Setup

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Configure Database (.env)
Configure `.env` with your MySQL connection details:
```env
DB_TYPE=mysql
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=Password
MYSQL_DATABASE=grievance_system

FLASK_DEBUG=False
PORT=5000
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

### 3. Train AI NLP Model (Optional / Auto-trained on startup)
```bash
python train.py
```

### 4. Run the Application
```bash
python app.py
```

---

## Access Portals

- **Student Portal**: [http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)
  - Log in with any registered roll number added in your MySQL database.
- **Assigned Personnel & Faculty Desk**: [http://127.0.0.1:5000/assigned](http://127.0.0.1:5000/assigned)
  - Master combined dashboard for all assigned faculty and departmental officers.
  - Switch officers or sign in via [http://127.0.0.1:5000/faculty/login](http://127.0.0.1:5000/faculty/login).
- **Central Administrative Console**: [http://127.0.0.1:5000/admin/login](http://127.0.0.1:5000/admin/login)
  - Default credentials: `admin` / `admin123`.
