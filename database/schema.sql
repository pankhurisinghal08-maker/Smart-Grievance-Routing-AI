CREATE DATABASE IF NOT EXISTS grievance_system;
USE grievance_system;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    roll_number VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    password VARCHAR(100) NOT NULL,
    role VARCHAR(50) DEFAULT 'student'
);

CREATE TABLE IF NOT EXISTS faculty (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS complaints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    description TEXT NOT NULL,
    predicted_department VARCHAR(100) NOT NULL,
    assigned_faculty_id INT NULL,
    priority_level VARCHAR(50) DEFAULT 'Low Priority',
    urgency_score INT DEFAULT 25,
    confidence_score FLOAT NULL,
    alternative_departments VARCHAR(255) NULL,
    status VARCHAR(100) DEFAULT 'Pending',
    completion_note TEXT NULL,
    student_feedback TEXT NULL,
    feedback_sentiment VARCHAR(20) NULL,
    rag_policy_status VARCHAR(50) DEFAULT 'Acceptable',
    rag_policy_citation VARCHAR(255) NULL,
    rag_policy_advisory TEXT NULL,
    escalation_level INT DEFAULT 0,
    closed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(id),
    FOREIGN KEY (assigned_faculty_id) REFERENCES faculty(id)
);

CREATE TABLE IF NOT EXISTS complaint_comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id INT NOT NULL,
    author_name VARCHAR(100) NOT NULL,
    author_role VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(id)
);

CREATE TABLE IF NOT EXISTS notifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    faculty_id INT NULL,
    complaint_id INT NOT NULL,
    message TEXT NOT NULL,
    is_read INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (complaint_id) REFERENCES complaints(id)
);

CREATE TABLE IF NOT EXISTS rag_documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doc_code VARCHAR(64) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    department VARCHAR(100) NOT NULL,
    citation VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    embedding_json LONGTEXT NULL,
    source_type VARCHAR(30) DEFAULT 'policy',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS access_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    actor_role VARCHAR(30) NOT NULL,
    actor_id VARCHAR(100) NULL,
    actor_name VARCHAR(100) NULL,
    action VARCHAR(120) NOT NULL,
    details TEXT NULL,
    ip_address VARCHAR(64) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
