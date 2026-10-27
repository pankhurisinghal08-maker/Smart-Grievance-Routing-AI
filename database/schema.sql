CREATE DATABASE IF NOT EXISTS grievance_system;
USE grievance_system;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    roll_number VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    password VARCHAR(100) NOT NULL,
    role VARCHAR(20) DEFAULT 'student'
);

CREATE TABLE faculty (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL
);

CREATE TABLE complaints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    description TEXT NOT NULL,
    predicted_department VARCHAR(100) NOT NULL,
    assigned_faculty_id INT,
    priority_level VARCHAR(20) DEFAULT 'Normal',
    status VARCHAR(20) DEFAULT 'Pending',
    completion_note TEXT NULL,
    student_feedback TEXT NULL,
    closed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(id),
    FOREIGN KEY (assigned_faculty_id) REFERENCES faculty(id)
);
