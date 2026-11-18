import os
import re
import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

MYSQL_HOST = os.getenv('MYSQL_HOST', '127.0.0.1').strip()
if MYSQL_HOST.lower() == 'localhost':
    MYSQL_HOST = '127.0.0.1'
MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'grievance_system').strip()

_mysql_initialized = False
_mysql_pool = None


def _validate_db_name(name):
    if not re.match(r'^[A-Za-z0-9_]+$', name):
        raise ValueError(f"Invalid MySQL database name: {name!r}")
    return name


def get_mysql_config(include_database=True):
    config = {
        'host': MYSQL_HOST,
        'port': MYSQL_PORT,
        'user': MYSQL_USER,
        'password': MYSQL_PASSWORD,
        'connection_timeout': 10
    }
    if include_database:
        config['database'] = MYSQL_DATABASE
    return config


def _schema(cursor):
    cursor.execute("""CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        roll_number VARCHAR(100) NOT NULL UNIQUE,
        name VARCHAR(100) NOT NULL,
        password VARCHAR(100) NOT NULL,
        role VARCHAR(50) DEFAULT 'student'
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS faculty (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        department VARCHAR(100) NOT NULL,
        email VARCHAR(100) NOT NULL
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS complaints (
        id INT AUTO_INCREMENT PRIMARY KEY,
        student_id INT,
        description TEXT NOT NULL,
        predicted_department VARCHAR(100),
        assigned_faculty_id INT,
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS complaint_comments (
        id INT AUTO_INCREMENT PRIMARY KEY,
        complaint_id INT NOT NULL,
        author_name VARCHAR(100) NOT NULL,
        author_role VARCHAR(100) NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS notifications (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        faculty_id INT,
        complaint_id INT,
        message TEXT NOT NULL,
        is_read INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS rag_documents (
        id INT AUTO_INCREMENT PRIMARY KEY,
        doc_code VARCHAR(64) NOT NULL UNIQUE,
        title VARCHAR(200) NOT NULL,
        department VARCHAR(100) NOT NULL,
        citation VARCHAR(200) NOT NULL,
        content TEXT NOT NULL,
        embedding_json LONGTEXT NULL,
        source_type VARCHAR(30) DEFAULT 'policy',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cursor.execute("""CREATE TABLE IF NOT EXISTS access_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        actor_role VARCHAR(30) NOT NULL,
        actor_id VARCHAR(100) NULL,
        actor_name VARCHAR(100) NULL,
        action VARCHAR(120) NOT NULL,
        details TEXT NULL,
        ip_address VARCHAR(64) NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Self-healing migration for existing databases missing newer columns
    migrations = [
        ('assigned_faculty_id', 'INT'),
        ('urgency_score', 'INT DEFAULT 25'),
        ('confidence_score', 'FLOAT NULL'),
        ('alternative_departments', 'VARCHAR(255) NULL'),
        ('completion_note', 'TEXT'),
        ('student_feedback', 'TEXT'),
        ('feedback_sentiment', 'VARCHAR(20)'),
        ('rag_policy_status', "VARCHAR(50) DEFAULT 'Acceptable'"),
        ('rag_policy_citation', 'VARCHAR(255) NULL'),
        ('rag_policy_advisory', 'TEXT NULL'),
        ('escalation_level', 'INT DEFAULT 0'),
        ('closed_at', 'TIMESTAMP NULL')
    ]
    cursor.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'complaints'
    """, (MYSQL_DATABASE,))
    existing_cols = {row[0] for row in cursor.fetchall()}

    for column, definition in migrations:
        if column not in existing_cols:
            cursor.execute(f"ALTER TABLE complaints ADD COLUMN {column} {definition}")


def init_mysql():
    global _mysql_initialized
    if _mysql_initialized:
        return

    db_name = _validate_db_name(MYSQL_DATABASE)

    server = None
    try:
        server = mysql.connector.connect(**get_mysql_config(include_database=False))
        cursor = server.cursor()
        try:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            server.commit()
        finally:
            cursor.close()
    finally:
        if server:
            server.close()

    conn = None
    try:
        conn = mysql.connector.connect(**get_mysql_config(include_database=True))
        cursor = conn.cursor()
        try:
            _schema(cursor)
            conn.commit()
        finally:
            cursor.close()
        _mysql_initialized = True
        print(f"[Database] MySQL connected: {db_name} ({MYSQL_HOST}:{MYSQL_PORT})")
    finally:
        if conn:
            conn.close()


def get_mysql_pool():
    global _mysql_pool
    if _mysql_pool is not None:
        return _mysql_pool

    init_mysql()
    _mysql_pool = pooling.MySQLConnectionPool(
        pool_name="grievance_pool",
        pool_size=10,
        pool_reset_session=True,
        **get_mysql_config(include_database=True)
    )
    return _mysql_pool


def get_backend_name():
    init_mysql()
    return 'mysql'


def get_connection():
    try:
        pool = get_mysql_pool()
        conn = pool.get_connection()
        conn.ping(reconnect=True, attempts=3, delay=1)
        return conn
    except Exception:
        init_mysql()
        return mysql.connector.connect(**get_mysql_config(include_database=True))


def execute_query(sql, params=(), fetchone=False, fetchall=False, commit=False):
    conn = get_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True, buffered=True)
        cursor.execute(sql, params)
        if commit:
            conn.commit()
            return cursor.lastrowid
        if fetchone:
            return cursor.fetchone()
        if fetchall:
            return cursor.fetchall()
        return None
    except Exception:
        if commit and conn:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if cursor:
            cursor.close()
        conn.close()
