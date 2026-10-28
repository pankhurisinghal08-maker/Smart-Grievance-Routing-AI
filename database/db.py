import os
import re
import sqlite3
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

MYSQL_HOST = os.getenv('MYSQL_HOST', '127.0.0.1')
if MYSQL_HOST == 'localhost':
    MYSQL_HOST = '127.0.0.1'
MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'grievance_system')
PREFERRED_DB = os.getenv('DB_TYPE', 'auto').lower()
SQLITE_DB_PATH = os.path.join(BASE_DIR, 'database', 'grievance_system.db')
_active_backend = None


def _schema(cursor, mysql=False):
    auto = 'INT AUTO_INCREMENT PRIMARY KEY' if mysql else 'INTEGER PRIMARY KEY AUTOINCREMENT'
    text = 'VARCHAR(100)' if mysql else 'TEXT'
    
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS users (
        id {auto},
        roll_number {text} NOT NULL UNIQUE,
        name {text} NOT NULL,
        password {text} NOT NULL,
        role {text} DEFAULT 'student'
    )""")
    
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS faculty (
        id {auto},
        name {text} NOT NULL,
        department {text} NOT NULL,
        email {text} NOT NULL
    )""")
    
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS complaints (
        id {auto},
        student_id INTEGER,
        description TEXT NOT NULL,
        predicted_department {text},
        assigned_faculty_id INTEGER,
        priority_level {text} DEFAULT 'Normal',
        status {text} DEFAULT 'Pending',
        completion_note TEXT NULL,
        student_feedback TEXT NULL,
        closed_at TIMESTAMP NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS complaint_comments (
        id {auto},
        complaint_id INTEGER NOT NULL,
        author_name {text} NOT NULL,
        author_role {text} NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS notifications (
        id {auto},
        user_id INTEGER,
        faculty_id INTEGER,
        complaint_id INTEGER,
        message TEXT NOT NULL,
        is_read INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # Self-healing migration for existing databases missing new columns
    migrations = [
        ('assigned_faculty_id', 'INT' if mysql else 'INTEGER'),
        ('completion_note', 'TEXT'),
        ('student_feedback', 'TEXT'),
        ('closed_at', 'TIMESTAMP NULL')
    ]
    for column, definition in migrations:
        try:
            cursor.execute(f'ALTER TABLE complaints ADD COLUMN {column} {definition}')
        except Exception:
            pass


def init_sqlite():
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    _schema(cursor, mysql=False)
    conn.commit()
    conn.close()


def init_mysql():
    import mysql.connector
    server = None
    conn = None
    try:
        server = mysql.connector.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD)
        cursor = server.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
        cursor.close()
    finally:
        if server:
            try:
                server.close()
            except Exception:
                pass

    try:
        conn = mysql.connector.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DATABASE)
        cursor = conn.cursor()
        _schema(cursor, mysql=True)
        conn.commit()
        cursor.close()
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def determine_backend():
    global _active_backend
    if _active_backend:
        return _active_backend
    if PREFERRED_DB == 'sqlite':
        init_sqlite()
        _active_backend = 'sqlite'
        return _active_backend
    if PREFERRED_DB == 'mysql':
        init_mysql()
        _active_backend = 'mysql'
        print(f"[Database] MySQL connected: {MYSQL_DATABASE}")
        return _active_backend
    try:
        init_mysql()
        _active_backend = 'mysql'
        print(f"[Database] MySQL connected: {MYSQL_DATABASE}")
    except Exception as err:
        print(f"[Database] MySQL unavailable ({err}); using local SQLite.")
        init_sqlite()
        _active_backend = 'sqlite'
    return _active_backend


def get_backend_name():
    return determine_backend()


def get_connection():
    if determine_backend() == 'mysql':
        import mysql.connector
        return mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(sql, params=(), fetchone=False, fetchall=False, commit=False):
    backend = determine_backend()
    conn = get_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True, buffered=True) if backend == 'mysql' else conn.cursor()
        # Safely replace %s placeholders with ? for SQLite, without affecting words like %suresh
        query = sql if backend == 'mysql' else re.sub(r'%s(?![a-zA-Z0-9_])', '?', sql)
        cursor.execute(query, params)
        if commit:
            conn.commit()
            return cursor.lastrowid
        if fetchone:
            row = cursor.fetchone()
            return row if backend == 'mysql' or row is None else dict(row)
        if fetchall:
            rows = cursor.fetchall()
            return rows if backend == 'mysql' else [dict(row) for row in rows]
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

# Pure MySQL backend integration without hardcoded mock records
