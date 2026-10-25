import os
import sqlite3
from dotenv import load_dotenv

# Load .env file if available
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'grievance_system')
PREFERRED_DB = os.getenv('DB_TYPE', 'auto').lower()

SQLITE_DB_PATH = os.path.join(BASE_DIR, 'database', 'grievance_system.db')

_active_backend = None

def init_sqlite():
    """Initialize local SQLite database with required tables and test student."""
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'student'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            description TEXT NOT NULL,
            predicted_department TEXT,
            priority_level TEXT DEFAULT 'Normal',
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO users (id, roll_number, name, password, role)
        VALUES (1, '123456', 'Test Student', 'password123', 'student')
    """)
    conn.commit()
    conn.close()

def init_mysql():
    """Attempt to connect to MySQL and initialize schema."""
    import mysql.connector

    # First attempt connection to MySQL server directly
    server_conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    server_cursor = server_conn.cursor()
    server_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
    server_cursor.close()
    server_conn.close()

    # Now connect to grievance_system database
    db = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            roll_number VARCHAR(20) NOT NULL UNIQUE,
            name VARCHAR(100) NOT NULL,
            password VARCHAR(100) NOT NULL,
            role VARCHAR(20) DEFAULT 'student'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id INT,
            description TEXT NOT NULL,
            predicted_department VARCHAR(50),
            priority_level VARCHAR(20) DEFAULT 'Normal',
            status VARCHAR(20) DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        INSERT IGNORE INTO users (roll_number, name, password, role)
        VALUES ('123456', 'Test Student', 'password123', 'student')
    """)
    db.commit()
    cursor.close()
    db.close()

def determine_backend():
    """Determine whether MySQL or SQLite should be used."""
    global _active_backend
    if _active_backend is not None:
        return _active_backend

    if PREFERRED_DB == 'sqlite':
        print("[Database] Using SQLite database as configured.")
        init_sqlite()
        _active_backend = 'sqlite'
        return _active_backend

    # Try MySQL first
    try:
        import mysql.connector
        init_mysql()
        print(f"[Database] Successfully connected to MySQL ('{MYSQL_DATABASE}' @ {MYSQL_HOST}:{MYSQL_PORT}).")
        _active_backend = 'mysql'
        return _active_backend
    except Exception as err:
        print(f"[Database Warning] MySQL connection failed ({err}).")
        print("[Database] Falling back seamlessly to local SQLite database.")
        init_sqlite()
        _active_backend = 'sqlite'
        return _active_backend

def get_backend_name():
    return determine_backend()

def get_connection():
    backend = determine_backend()
    if backend == 'mysql':
        import mysql.connector
        return mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def execute_query(sql, params=(), fetchone=False, fetchall=False, commit=False):
    """
    Unified query runner. Automatically maps SQL placeholders (%s vs ?)
    and returns dictionary-like records.
    """
    backend = determine_backend()
    conn = get_connection()

    try:
        if backend == 'mysql':
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params)
            if commit:
                conn.commit()
                last_id = cursor.lastrowid
                cursor.close()
                return last_id
            if fetchone:
                row = cursor.fetchone()
                cursor.close()
                return row
            if fetchall:
                rows = cursor.fetchall()
                cursor.close()
                return rows
            cursor.close()
            return None
        else:
            # For SQLite, convert %s to ?
            sqlite_sql = sql.replace('%s', '?')
            cursor = conn.cursor()
            cursor.execute(sqlite_sql, params)
            if commit:
                conn.commit()
                last_id = cursor.lastrowid
                cursor.close()
                return last_id
            if fetchone:
                row = cursor.fetchone()
                cursor.close()
                return dict(row) if row else None
            if fetchall:
                rows = cursor.fetchall()
                cursor.close()
                return [dict(r) for r in rows]
            cursor.close()
            return None
    finally:
        conn.close()
