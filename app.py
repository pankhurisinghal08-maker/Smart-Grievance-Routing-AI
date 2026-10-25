from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from model.rules import get_priority_level
from model.nlp_model import predict_department
from database.db import execute_query, get_backend_name, get_connection

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'smart-grievance-routing-ai-key-2026')

def get_db():
    """Backward compatibility helper for direct database connection."""
    return get_connection()

@app.route('/')
def home():
    backend = get_backend_name()
    return render_template('submit_form.html', backend=backend)

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'GET':
        return redirect(url_for('home'))

    roll_number = request.form.get('roll_number', '').strip()
    description = request.form.get('description', '').strip()
    backend = get_backend_name()

    if not roll_number or not description:
        error_result = {
            'success': False,
            'message': 'Please provide both your roll number and a problem description.',
            'roll_number': roll_number,
            'description': description
        }
        return render_template('submit_form.html', result=error_result, backend=backend)

    # 1. AI Department Classification & Rule-based Priority Detection
    department = predict_department(description)
    priority = get_priority_level(description)

    # 2. Look up the student in the database
    student = execute_query(
        "SELECT id, name FROM users WHERE roll_number = %s",
        (roll_number,),
        fetchone=True
    )

    if student:
        student_id = student['id']
        student_name = student.get('name', 'Student')

        # 3. Save complaint into database
        execute_query(
            "INSERT INTO complaints (student_id, description, predicted_department, priority_level) VALUES (%s, %s, %s, %s)",
            (student_id, description, department, priority),
            commit=True
        )

        result = {
            'success': True,
            'message': f"Grievance successfully submitted and routed by AI!",
            'department': department,
            'priority': priority,
            'roll_number': roll_number,
            'student_name': student_name
        }
    else:
        result = {
            'success': False,
            'message': "Error: Roll number not found in the database. Use 123456 to test.",
            'roll_number': roll_number,
            'description': description
        }

    return render_template('submit_form.html', result=result, backend=backend)

@app.route('/admin')
def admin():
    backend = get_backend_name()
    
    # Get all complaints joined with student information
    query = """
        SELECT c.id, u.roll_number, u.name AS student_name, c.description,
               c.predicted_department, c.priority_level, c.status, c.created_at
        FROM complaints c
        LEFT JOIN users u ON c.student_id = u.id
        ORDER BY c.id DESC
    """
    complaints = execute_query(query, fetchall=True) or []

    # Calculate statistics for the dashboard
    stats = {
        'total': len(complaints),
        'high_priority': sum(1 for c in complaints if c.get('priority_level') == 'High Priority'),
        'pending': sum(1 for c in complaints if c.get('status') == 'Pending'),
        'resolved': sum(1 for c in complaints if c.get('status') == 'Resolved'),
    }

    return render_template('admin_dashboard.html', complaints=complaints, stats=stats, backend=backend)

@app.route('/admin/update-status/<int:complaint_id>', methods=['POST'])
def update_status(complaint_id):
    new_status = request.form.get('status', 'Pending').strip()
    valid_statuses = ['Pending', 'In Progress', 'Resolved']

    if new_status in valid_statuses:
        execute_query(
            "UPDATE complaints SET status = %s WHERE id = %s",
            (new_status, complaint_id),
            commit=True
        )

    return redirect(url_for('admin'))

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 'yes')
    print(f"\n========================================================")
    print(f"  Smart Grievance Routing AI Server")
    print(f"  Running on: http://127.0.0.1:{port}")
    print(f"  Admin View: http://127.0.0.1:{port}/admin")
    print(f"========================================================\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
