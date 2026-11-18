from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from model.rules import calculate_urgency
from model.nlp_model import predict_department_with_confidence, VALID_DEPARTMENTS
from model.sentiment_model import predict_feedback_sentiment
from model.rag_engine import (
    evaluate_complaint_policy,
    generate_chatbot_response,
    list_policy_documents,
    add_policy_document,
    delete_policy_document,
    retrieve_policies,
    is_gemini_configured
)
from database.db import execute_query, get_backend_name

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'smart-grievance-ai-secret-key-2026')

# Administrative Credentials
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'Pankhuri')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'Pankhuri1007')
VALID_STATUSES = ['Pending', 'In Progress', 'Completed - Awaiting Feedback', 'Closed', 'Resolved', 'Rejected (Policy Violation)']
VALID_FACULTY_STATUSES = ['Pending', 'In Progress', 'Completed - Awaiting Feedback']


# ==============================================================================
# ROLE-BASED ACCESS CONTROL (RBAC) DECORATORS
# ==============================================================================

def student_for_roll(roll_number):
    if not roll_number:
        return None
    cleaned = str(roll_number).strip()
    return execute_query(
        'SELECT id, name, roll_number FROM users WHERE roll_number = %s AND LOWER(role) = %s',
        (cleaned, 'student'),
        fetchone=True
    )

def student_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        role = session.get('role')
        if not role:
            return redirect(url_for('login', tab='student'))
        if role == 'admin':
            flash('Access Restricted: Administrators cannot access Student View. Please use Switch User or log out first.', 'warning')
            return redirect(url_for('admin'))
        if role == 'faculty':
            flash('Access Restricted: Faculty officers cannot access Student View. Please use Switch User or log out first.', 'warning')
            return redirect(url_for('assigned_dashboard'))
        if role != 'student' or not session.get('student_id'):
            session.clear()
            return redirect(url_for('login', tab='student'))
        return view(*args, **kwargs)
    return wrapped

def faculty_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        role = session.get('role')
        if not role:
            return redirect(url_for('login', tab='faculty'))
        if role == 'student':
            flash('Access Denied: Students do not have permission to access the Faculty Assigned Operations Desk.', 'danger')
            return redirect(url_for('student_portal'))
        if role == 'admin':
            flash('Access Restricted: Administrators must use the Central Admin Console.', 'warning')
            return redirect(url_for('admin'))
        if role != 'faculty' or not session.get('faculty_id'):
            session.clear()
            return redirect(url_for('login', tab='faculty'))
        return view(*args, **kwargs)
    return wrapped

def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        role = session.get('role')
        if not role:
            return redirect(url_for('login', tab='admin'))
        if role == 'student':
            flash('Access Denied: Students do not have permission to access the Central Admin Console.', 'danger')
            return redirect(url_for('student_portal'))
        if role == 'faculty':
            flash('Access Denied: Faculty officers do not have permission to access the Central Admin Console.', 'danger')
            return redirect(url_for('assigned_dashboard'))
        if role != 'admin':
            session.clear()
            return redirect(url_for('login', tab='admin'))
        return view(*args, **kwargs)
    return wrapped


# ==============================================================================
# NOTIFICATION & HISTORY HELPERS
# ==============================================================================

def log_access(actor_role, actor_id, actor_name, action, details=None):
    try:
        ip_addr = request.remote_addr if request else None
        execute_query(
            'INSERT INTO access_logs (actor_role, actor_id, actor_name, action, details, ip_address) VALUES (%s, %s, %s, %s, %s, %s)',
            (actor_role, str(actor_id) if actor_id is not None else None, actor_name, action, details[:500] if details else None, ip_addr),
            commit=True
        )
    except Exception as e:
        print(f"[Access Log Error] {e}")

def add_notification(message, complaint_id, user_id=None, faculty_id=None):
    try:
        execute_query(
            'INSERT INTO notifications (user_id, faculty_id, complaint_id, message) VALUES (%s, %s, %s, %s)',
            (user_id, faculty_id, complaint_id, message),
            commit=True
        )
    except Exception as e:
        print(f"[Notification Error] {e}")

def complaint_history(student_id):
    complaints = execute_query('''
        SELECT c.id, c.description, c.predicted_department, c.priority_level, c.urgency_score,
               c.confidence_score, c.alternative_departments, c.status,
               c.created_at, c.completion_note, c.student_feedback, c.feedback_sentiment, c.closed_at,
               c.rag_policy_status, c.rag_policy_citation, c.rag_policy_advisory, c.escalation_level,
               f.name AS faculty_name, f.email AS faculty_email 
        FROM complaints c 
        LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
        WHERE c.student_id = %s 
        ORDER BY c.id DESC
    ''', (student_id,), fetchall=True) or []
    
    if complaints:
        ids = [c['id'] for c in complaints]
        placeholders = ', '.join(['%s'] * len(ids))
        all_comments = execute_query(
            f'SELECT complaint_id, author_name, author_role, message, created_at FROM complaint_comments WHERE complaint_id IN ({placeholders}) ORDER BY id ASC',
            tuple(ids),
            fetchall=True
        ) or []
        comments_by_cid = {}
        for c in all_comments:
            comments_by_cid.setdefault(c['complaint_id'], []).append(c)
        for item in complaints:
            item['comments'] = comments_by_cid.get(item['id'], [])
    return complaints


# ==============================================================================
# MAIN GATEWAY & UNIFIED PORTAL LOGIN
# ==============================================================================

@app.route('/')
def home():
    role = session.get('role')
    if role == 'admin':
        return redirect(url_for('admin'))
    if role == 'faculty' and session.get('faculty_id'):
        return redirect(url_for('assigned_dashboard'))
    if role == 'student' and session.get('student_id'):
        return redirect(url_for('student_portal'))
    return redirect(url_for('login'))

@app.route('/switch')
@app.route('/switch/user')
def switch_user():
    target_tab = request.args.get('tab', 'student').strip().lower()
    if target_tab not in ('student', 'faculty', 'admin'):
        target_tab = 'student'
    session.clear()
    return redirect(url_for('login', tab=target_tab, switch='1'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.args.get('switch') or request.args.get('switched') or request.args.get('logout'):
        session.clear()

    faculty_list = execute_query('SELECT id, name, department, email FROM faculty ORDER BY department ASC, name ASC', fetchall=True) or []
    active_tab = request.args.get('tab', 'student').strip().lower()
    if active_tab not in ('student', 'faculty', 'admin'):
        active_tab = 'student'

    error = None
    if request.method == 'POST':
        role_type = request.form.get('role_type', '').strip().lower()

        # 1. STUDENT LOGIN
        if role_type == 'student':
            active_tab = 'student'
            roll = request.form.get('roll_number', '').strip()
            if not roll:
                error = 'Please enter your Registration Number.'
            else:
                student = student_for_roll(roll)
                if student:
                    session.clear()
                    session['role'] = 'student'
                    session['student_id'] = student['id']
                    session['student_name'] = student['name']
                    session['roll_number'] = student['roll_number']
                    log_access('student', student['id'], student['name'], 'LOGIN', f"Roll: {student['roll_number']}")
                    return redirect(url_for('student_portal'))
                error = f"Registration number '{roll}' was not found in the university database. Please verify your Registration Number."

        # 2. FACULTY ASSIGNED DESK LOGIN
        elif role_type == 'faculty':
            active_tab = 'faculty'
            faculty_id = request.form.get('faculty_id', '').strip()
            matched = None

            if faculty_id.isdigit():
                matched = execute_query(
                    'SELECT id, name, department, email FROM faculty WHERE id = %s',
                    (int(faculty_id),),
                    fetchone=True
                )

            if matched:
                session.clear()
                session['role'] = 'faculty'
                session['faculty_id'] = matched['id']
                session['faculty_name'] = matched['name']
                session['faculty_dept'] = matched['department']
                log_access('faculty', matched['id'], matched['name'], 'LOGIN', f"Dept: {matched['department']}")
                return redirect(url_for('assigned_dashboard'))
            error = "Please select a valid assigned officer from the list."

        # 3. ADMIN CONSOLE LOGIN
        elif role_type == 'admin':
            active_tab = 'admin'
            user = request.form.get('username', '').strip()
            pwd = request.form.get('password', '').strip()
            if ADMIN_USERNAME and ADMIN_PASSWORD and user == ADMIN_USERNAME and pwd == ADMIN_PASSWORD:
                session.clear()
                session['role'] = 'admin'
                session['admin_user'] = ADMIN_USERNAME
                log_access('admin', ADMIN_USERNAME, ADMIN_USERNAME, 'LOGIN', 'Admin Console Access')
                return redirect(url_for('admin'))
            error = 'Invalid administrator credentials. Please check your username and password.'

    return render_template(
        'login.html',
        faculty_list=faculty_list,
        active_tab=active_tab,
        error=error
    )

@app.route('/admin/login')
def admin_login():
    return redirect(url_for('login', tab='admin'))

@app.route('/faculty/login')
def faculty_login():
    return redirect(url_for('login', tab='faculty'))

@app.route('/student/login')
def student_login_redirect():
    return redirect(url_for('login', tab='student'))


# ==============================================================================
# MODULE 1: USER VIEW (STUDENT GRIEVANCE PORTAL)
# ==============================================================================

@app.route('/student')
@student_required
def student_portal():
    student = execute_query('SELECT id, name, roll_number FROM users WHERE id = %s', (session['student_id'],), fetchone=True)
    if not student:
        session.clear()
        return redirect(url_for('login', tab='student'))

    alerts = execute_query(
        'SELECT id, message, created_at FROM notifications WHERE user_id = %s ORDER BY id DESC LIMIT 20',
        (student['id'],),
        fetchall=True
    ) or []

    submitted_id = request.args.get('submitted')
    result = None
    if submitted_id:
        comp = execute_query('''
            SELECT c.id, c.description, c.predicted_department, c.priority_level, c.urgency_score,
                   c.confidence_score, c.alternative_departments, c.status,
                   c.rag_policy_status, c.rag_policy_citation, c.rag_policy_advisory,
                   u.name AS student_name, u.roll_number, f.name AS faculty_name, f.email AS faculty_email
            FROM complaints c
            JOIN users u ON c.student_id = u.id
            LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
            WHERE c.id = %s AND c.student_id = %s
        ''', (submitted_id, student['id']), fetchone=True)
        if comp:
            result = {
                'success': True,
                'id': comp['id'],
                'student_name': comp['student_name'],
                'roll_number': comp['roll_number'],
                'department': comp['predicted_department'],
                'faculty_name': comp['faculty_name'] or f"Department Desk ({comp['predicted_department']})",
                'faculty_email': comp.get('faculty_email') or 'desk@college.edu',
                'priority': comp['priority_level'],
                'urgency_score': comp.get('urgency_score') or 25,
                'confidence_score': comp.get('confidence_score'),
                'alternative_departments': comp.get('alternative_departments'),
                'status': comp['status'],
                'rag_policy_status': comp.get('rag_policy_status') or 'Acceptable',
                'rag_policy_citation': comp.get('rag_policy_citation'),
                'rag_policy_advisory': comp.get('rag_policy_advisory'),
                'description': comp['description']
            }

    history = complaint_history(student['id'])
    
    requested_tab = request.args.get('tab', '').strip().lower()
    if requested_tab in ('request', 'status', 'notifications'):
        active_tab = requested_tab
    elif result:
        active_tab = 'request'
    elif history:
        active_tab = 'status'
    else:
        active_tab = 'request'

    return render_template(
        'submit_form.html',
        backend=get_backend_name(),
        student=student,
        history=history,
        alerts=alerts,
        result=result,
        active_tab=active_tab,
        gemini_enabled=is_gemini_configured()
    )

@app.route('/submit', methods=['POST'])
@student_required
def submit():
    student = execute_query('SELECT id, name FROM users WHERE id = %s', (session['student_id'],), fetchone=True)
    if not student:
        session.clear()
        return redirect(url_for('login', tab='student'))

    description = request.form.get('description', '').strip()
    if not description:
        return redirect(url_for('student_portal', error='Please provide a description of your grievance.', tab='request'))

    # 1. Multiclass ML Department Classification (with confidence & alternatives)
    pred = predict_department_with_confidence(description)
    department = pred['department']
    confidence_score = pred['confidence']
    alt_str = ', '.join(f"{a['department']} ({a['confidence']}%)" for a in pred.get('alternatives', [])) or None

    # 2. Numerical Urgency Scoring (0-100) + High / Medium / Low Priority
    urgency_score, priority, _ = calculate_urgency(description)

    # 3. RAG Policy Verification against University Terms & Uploaded Documents
    policy_eval = evaluate_complaint_policy(description, department=department)

    # 4. Intelligently assign to the faculty member in that department with the lowest active workload
    faculty = execute_query('''
        SELECT f.id, f.name, f.department, f.email, COUNT(c.id) AS active_load
        FROM faculty f
        LEFT JOIN complaints c ON f.id = c.assigned_faculty_id AND c.status NOT IN ('Closed', 'Resolved', 'Rejected (Policy Violation)')
        WHERE f.department = %s
        GROUP BY f.id, f.name, f.department, f.email
        ORDER BY active_load ASC, f.id ASC
        LIMIT 1
    ''', (department,), fetchone=True)

    if not faculty:
        faculty = execute_query('''
            SELECT f.id, f.name, f.department, f.email, COUNT(c.id) AS active_load
            FROM faculty f
            LEFT JOIN complaints c ON f.id = c.assigned_faculty_id AND c.status NOT IN ('Closed', 'Resolved', 'Rejected (Policy Violation)')
            GROUP BY f.id, f.name, f.department, f.email
            ORDER BY active_load ASC, f.id ASC
            LIMIT 1
        ''', fetchone=True)
        if faculty:
            department = faculty['department']

    is_violation = (policy_eval['status'] == 'Policy Violation')
    ticket_status = 'Rejected (Policy Violation)' if is_violation else 'Pending'
    assigned_fid = None if is_violation else (faculty['id'] if faculty else None)

    complaint_id = execute_query(
        '''INSERT INTO complaints (
               student_id, description, predicted_department, assigned_faculty_id,
               priority_level, urgency_score, confidence_score, alternative_departments,
               status, rag_policy_status, rag_policy_citation, rag_policy_advisory, escalation_level
           ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
        (
            student['id'], description, department, assigned_fid,
            priority, urgency_score, confidence_score, alt_str,
            ticket_status, policy_eval['status'], policy_eval['citation'], policy_eval['advisory'], 0
        ),
        commit=True
    )

    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, student['name'], 'Student', 'Grievance submitted for review.'),
        commit=True
    )

    if is_violation:
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, 'University Policy RAG', 'System', f"Policy Check [{policy_eval['citation']}]: {policy_eval['advisory']}"),
            commit=True
        )
        add_notification(
            f"Grievance #{complaint_id} was flagged under {policy_eval['citation']}. Please review the policy advisory.",
            complaint_id,
            user_id=student['id']
        )
    else:
        faculty_text = f"{faculty['name']} ({faculty['department']})" if faculty else department
        add_notification(
            f"Grievance #{complaint_id} routed to {faculty_text}. {priority} (Urgency Score: {urgency_score}/100) · Policy: {policy_eval['citation']}.",
            complaint_id,
            user_id=student['id'],
            faculty_id=assigned_fid
        )

    log_access('student', student['id'], student['name'], 'SUBMIT_GRIEVANCE', f"Ticket #{complaint_id} -> {department} ({confidence_score}%), Score {urgency_score}/100, RAG: {policy_eval['status']}")
    return redirect(url_for('student_portal', submitted=complaint_id, tab='request'))

@app.route('/student/escalate/<int:complaint_id>', methods=['POST'])
@student_required
def student_escalate(complaint_id):
    student_id = session.get('student_id')
    student_name = session.get('student_name', 'Student')
    reason = request.form.get('escalation_reason', '').strip() or "Student requested urgent escalation for faster redressal."

    comp = execute_query(
        'SELECT id, assigned_faculty_id, status FROM complaints WHERE id = %s AND student_id = %s',
        (complaint_id, student_id),
        fetchone=True
    )
    if not comp:
        flash('Grievance not found.', 'danger')
        return redirect(url_for('student_portal', tab='status'))

    if comp['status'] in ('Closed', 'Resolved', 'Rejected (Policy Violation)'):
        flash('Closed or policy-rejected grievances cannot be escalated.', 'warning')
        return redirect(url_for('student_portal', tab='status'))

    execute_query(
        "UPDATE complaints SET priority_level = 'High Priority', urgency_score = GREATEST(COALESCE(urgency_score, 25), 85), escalation_level = COALESCE(escalation_level, 0) + 1 WHERE id = %s",
        (complaint_id,),
        commit=True
    )
    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, f"{student_name} (Escalation)", 'Student', f"[ESCALATED TO HIGH PRIORITY]: {reason[:400]}"),
        commit=True
    )
    add_notification(
        f"Grievance #{complaint_id} was escalated to High Priority by student: {reason[:90]}",
        complaint_id,
        user_id=student_id,
        faculty_id=comp.get('assigned_faculty_id')
    )
    log_access('student', student_id, student_name, 'ESCALATE_GRIEVANCE', f"Ticket #{complaint_id}")
    flash(f'Grievance #{complaint_id} has been escalated to High Priority (Urgency Score >= 85/100).', 'warning')
    return redirect(url_for('student_portal', tab='status'))

@app.route('/student/comment/<int:complaint_id>', methods=['POST'])
@student_required
def student_add_comment(complaint_id):
    student = execute_query('SELECT id, name FROM users WHERE id = %s', (session['student_id'],), fetchone=True)
    if not student:
        session.clear()
        return redirect(url_for('login', tab='student'))

    message = request.form.get('message', '').strip()
    complaint = execute_query(
        'SELECT id, student_id, assigned_faculty_id FROM complaints WHERE id = %s AND student_id = %s AND status NOT IN (%s, %s, %s)',
        (complaint_id, student['id'], 'Closed', 'Resolved', 'Rejected (Policy Violation)'),
        fetchone=True
    )
    if complaint and message:
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, student['name'], 'Student', message[:500]),
            commit=True
        )
        add_notification(
            f"Student reply on #{complaint_id}: {message[:100]}",
            complaint_id,
            faculty_id=complaint.get('assigned_faculty_id')
        )
        flash(f'Reply sent on Grievance #{complaint_id}.', 'success')
    return redirect(url_for('student_portal', tab='status'))

@app.route('/feedback/<int:complaint_id>', methods=['POST'])
@student_required
def submit_feedback(complaint_id):
    feedback = request.form.get('feedback', '').strip()
    complaint = execute_query(
        "SELECT id, assigned_faculty_id FROM complaints WHERE id = %s AND student_id = %s AND status = %s AND (student_feedback IS NULL OR student_feedback = '')",
        (complaint_id, session['student_id'], 'Completed - Awaiting Feedback'),
        fetchone=True
    )
    if complaint and feedback:
        sentiment, conf = predict_feedback_sentiment(feedback)
        execute_query(
            'UPDATE complaints SET student_feedback = %s, feedback_sentiment = %s WHERE id = %s',
            (feedback[:500], sentiment, complaint_id),
            commit=True
        )
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, session.get('student_name', 'Student'), 'Student', f"Student Feedback ({sentiment}): {feedback[:500]}"),
            commit=True
        )
        add_notification(
            f'Student submitted {sentiment.lower()} feedback on grievance #{complaint_id}. Awaiting central admin review.',
            complaint_id,
            faculty_id=complaint.get('assigned_faculty_id')
        )
        flash('Thank you! Your resolution feedback has been recorded and submitted to University Administration for review and case closure.', 'success')
    return redirect(url_for('student_portal', tab='status'))


# ==============================================================================
# MODULE 2: ASSIGNED TASK ROLES (FACULTY OPERATIONS & RESOLUTION DESK)
# ==============================================================================

@app.route('/faculty')
@app.route('/assigned')
@faculty_required
def assigned_dashboard():
    active_faculty_id = session.get('faculty_id')
    current_faculty = execute_query('SELECT id, name, department, email FROM faculty WHERE id = %s', (active_faculty_id,), fetchone=True)
    if not current_faculty:
        session.clear()
        return redirect(url_for('login', tab='faculty'))

    selected_status = request.args.get('status', 'all').strip()

    # Query grievances assigned to this specific faculty member
    query = '''
        SELECT c.id, u.roll_number, u.name AS student_name, c.description, 
               c.predicted_department, c.priority_level, c.urgency_score, c.confidence_score,
               c.alternative_departments, c.status, c.created_at, 
               c.completion_note, c.student_feedback, c.feedback_sentiment, c.assigned_faculty_id, c.closed_at,
               c.rag_policy_status, c.rag_policy_citation, c.rag_policy_advisory, c.escalation_level,
               f.name AS faculty_name, f.email AS faculty_email 
        FROM complaints c 
        LEFT JOIN users u ON c.student_id = u.id 
        LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
        WHERE c.assigned_faculty_id = %s
    '''
    conditions = []
    params = [active_faculty_id]

    if selected_status and selected_status != 'all':
        if selected_status == 'High Priority':
            conditions.append("c.priority_level = 'High Priority'")
        elif selected_status == 'Closed':
            conditions.append("c.status IN ('Closed', 'Resolved')")
        else:
            conditions.append('c.status = %s')
            params.append(selected_status)

    if conditions:
        query += ' AND ' + ' AND '.join(conditions)

    query += " ORDER BY CASE c.priority_level WHEN 'High Priority' THEN 0 WHEN 'Medium Priority' THEN 1 ELSE 2 END, COALESCE(c.urgency_score, 25) DESC, c.id DESC"
    complaints = execute_query(query, tuple(params), fetchall=True) or []

    if complaints:
        ids = [c['id'] for c in complaints]
        placeholders = ', '.join(['%s'] * len(ids))
        all_comments = execute_query(
            f'SELECT complaint_id, author_name, author_role, message, created_at FROM complaint_comments WHERE complaint_id IN ({placeholders}) ORDER BY id ASC',
            tuple(ids),
            fetchall=True
        ) or []
        comments_by_cid = {}
        for cmt in all_comments:
            comments_by_cid.setdefault(cmt['complaint_id'], []).append(cmt)
        for comp in complaints:
            comp['comments'] = comments_by_cid.get(comp['id'], [])

    # Officer's Personal Task Stats
    my_all = execute_query("SELECT priority_level, status FROM complaints WHERE assigned_faculty_id = %s", (active_faculty_id,), fetchall=True) or []
    stats = {
        'total': len(my_all),
        'high_priority': sum(x['priority_level'] == 'High Priority' for x in my_all),
        'pending': sum(x['status'] == 'Pending' for x in my_all),
        'in_progress': sum(x['status'] == 'In Progress' for x in my_all),
        'awaiting_feedback': sum(x['status'] == 'Completed - Awaiting Feedback' for x in my_all),
        'closed': sum(x['status'] in ('Closed', 'Resolved') for x in my_all)
    }

    # Faculty list for transfer option
    other_faculty = execute_query('SELECT id, name, department, email FROM faculty WHERE id != %s ORDER BY department ASC, name ASC', (active_faculty_id,), fetchall=True) or []

    return render_template(
        'assigned_dashboard.html',
        complaints=complaints,
        stats=stats,
        backend=get_backend_name(),
        current_faculty=current_faculty,
        other_faculty=other_faculty,
        selected_status=selected_status
    )

@app.route('/faculty/update-status/<int:complaint_id>', methods=['POST'])
@faculty_required
def faculty_update_status(complaint_id):
    status = request.form.get('status', '').strip()
    active_faculty_id = session.get('faculty_id')

    if status in ('Closed', 'Resolved'):
        flash('Access Denied: Faculty cannot close complaints. Only Central University Administration can mark a grievance closed after verified positive student feedback.', 'danger')
        return redirect(url_for('assigned_dashboard'))

    complaint = execute_query(
        'SELECT student_id, assigned_faculty_id, status FROM complaints WHERE id = %s AND assigned_faculty_id = %s',
        (complaint_id, active_faculty_id),
        fetchone=True
    )

    if not complaint:
        flash('That grievance is not assigned to your account.', 'danger')
        return redirect(url_for('assigned_dashboard'))

    if complaint['status'] in ('Closed', 'Resolved', 'Rejected (Policy Violation)'):
        flash('This grievance is closed or policy-rejected and cannot be modified.', 'warning')
        return redirect(url_for('assigned_dashboard'))

    if status in VALID_FACULTY_STATUSES:
        author_name = session.get('faculty_name', 'Assigned Officer')
        execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, author_name, 'Faculty', f"Working status updated to: {status}"),
            commit=True
        )
        add_notification(
            f"Officer {author_name} updated ticket #{complaint_id} status to: {status}.",
            complaint_id,
            user_id=complaint['student_id']
        )
        flash(f"Grievance #{complaint_id} status updated to '{status}'.", 'success')
    return redirect(url_for('assigned_dashboard'))

@app.route('/faculty/comment/<int:complaint_id>', methods=['POST'])
@faculty_required
def faculty_add_comment(complaint_id):
    message = request.form.get('message', '').strip()
    active_faculty_id = session.get('faculty_id')
    complaint = execute_query(
        'SELECT student_id FROM complaints WHERE id = %s AND assigned_faculty_id = %s AND status NOT IN (%s, %s)',
        (complaint_id, active_faculty_id, 'Closed', 'Resolved'),
        fetchone=True
    )

    if complaint and message:
        author_name = session.get('faculty_name', 'Assigned Officer')
        dept_name = session.get('faculty_dept', 'Department')
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, author_name, 'Faculty', message[:500]),
            commit=True
        )
        add_notification(
            f"Officer {author_name} ({dept_name}) replied on #{complaint_id}: {message[:100]}",
            complaint_id,
            user_id=complaint['student_id']
        )
        flash(f'Reply sent to student on Grievance #{complaint_id}.', 'success')
    return redirect(url_for('assigned_dashboard'))

@app.route('/faculty/complete/<int:complaint_id>', methods=['POST'])
@faculty_required
def faculty_complete_complaint(complaint_id):
    note = request.form.get('completion_note', '').strip()
    active_faculty_id = session.get('faculty_id')
    complaint = execute_query(
        'SELECT student_id FROM complaints WHERE id = %s AND assigned_faculty_id = %s AND status NOT IN (%s, %s)',
        (complaint_id, active_faculty_id, 'Closed', 'Resolved'),
        fetchone=True
    )

    if complaint and note:
        author_name = session.get('faculty_name', 'Assigned Officer')
        execute_query(
            'UPDATE complaints SET status = %s, completion_note = %s WHERE id = %s',
            ('Completed - Awaiting Feedback', note[:500], complaint_id),
            commit=True
        )
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, author_name, 'Faculty', f"Official Resolution: {note[:500]}"),
            commit=True
        )
        add_notification(
            f"Officer {author_name} marked grievance #{complaint_id} completed. Please review remarks and submit feedback.",
            complaint_id,
            user_id=complaint['student_id']
        )
        flash(f'Grievance #{complaint_id} marked Completed - Awaiting Student Feedback.', 'success')
    return redirect(url_for('assigned_dashboard'))

@app.route('/faculty/reassign/<int:complaint_id>', methods=['POST'])
@faculty_required
def faculty_reassign(complaint_id):
    target_faculty_id = request.form.get('faculty_id', '').strip()
    active_faculty_id = session.get('faculty_id')
    if not target_faculty_id.isdigit() or int(target_faculty_id) == int(active_faculty_id):
        flash('Please select a different faculty officer to transfer this ticket.', 'warning')
        return redirect(url_for('assigned_dashboard'))

    complaint = execute_query(
        'SELECT id, student_id FROM complaints WHERE id = %s AND assigned_faculty_id = %s AND status NOT IN (%s, %s)',
        (complaint_id, active_faculty_id, 'Closed', 'Resolved'),
        fetchone=True
    )
    target_faculty = execute_query(
        'SELECT id, name, department FROM faculty WHERE id = %s',
        (int(target_faculty_id),),
        fetchone=True
    )

    if complaint and target_faculty:
        old_name = session.get('faculty_name', 'Assigned Officer')
        execute_query(
            'UPDATE complaints SET assigned_faculty_id = %s, predicted_department = %s WHERE id = %s',
            (target_faculty['id'], target_faculty['department'], complaint_id),
            commit=True
        )
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, old_name, 'Faculty', f"Transferred job to {target_faculty['name']} ({target_faculty['department']} department)."),
            commit=True
        )
        add_notification(
            f"Grievance #{complaint_id} was transferred to {target_faculty['name']} ({target_faculty['department']}).",
            complaint_id,
            user_id=complaint['student_id'],
            faculty_id=target_faculty['id']
        )
        flash(f"Grievance #{complaint_id} transferred to {target_faculty['name']} ({target_faculty['department']}).", 'info')
    return redirect(url_for('assigned_dashboard'))


# ==============================================================================
# MODULE 3: ADMIN VIEW (CENTRAL CAMPUS GOVERNANCE & TRIAGE HEADQUARTERS)
# ==============================================================================

@app.route('/admin')
@admin_required
def admin():
    selected_dept = request.args.get('dept', 'all').strip()
    selected_status = request.args.get('status', 'all').strip()

    faculty_list = execute_query('SELECT id, name, department, email FROM faculty ORDER BY department ASC, name ASC', fetchall=True) or []
    departments = sorted(list({f['department'] for f in faculty_list if f.get('department')}))

    # Admin Exclusive: Faculty Active Workload Table
    faculty_workloads = execute_query('''
        SELECT f.id, f.name, f.department, f.email,
               COUNT(CASE WHEN c.status NOT IN ('Closed', 'Resolved', 'Rejected (Policy Violation)') THEN c.id END) AS active_load,
               COUNT(c.id) AS total_handled
        FROM faculty f
        LEFT JOIN complaints c ON f.id = c.assigned_faculty_id
        GROUP BY f.id, f.name, f.department, f.email
        ORDER BY f.department ASC, active_load DESC
    ''', fetchall=True) or []

    query = '''
        SELECT c.id, u.roll_number, u.name AS student_name, c.description, 
               c.predicted_department, c.priority_level, c.urgency_score, c.confidence_score,
               c.alternative_departments, c.status, c.created_at, 
               c.completion_note, c.student_feedback, c.feedback_sentiment, c.assigned_faculty_id,
               c.rag_policy_status, c.rag_policy_citation, c.rag_policy_advisory, c.escalation_level,
               f.name AS faculty_name, f.email AS faculty_email 
        FROM complaints c 
        LEFT JOIN users u ON c.student_id = u.id 
        LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
    '''
    conditions = []
    params = []

    if selected_dept and selected_dept != 'all':
        conditions.append('c.predicted_department = %s')
        params.append(selected_dept)

    if selected_status and selected_status != 'all':
        if selected_status == 'Closed':
            conditions.append("c.status IN ('Closed', 'Resolved')")
        else:
            conditions.append('c.status = %s')
            params.append(selected_status)

    if conditions:
        query += ' WHERE ' + ' AND '.join(conditions)

    query += " ORDER BY CASE c.priority_level WHEN 'High Priority' THEN 0 WHEN 'Medium Priority' THEN 1 ELSE 2 END, COALESCE(c.urgency_score, 25) DESC, c.id DESC"
    complaints = execute_query(query, tuple(params), fetchall=True) or []

    if complaints:
        ids = [c['id'] for c in complaints]
        placeholders = ', '.join(['%s'] * len(ids))
        all_comments = execute_query(
            f'SELECT complaint_id, author_name, author_role, message, created_at FROM complaint_comments WHERE complaint_id IN ({placeholders}) ORDER BY id ASC',
            tuple(ids),
            fetchall=True
        ) or []
        comments_by_cid = {}
        for cmt in all_comments:
            comments_by_cid.setdefault(cmt['complaint_id'], []).append(cmt)
        for comp in complaints:
            comp['comments'] = comments_by_cid.get(comp['id'], [])

    # System-wide Overall Campus Stats
    all_complaints = execute_query("SELECT priority_level, status FROM complaints", fetchall=True) or []
    stats = {
        'total': len(all_complaints),
        'high_priority': sum(x['priority_level'] == 'High Priority' for x in all_complaints),
        'pending': sum(x['status'] == 'Pending' for x in all_complaints),
        'in_progress': sum(x['status'] == 'In Progress' for x in all_complaints),
        'awaiting_feedback': sum(x['status'] == 'Completed - Awaiting Feedback' for x in all_complaints),
        'resolved': sum(x['status'] in ('Closed', 'Resolved') for x in all_complaints)
    }

    rag_documents = list_policy_documents()

    return render_template(
        'admin_dashboard.html',
        complaints=complaints,
        stats=stats,
        backend=get_backend_name(),
        faculty_list=faculty_list,
        faculty_workloads=faculty_workloads,
        departments=departments,
        rag_documents=rag_documents,
        valid_departments=VALID_DEPARTMENTS + ['General'],
        selected_dept=selected_dept,
        selected_status=selected_status,
        admin_name=ADMIN_USERNAME
    )

@app.route('/admin/update-status/<int:complaint_id>', methods=['POST'])
@admin_required
def update_status(complaint_id):
    status = request.form.get('status', '').strip()
    complaint = execute_query(
        'SELECT student_id, assigned_faculty_id, status, student_feedback, feedback_sentiment FROM complaints WHERE id = %s',
        (complaint_id,),
        fetchone=True
    )
    if not complaint:
        return redirect(request.referrer or url_for('admin'))

    if complaint['status'] in ('Closed', 'Resolved', 'Rejected (Policy Violation)'):
        flash(f'Grievance #{complaint_id} is closed or policy-rejected and cannot be modified.', 'warning')
        return redirect(request.referrer or url_for('admin'))

    if status in ('Closed', 'Resolved'):
        # 1. Require student feedback
        if not complaint.get('student_feedback') or not complaint['student_feedback'].strip():
            flash(f'Cannot close Grievance #{complaint_id}: Student has not submitted resolution feedback yet. Cases cannot be closed without student verification.', 'warning')
            return redirect(request.referrer or url_for('admin'))

        # 2. Evaluate sentiment if needed
        sentiment = complaint.get('feedback_sentiment')
        if not sentiment:
            sentiment, _ = predict_feedback_sentiment(complaint['student_feedback'])
            execute_query('UPDATE complaints SET feedback_sentiment = %s WHERE id = %s', (sentiment, complaint_id), commit=True)

        # 3. Require positive feedback
        if sentiment != 'Positive':
            snippet = complaint['student_feedback']
            if len(snippet) > 90:
                snippet = snippet[:90] + '...'
            flash(f'Cannot close Grievance #{complaint_id}: Student feedback was evaluated as Negative/Unsatisfied ("{snippet}"). The issue must be resolved satisfactorily before administrative closure.', 'danger')
            return redirect(request.referrer or url_for('admin'))

        # Positive feedback verified -> close
        execute_query('UPDATE complaints SET status = %s, closed_at = CURRENT_TIMESTAMP WHERE id = %s', ('Closed', complaint_id), commit=True)
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, f'Admin ({ADMIN_USERNAME})', 'Admin', f"Administrative Closure: Case verified and permanently closed following positive student feedback ({sentiment})."),
            commit=True
        )
        add_notification(f'Grievance #{complaint_id} has been verified and permanently closed by Administration.', complaint_id, user_id=complaint['student_id'], faculty_id=complaint.get('assigned_faculty_id'))
        flash(f'Grievance #{complaint_id} verified and permanently closed following positive student feedback.', 'success')
        return redirect(request.referrer or url_for('admin'))

    elif status in ('Pending', 'In Progress', 'Completed - Awaiting Feedback'):
        execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, f'Admin ({ADMIN_USERNAME})', 'Admin', f"Administrative status update: {status}"),
            commit=True
        )
        add_notification(f'Grievance #{complaint_id} status changed to {status}.', complaint_id, user_id=complaint['student_id'])
        flash(f'Grievance #{complaint_id} status updated to {status}.', 'info')

    return redirect(request.referrer or url_for('admin'))

@app.route('/admin/comment/<int:complaint_id>', methods=['POST'])
@admin_required
def add_comment(complaint_id):
    message = request.form.get('message', '').strip()
    complaint = execute_query('SELECT student_id FROM complaints WHERE id = %s AND status NOT IN (%s, %s)', (complaint_id, 'Closed', 'Resolved'), fetchone=True)
    if complaint and message:
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, f'Administrator ({ADMIN_USERNAME})', 'Admin', message[:500]),
            commit=True
        )
        add_notification(f'Central Administrative update on grievance #{complaint_id}: {message[:120]}', complaint_id, user_id=complaint['student_id'])
        flash(f'Update remark sent on Grievance #{complaint_id}.', 'success')
    return redirect(request.referrer or url_for('admin'))

@app.route('/admin/complete/<int:complaint_id>', methods=['POST'])
@admin_required
def complete_complaint(complaint_id):
    note = request.form.get('completion_note', '').strip()
    complaint = execute_query('SELECT student_id FROM complaints WHERE id = %s AND status NOT IN (%s, %s)', (complaint_id, 'Closed', 'Resolved'), fetchone=True)
    if complaint and note:
        execute_query(
            'UPDATE complaints SET status = %s, completion_note = %s WHERE id = %s',
            ('Completed - Awaiting Feedback', note[:500], complaint_id),
            commit=True
        )
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, f'Administrator ({ADMIN_USERNAME})', 'Admin', f"Administrative Resolution: {note[:500]}"),
            commit=True
        )
        add_notification(
            f'Grievance #{complaint_id} was marked completed by University Admin. Please review the note and submit feedback.',
            complaint_id,
            user_id=complaint['student_id']
        )
        flash(f'Grievance #{complaint_id} marked Completed - Awaiting Student Feedback.', 'success')
    return redirect(request.referrer or url_for('admin'))

@app.route('/admin/reassign/<int:complaint_id>', methods=['POST'])
@admin_required
def reassign_complaint(complaint_id):
    faculty_id = request.form.get('faculty_id', '').strip()
    if not faculty_id.isdigit():
        flash('Invalid faculty officer selected.', 'warning')
        return redirect(request.referrer or url_for('admin'))

    complaint = execute_query(
        'SELECT id, student_id, predicted_department, status, feedback_sentiment, completion_note, student_feedback FROM complaints WHERE id = %s AND status NOT IN (%s, %s)',
        (complaint_id, 'Closed', 'Resolved'),
        fetchone=True
    )
    faculty = execute_query('SELECT id, name, department FROM faculty WHERE id = %s', (int(faculty_id),), fetchone=True)
    if complaint and faculty:
        reopening_negative = (
            complaint['status'] == 'Completed - Awaiting Feedback'
            and complaint.get('feedback_sentiment') == 'Negative'
        )
        execute_query(
            '''UPDATE complaints
               SET assigned_faculty_id = %s, predicted_department = %s, status = %s,
                   completion_note = %s, student_feedback = %s, feedback_sentiment = %s
               WHERE id = %s''',
            (
                faculty['id'],
                faculty['department'],
                'In Progress' if reopening_negative else complaint['status'],
                None if reopening_negative else complaint.get('completion_note'),
                None if reopening_negative else complaint.get('student_feedback'),
                None if reopening_negative else complaint.get('feedback_sentiment'),
                complaint_id
            ),
            commit=True
        )
        comment_text = (
            f"Reopened after negative feedback and re-routed by Central Admin to {faculty['name']} ({faculty['department']})."
            if reopening_negative
            else f"Re-routed by Central Admin to {faculty['name']} ({faculty['department']})."
        )
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, f'Admin ({ADMIN_USERNAME})', 'Admin', comment_text),
            commit=True
        )
        add_notification(
            f"Grievance #{complaint_id} has been re-assigned to {faculty['name']} ({faculty['department']}).",
            complaint_id,
            user_id=complaint['student_id'],
            faculty_id=faculty['id']
        )
        flash(f"Grievance #{complaint_id} transferred to {faculty['name']} ({faculty['department']}).", 'info')
    return redirect(request.referrer or url_for('admin'))

@app.route('/admin/rag/upload', methods=['POST'])
@admin_required
def admin_upload_rag_doc():
    title = request.form.get('title', '').strip()
    department = request.form.get('department', 'General').strip() or 'General'
    citation = request.form.get('citation', '').strip() or 'University Circular / Uploaded Policy'
    content = request.form.get('content', '').strip()

    uploaded_file = request.files.get('policy_file')
    if uploaded_file and uploaded_file.filename:
        fname = uploaded_file.filename.lower()
        if not (fname.endswith('.txt') or fname.endswith('.md')):
            flash('Please upload a plain text (.txt) or Markdown (.md) policy document.', 'warning')
            return redirect(url_for('admin'))
        try:
            raw_bytes = uploaded_file.read(200_000)
            file_text = raw_bytes.decode('utf-8', errors='ignore').strip()
            if file_text:
                content = file_text
            if not title:
                title = os.path.splitext(os.path.basename(uploaded_file.filename))[0]
        except Exception as e:
            flash(f'Could not read uploaded file: {e}', 'danger')
            return redirect(url_for('admin'))

    if not title or not content or len(content) < 20:
        flash('Please provide a policy title and at least 20 characters of policy text (or upload a .txt/.md file).', 'warning')
        return redirect(url_for('admin'))

    doc_id, doc_code = add_policy_document(title, department, citation, content)
    log_access('admin', ADMIN_USERNAME, ADMIN_USERNAME, 'UPLOAD_RAG_DOC', f"{doc_code}: {title} ({department})")
    flash(f"University policy document '{title}' ({doc_code}) indexed into RAG Knowledge Base!", 'success')
    return redirect(url_for('admin'))


@app.route('/admin/rag/delete/<int:doc_id>', methods=['POST'])
@admin_required
def admin_delete_rag_doc(doc_id):
    if delete_policy_document(doc_id):
        log_access('admin', ADMIN_USERNAME, ADMIN_USERNAME, 'DELETE_RAG_DOC', f"Doc ID #{doc_id}")
        flash('Uploaded policy document removed from RAG Knowledge Base.', 'info')
    else:
        flash('Core university policies cannot be deleted (only uploaded documents can be removed).', 'warning')
    return redirect(url_for('admin'))


# ==============================================================================
# MODULE 4: AI CHATBOT, RAG POLICY & ML CLASSIFICATION REST APIs
# ==============================================================================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get('message') or '').strip()
    if len(message) > 1000:
        return jsonify({'error': 'Message too long (max 1000 characters).'}), 400

    student_id = session.get('student_id') if session.get('role') == 'student' else None
    student_name = session.get('student_name') if session.get('role') == 'student' else None
    prev_id = session.get('chat_interaction_id')

    result = generate_chatbot_response(
        message,
        student_id=student_id,
        student_name=student_name,
        previous_interaction_id=prev_id
    )
    if result.get('interaction_id'):
        session['chat_interaction_id'] = result['interaction_id']
    return jsonify(result)


@app.route('/api/chat/reset', methods=['POST'])
def api_chat_reset():
    session.pop('chat_interaction_id', None)
    return jsonify({'reset': True})


@app.route('/api/rag/query', methods=['POST'])
def api_rag_query():
    payload = request.get_json(silent=True) or {}
    query = (payload.get('query') or payload.get('description') or '').strip()
    department = payload.get('department')
    if not query:
        return jsonify({'error': 'Please provide a query string.'}), 400
    docs = retrieve_policies(query, department=department, top_k=3)
    evaluation = evaluate_complaint_policy(query, department=department)
    return jsonify({
        'evaluation': evaluation,
        'documents': [
            {
                'doc_code': d['doc_code'],
                'title': d['title'],
                'department': d['department'],
                'citation': d['citation'],
                'similarity': d['similarity'],
                'content': d['content']
            }
            for d in docs
        ]
    })


@app.route('/api/rag/policies', methods=['GET'])
def api_rag_policies():
    docs = list_policy_documents()
    return jsonify({
        'count': len(docs),
        'policies': [
            {
                'id': d['id'],
                'doc_code': d['doc_code'],
                'title': d['title'],
                'department': d['department'],
                'citation': d['citation'],
                'source_type': d.get('source_type', 'policy'),
                'content': d['content']
            }
            for d in docs
        ]
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get('description') or payload.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Please provide a description.'}), 400

    pred = predict_department_with_confidence(text)
    urgency_score, priority, reasons = calculate_urgency(text)
    policy_eval = evaluate_complaint_policy(text, department=pred['department'])
    return jsonify({
        'department': pred['department'],
        'confidence': pred['confidence'],
        'alternatives': pred['alternatives'],
        'urgency_score': urgency_score,
        'priority': priority,
        'urgency_reasons': reasons,
        'policy': policy_eval
    })


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(
        host=os.getenv('HOST', '127.0.0.1'),
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        threaded=True
    )
