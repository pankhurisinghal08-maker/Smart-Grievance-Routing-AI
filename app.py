from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from functools import wraps
import os
import csv
import io
import datetime
from model.rules import get_priority_level
from model.nlp_model import predict_department
from model.sentiment_model import predict_feedback_sentiment
from database.db import execute_query, get_backend_name
from model.rag_engine import (
    evaluate_complaint_acceptability,
    find_similar_complaints,
    generate_rag_resolution_draft,
    calculate_hazard_risk_score,
    calculate_dynamic_sla,
    evaluate_sla_breach_risk,
    generate_topic_hashtags,
    CAMPUS_POLICIES
)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or os.urandom(32)

# Admin access is configured via environment variables
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', '')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', '')
VALID_FACULTY_STATUSES = ['Pending', 'In Progress', 'Completed - Awaiting Feedback', 'Completed', 'Waited', 'Waiting']


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
            return redirect(url_for('admin', tab='teachers'))
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
        SELECT c.id, c.description, c.predicted_department, c.priority_level, c.status, 
               c.created_at, c.completion_note, c.student_feedback, c.feedback_sentiment, c.closed_at,
               c.rag_acceptability, c.rag_policy_score, c.rag_policy_citation, c.rag_policy_advisory,
               c.rag_cluster_id, c.sla_deadline, c.sla_breach_risk, c.urgency_score, c.topic_tags, c.escalation_level,
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
            f"SELECT complaint_id, author_name, author_role, message, created_at FROM complaint_comments WHERE complaint_id IN ({placeholders}) AND author_role NOT LIKE '%%To Admin%%' ORDER BY id ASC",
            tuple(ids),
            fetchall=True
        ) or []
        comments_by_cid = {}
        for c in all_comments:
            comments_by_cid.setdefault(c['complaint_id'], []).append(c)
        for item in complaints:
            item['comments'] = comments_by_cid.get(item['id'], [])
            sla_h = 6 if item.get('priority_level') == 'High Priority' else 24
            risk_lvl, rem_str = evaluate_sla_breach_risk(item.get('created_at'), sla_h, item.get('status'))
            item['sla_breach_risk'] = risk_lvl
            item['sla_remaining_text'] = rem_str
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
                    return redirect(url_for('student_portal'))
                error = f"Registration number '{roll}' was not found in the university database. Please verify your Registration Number."

        # 2. FACULTY ASSIGNED DESK LOGIN
        elif role_type == 'faculty':
            active_tab = 'faculty'
            faculty_id = request.form.get('faculty_id')
            matched = None

            if not faculty_id:
                faculty_id = '1'

            matched = execute_query('SELECT id, name, department, email FROM faculty WHERE id = %s', (faculty_id,), fetchone=True)
            if not matched:
                matched = execute_query('SELECT id, name, department, email FROM faculty WHERE id = 1', fetchone=True)

            if matched:
                session.clear()
                session['role'] = 'faculty'
                session['faculty_id'] = str(matched['id'])
                session['faculty_name'] = matched['name']
                session['faculty_dept'] = matched['department']
                return redirect(url_for('assigned_dashboard'))
            error = "Please select an assigned officer from the list."

        # 3. ADMIN CONSOLE LOGIN
        elif role_type == 'admin':
            active_tab = 'admin'
            user = request.form.get('username', '').strip()
            pwd = request.form.get('password', '').strip()
            if ADMIN_USERNAME and ADMIN_PASSWORD and user == ADMIN_USERNAME and pwd == ADMIN_PASSWORD:
                session.clear()
                session['role'] = 'admin'
                session['admin_user'] = ADMIN_USERNAME
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
# MODULE 1: USER VIEW (STUDENT GRIEVANCE PORTAL & RAG POLICY CHECK)
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

    bulletins = execute_query(
        'SELECT id, title, department, message, severity, created_at FROM campus_bulletins WHERE is_active = 1 ORDER BY id DESC LIMIT 5',
        fetchall=True
    ) or []

    submitted_id = request.args.get('submitted')
    result = None
    if submitted_id:
        comp = execute_query('''
            SELECT c.id, c.description, c.predicted_department, c.priority_level, c.status,
                   c.rag_acceptability, c.rag_policy_score, c.rag_policy_citation, c.rag_policy_advisory,
                   c.rag_cluster_id, c.sla_deadline, c.sla_breach_risk, c.urgency_score, c.topic_tags,
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
                'description': comp['description'],
                'status': comp['status'],
                'rag_acceptability': comp.get('rag_acceptability') or 'Acceptable',
                'rag_policy_score': comp.get('rag_policy_score') or 85.0,
                'rag_policy_citation': comp.get('rag_policy_citation') or 'LPU Standard Student Grievance Policy',
                'rag_policy_advisory': comp.get('rag_policy_advisory') or '',
                'rag_cluster_id': comp.get('rag_cluster_id'),
                'sla_deadline': comp.get('sla_deadline'),
                'urgency_score': comp.get('urgency_score') or 15,
                'topic_tags': comp.get('topic_tags') or f"#{comp['predicted_department']}"
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
        bulletins=bulletins,
        policies=CAMPUS_POLICIES,
        active_tab=active_tab
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

    department = predict_department(description)
    priority = get_priority_level(description)

    # 1. RAG Policy & Standards Verification
    rag_result = evaluate_complaint_acceptability(description, department)
    hazard_score, hazard_reasons = calculate_hazard_risk_score(description)
    sla_hours, sla_deadline_dt = calculate_dynamic_sla(priority, department)
    sla_deadline_str = sla_deadline_dt.strftime('%Y-%m-%d %H:%M:%S')
    topic_tags = generate_topic_hashtags(description, department)

    # 2. RAG Semantic Similarity & Clustering against active complaints
    all_existing = execute_query('SELECT id, description, predicted_department, priority_level, status, created_at, rag_cluster_id FROM complaints', fetchall=True) or []
    similar_list, cluster_id, is_duplicate = find_similar_complaints(description, all_existing)
    if similar_list:
        top_match = similar_list[0]
        if top_match.get('rag_cluster_id'):
            cluster_id = top_match['rag_cluster_id']
        else:
            cluster_id = f"CLUST-{department[:4].upper()}-{top_match['id']}"
            execute_query("UPDATE complaints SET rag_cluster_id = %s WHERE id = %s", (cluster_id, top_match['id']), commit=True)
    if not cluster_id:
        cluster_id = f"CLUST-{department[:4].upper()}-{student['id']}"

    # 3. Workload balancing for departmental faculty officer
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
            WHERE f.department = %s
            GROUP BY f.id, f.name, f.department, f.email
            ORDER BY active_load ASC, f.id ASC
            LIMIT 1
        ''', (department,), fetchone=True)

    # 4. Status determination based on RAG Policy Compliance
    if rag_result['status'] == 'Policy Violation':
        status = 'Rejected (Policy Violation)'
        assigned_fid = None
    elif rag_result['status'] == 'Conditional':
        status = 'Conditional (Pending Evidence)'
        assigned_fid = faculty['id'] if faculty else None
    else:
        status = 'Pending'
        assigned_fid = faculty['id'] if faculty else None

    # 5. Insert Complaint with full RAG metadata
    complaint_id = execute_query('''
        INSERT INTO complaints (
            student_id, description, predicted_department, assigned_faculty_id,
            priority_level, status, rag_acceptability, rag_policy_score,
            rag_policy_citation, rag_policy_advisory, rag_cluster_id,
            sla_deadline, sla_breach_risk, urgency_score, topic_tags, escalation_level
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ''', (
        student['id'], description, department, assigned_fid,
        priority, status, rag_result['status'], rag_result['policy_score'],
        rag_result['citation'], rag_result['advisory_response'], cluster_id,
        sla_deadline_str, 'Low', hazard_score, topic_tags, 0
    ), commit=True)

    # 6. Insert clean initial discussion thread entry
    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, student['name'], 'Student', 'Grievance submitted for review.'),
        commit=True
    )

    if rag_result['status'] == 'Policy Violation':
        guardrail_msg = f"Policy Verification [REJECTED]: {rag_result['citation']}. Reason: {rag_result['advisory_response']}"
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, 'AI Policy Guardrail', 'System', guardrail_msg),
            commit=True
        )
        add_notification(
            f"Grievance #{complaint_id} failed policy check under {rag_result['citation']}. Please review advisory.",
            complaint_id,
            user_id=student['id']
        )

        faculty_text = f"{faculty['name']} ({faculty['department']})" if faculty else department
        add_notification(
            f"Grievance #{complaint_id} verified under {rag_result['citation']}. Routed to {faculty_text}. Priority: {priority} (SLA: {sla_hours}h).",
            complaint_id,
            user_id=student['id'],
            faculty_id=faculty['id'] if faculty else None
        )

    return redirect(url_for('student_portal', submitted=complaint_id, tab='request'))

@app.route('/student/comment/<int:complaint_id>', methods=['POST'])
@student_required
def student_add_comment(complaint_id):
    student = execute_query('SELECT id, name FROM users WHERE id = %s', (session['student_id'],), fetchone=True)
    if not student:
        session.clear()
        return redirect(url_for('login', tab='student'))

    message = request.form.get('message', '').strip()
    complaint = execute_query(
        'SELECT id, student_id, assigned_faculty_id FROM complaints WHERE id = %s AND student_id = %s AND status NOT IN (%s, %s)',
        (complaint_id, student['id'], 'Closed', 'Resolved'),
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
    return redirect(url_for('student_portal', tab='status'))

@app.route('/student/escalate/<int:complaint_id>', methods=['POST'])
@student_required
def student_escalate(complaint_id):
    student_id = session.get('student_id')
    student_name = session.get('student_name', 'Student')
    reason = request.form.get('escalation_reason', '').strip()
    if not reason:
        reason = "Student requested urgent managerial review due to delayed resolution or dissatisfaction."

    comp = execute_query(
        'SELECT id, assigned_faculty_id, priority_level, status FROM complaints WHERE id = %s AND student_id = %s',
        (complaint_id, student_id),
        fetchone=True
    )
    if not comp:
        flash('Complaint not found.', 'danger')
        return redirect(url_for('student_portal', tab='status'))

    execute_query(
        "UPDATE complaints SET priority_level = 'High Priority', escalation_level = escalation_level + 1 WHERE id = %s",
        (complaint_id,),
        commit=True
    )
    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, f"{student_name} (Escalation)", 'Student', f"[URGENT CASE ESCALATION]: {reason[:400]}"),
        commit=True
    )
    add_notification(
        f"🚨 Grievance #{complaint_id} was ESCALATED to High Priority by student: {reason[:80]}",
        complaint_id,
        user_id=student_id,
        faculty_id=comp.get('assigned_faculty_id')
    )
    flash(f"Grievance #{complaint_id} has been formally escalated to High Priority. Central Administration and the Department Head have been alerted.", 'warning')
    return redirect(url_for('student_portal', tab='status'))

@app.route('/feedback/<int:complaint_id>', methods=['POST'])
@student_required
def submit_feedback(complaint_id):
    feedback = request.form.get('feedback', '').strip()
    complaint = execute_query(
        'SELECT id, assigned_faculty_id FROM complaints WHERE id = %s AND student_id = %s AND status = %s',
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
# MODULE 2: ASSIGNED TASK ROLES (FACULTY OPERATIONS, RAG CO-PILOT & MASS REPLY)
# ==============================================================================

@app.route('/faculty')
@app.route('/assigned')
@app.route('/assigned_dashboard')
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
               c.predicted_department, c.priority_level, c.status, c.created_at, 
               c.completion_note, c.student_feedback, c.feedback_sentiment, c.assigned_faculty_id, c.closed_at,
               c.rag_acceptability, c.rag_policy_score, c.rag_policy_citation, c.rag_policy_advisory,
               c.rag_cluster_id, c.sla_deadline, c.sla_breach_risk, c.urgency_score, c.topic_tags, c.escalation_level,
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

    query += " ORDER BY CASE c.priority_level WHEN 'High Priority' THEN 0 ELSE 1 END, c.id DESC"
    complaints = execute_query(query, tuple(params), fetchall=True) or []

    # Map comments, dynamic SLAs and similarity cluster counts
    cluster_counts = {}
    cluster_members = {}
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
        
        # Calculate cluster sizes across entire department
        clusters = list({c['rag_cluster_id'] for c in complaints if c.get('rag_cluster_id')})
        if clusters:
            c_placeholders = ', '.join(['%s'] * len(clusters))
            c_rows = execute_query(
                f"SELECT id, student_id, rag_cluster_id, description, status FROM complaints WHERE rag_cluster_id IN ({c_placeholders})",
                tuple(clusters),
                fetchall=True
            ) or []
            for r in c_rows:
                cid = r['rag_cluster_id']
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
                cluster_members.setdefault(cid, []).append(r)

        for comp in complaints:
            comp['comments'] = comments_by_cid.get(comp['id'], [])
            cid = comp.get('rag_cluster_id')
            comp['cluster_size'] = cluster_counts.get(cid, 1)
            comp['cluster_members'] = cluster_members.get(cid, [])
            sla_h = 6 if comp.get('priority_level') == 'High Priority' else 24
            risk_lvl, rem_str = evaluate_sla_breach_risk(comp.get('created_at'), sla_h, comp.get('status'))
            comp['sla_breach_risk'] = risk_lvl
            comp['sla_remaining_text'] = rem_str

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

    # Incident Outbreak detection for officer (any cluster with >= 2 active complaints)
    active_clusters_list = [
        {'cluster_id': cid, 'count': cnt, 'members': cluster_members.get(cid, [])}
        for cid, cnt in cluster_counts.items() if cnt >= 2
    ]

    return render_template(
        'assigned_dashboard.html',
        complaints=complaints,
        stats=stats,
        backend=get_backend_name(),
        current_faculty=current_faculty,
        selected_status=selected_status,
        active_clusters=active_clusters_list
    )

@app.route('/faculty/update-status/<int:complaint_id>', methods=['POST'])
@faculty_required
def faculty_update_status(complaint_id):
    raw_status = request.form.get('status', '').strip()
    completion_note = request.form.get('completion_note', '').strip()
    active_faculty_id = session.get('faculty_id')

    # Normalize status for task management (completed, waited, in-progress)
    if raw_status.lower() in ('completed', 'completed - awaiting feedback'):
        status = 'Completed - Awaiting Feedback'
    elif raw_status.lower() in ('waited', 'waiting', 'pending'):
        status = 'Pending'
    elif raw_status.lower() in ('in progress', 'inprogress', 'progress', 'other'):
        status = 'In Progress'
    else:
        status = raw_status

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

    if complaint['status'] in ('Closed', 'Resolved'):
        flash('This grievance has already reached the verified closed stage by Administration.', 'warning')
        return redirect(url_for('assigned_dashboard'))

    teacher_name = session.get('faculty_name', 'Assigned Teacher')
    if status == 'Completed - Awaiting Feedback':
        execute_query(
            'UPDATE complaints SET status = %s, completion_note = %s WHERE id = %s',
            (status, completion_note[:500] if completion_note else 'Work completed by assigned teacher. Awaiting student feedback.', complaint_id),
            commit=True
        )
        update_message = f"Assigned teacher marked the task as Completed: {completion_note[:300] if completion_note else 'Resolution completed, awaiting student feedback.'}"
        notification_message = f"Grievance #{complaint_id} marked as Completed. Review the resolution summary and share your feedback."
    elif status == 'Pending':
        execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)
        update_message = f"Assigned teacher updated the task status to: Waited (Pending Action).{f' Remarks: {completion_note[:300]}' if completion_note else ''}"
        notification_message = f"Assigned teacher updated grievance #{complaint_id} to Waited (Pending Action)."
    else:
        execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)
        update_message = f"Assigned teacher updated the task status to: In Progress.{f' Remarks: {completion_note[:300]}' if completion_note else ''}"
        notification_message = f"Assigned teacher updated grievance #{complaint_id} to In Progress."

    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, teacher_name, 'Faculty', update_message),
        commit=True
    )
    add_notification(
        notification_message,
        complaint_id,
        user_id=complaint['student_id']
    )
    flash(f"Task status updated to '{status}' successfully.", 'success')
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
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, author_name, 'Faculty', message[:500]),
            commit=True
        )
        add_notification(
            f'Assigned officer {author_name} sent you an update on grievance #{complaint_id}.',
            complaint_id,
            user_id=complaint['student_id']
        )
        flash('Message sent to the student.', 'success')
    return redirect(url_for('assigned_dashboard'))

@app.route('/faculty/mass-reply', methods=['POST'])
@faculty_required
def faculty_mass_reply():
    cluster_id = request.form.get('cluster_id', '').strip()
    status = request.form.get('status', 'Completed - Awaiting Feedback').strip()
    reply_message = request.form.get('reply_message', '').strip()
    active_faculty_id = session.get('faculty_id')
    faculty_name = session.get('faculty_name', 'Assigned Officer')

    if not cluster_id or not reply_message:
        flash('Cluster ID and reply message are required for mass redressal.', 'warning')
        return redirect(url_for('assigned_dashboard'))

    if status.lower() in ('completed', 'completed - awaiting feedback'):
        new_status = 'Completed - Awaiting Feedback'
    elif status.lower() in ('in progress', 'progress'):
        new_status = 'In Progress'
    else:
        new_status = 'Pending'

    complaints = execute_query(
        "SELECT id, student_id FROM complaints WHERE rag_cluster_id = %s AND status NOT IN ('Closed', 'Resolved')",
        (cluster_id,),
        fetchall=True
    ) or []

    if not complaints:
        flash(f'No active complaints found in Cluster {cluster_id}.', 'warning')
        return redirect(url_for('assigned_dashboard'))

    count = 0
    for comp in complaints:
        cid = comp['id']
        sid = comp['student_id']
        if new_status == 'Completed - Awaiting Feedback':
            execute_query(
                'UPDATE complaints SET status = %s, completion_note = %s WHERE id = %s',
                (new_status, reply_message[:500], cid),
                commit=True
            )
        else:
            execute_query('UPDATE complaints SET status = %s WHERE id = %s', (new_status, cid), commit=True)

        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (cid, f"{faculty_name} (Mass Redressal)", 'Faculty', f"[Group Redressal]: {reply_message[:500]}"),
            commit=True
        )
        add_notification(
            f"Update on your grievance #{cid} via Group Redressal ({cluster_id}): {reply_message[:100]}",
            cid,
            user_id=sid
        )
        count += 1

    flash(f"⚡ Success! Mass resolution broadcast and status '{new_status}' applied simultaneously to {count} students in Cluster {cluster_id}!", 'success')
    return redirect(url_for('assigned_dashboard'))


# ==============================================================================
# MODULE 3: ADMIN VIEW (CENTRAL CAMPUS GOVERNANCE, RAG AUDIT & OUTBREAKS)
# ==============================================================================

@app.route('/admin')
@admin_required
def admin():
    selected_dept = request.args.get('dept', 'all').strip()
    selected_status = request.args.get('status', 'all').strip()

    faculty_list = execute_query('SELECT id, name, department, email FROM faculty ORDER BY department ASC, name ASC', fetchall=True) or []
    departments = sorted(list({f['department'] for f in faculty_list if f.get('department')}))

    faculty_workloads = execute_query('''
        SELECT f.id, f.name, f.department, f.email,
               COUNT(CASE WHEN c.status NOT IN ('Closed', 'Resolved', 'Rejected (Policy Violation)') THEN c.id END) AS active_load,
               COUNT(c.id) AS total_handled
        FROM faculty f
        LEFT JOIN complaints c ON f.id = c.assigned_faculty_id
        GROUP BY f.id, f.name, f.department, f.email
        ORDER BY f.department ASC, active_load DESC
    ''', fetchall=True) or []

    internal_notes = execute_query('''
        SELECT cm.id, cm.complaint_id, cm.author_name, cm.message, cm.created_at,
               c.status, c.predicted_department
        FROM complaint_comments cm
        INNER JOIN complaints c ON c.id = cm.complaint_id
        WHERE cm.author_role = %s
        ORDER BY cm.id DESC
        LIMIT 8
    ''', ('Faculty (To Admin)',), fetchall=True) or []

    query = '''
        SELECT c.id, u.roll_number, u.name AS student_name, c.description, 
               c.predicted_department, c.priority_level, c.status, c.created_at, 
               c.completion_note, c.student_feedback, c.feedback_sentiment, c.assigned_faculty_id,
               c.rag_acceptability, c.rag_policy_score, c.rag_policy_citation, c.rag_policy_advisory,
               c.rag_cluster_id, c.sla_deadline, c.sla_breach_risk, c.urgency_score, c.topic_tags, c.escalation_level,
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

    query += " ORDER BY CASE c.priority_level WHEN 'High Priority' THEN 0 ELSE 1 END, c.id DESC"
    complaints = execute_query(query, tuple(params), fetchall=True) or []

    cluster_counts = {}
    cluster_members = {}
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
        
        clusters = list({c['rag_cluster_id'] for c in complaints if c.get('rag_cluster_id')})
        if clusters:
            c_placeholders = ', '.join(['%s'] * len(clusters))
            c_rows = execute_query(
                f"SELECT id, student_id, rag_cluster_id, description, status FROM complaints WHERE rag_cluster_id IN ({c_placeholders})",
                tuple(clusters),
                fetchall=True
            ) or []
            for r in c_rows:
                cid = r['rag_cluster_id']
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
                cluster_members.setdefault(cid, []).append(r)

        for comp in complaints:
            comp['comments'] = comments_by_cid.get(comp['id'], [])
            cid = comp.get('rag_cluster_id')
            comp['cluster_size'] = cluster_counts.get(cid, 1)
            comp['cluster_members'] = cluster_members.get(cid, [])
            sla_h = 6 if comp.get('priority_level') == 'High Priority' else 24
            risk_lvl, rem_str = evaluate_sla_breach_risk(comp.get('created_at'), sla_h, comp.get('status'))
            comp['sla_breach_risk'] = risk_lvl
            comp['sla_remaining_text'] = rem_str

    # Campus-wide statistics
    all_complaints = execute_query("SELECT priority_level, status, rag_acceptability, rag_cluster_id, urgency_score FROM complaints", fetchall=True) or []
    total_count = len(all_complaints)
    acceptable_count = sum(1 for x in all_complaints if x.get('rag_acceptability') == 'Acceptable')
    policy_violations_count = sum(1 for x in all_complaints if x.get('rag_acceptability') == 'Policy Violation')
    clusters_set = {x.get('rag_cluster_id') for x in all_complaints if x.get('rag_cluster_id')}

    stats = {
        'total': total_count,
        'high_priority': sum(x['priority_level'] == 'High Priority' for x in all_complaints),
        'pending': sum(x['status'] == 'Pending' for x in all_complaints),
        'in_progress': sum(x['status'] == 'In Progress' for x in all_complaints),
        'awaiting_feedback': sum(x['status'] == 'Completed - Awaiting Feedback' for x in all_complaints),
        'resolved': sum(x['status'] in ('Closed', 'Resolved') for x in all_complaints),
        'policy_pass_rate': round((acceptable_count / total_count * 100), 1) if total_count > 0 else 100.0,
        'policy_violations': policy_violations_count,
        'active_clusters': len(clusters_set),
        'avg_hazard': round(sum(x.get('urgency_score') or 15 for x in all_complaints) / total_count, 1) if total_count > 0 else 15.0
    }

    active_clusters_list = [
        {'cluster_id': cid, 'count': cnt, 'members': cluster_members.get(cid, [])}
        for cid, cnt in cluster_counts.items() if cnt >= 2
    ]

    return render_template(
        'admin_dashboard.html',
        complaints=complaints,
        stats=stats,
        backend=get_backend_name(),
        faculty_list=faculty_list,
        faculty_workloads=faculty_workloads,
        internal_notes=internal_notes,
        departments=departments,
        selected_dept=selected_dept,
        selected_status=selected_status,
        admin_name=ADMIN_USERNAME,
        active_clusters=active_clusters_list,
        active_tab=request.args.get('tab', 'triage')
    )

@app.route('/admin/task-status/<int:complaint_id>', methods=['POST'])
@admin_required
def admin_update_task_status(complaint_id):
    raw_status = request.form.get('status', '').strip()
    completion_note = request.form.get('completion_note', '').strip()

    if raw_status.lower() in ('completed', 'completed - awaiting feedback'):
        status = 'Completed - Awaiting Feedback'
    elif raw_status.lower() in ('waited', 'waiting', 'pending'):
        status = 'Pending'
    elif raw_status.lower() in ('in progress', 'inprogress', 'progress', 'other'):
        status = 'In Progress'
    elif raw_status.lower() in ('closed', 'resolved'):
        status = 'Closed'
    else:
        status = raw_status

    complaint = execute_query('SELECT student_id, assigned_faculty_id, status FROM complaints WHERE id = %s', (complaint_id,), fetchone=True)
    if not complaint:
        flash('Complaint not found.', 'danger')
        return redirect(request.referrer or url_for('admin'))

    if status == 'Closed':
        return update_status(complaint_id)

    if status == 'Completed - Awaiting Feedback':
        execute_query(
            'UPDATE complaints SET status = %s, completion_note = %s WHERE id = %s',
            (status, completion_note[:500] if completion_note else 'Task marked completed. Awaiting student feedback.', complaint_id),
            commit=True
        )
        update_msg = f"Task marked as Completed: {completion_note[:300] if completion_note else 'Resolution completed, awaiting student feedback.'}"
        notif_msg = f"Grievance #{complaint_id} task marked as Completed. Please review and provide feedback."
    elif status == 'Pending':
        execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)
        update_msg = f"Task status updated to: Waited (Pending Action).{f' Remarks: {completion_note[:300]}' if completion_note else ''}"
        notif_msg = f"Grievance #{complaint_id} status updated to Waited (Pending Action)."
    elif status == 'In Progress':
        execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)
        update_msg = f"Task status updated to: In Progress.{f' Remarks: {completion_note[:300]}' if completion_note else ''}"
        notif_msg = f"Grievance #{complaint_id} status updated to In Progress."
    else:
        execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)
        update_msg = f"Task status updated to: {status}"
        notif_msg = f"Grievance #{complaint_id} status updated to {status}."

    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, f"Admin ({ADMIN_USERNAME})", 'Admin', update_msg),
        commit=True
    )
    add_notification(notif_msg, complaint_id, user_id=complaint['student_id'], faculty_id=complaint.get('assigned_faculty_id'))
    flash(f"Task status for Grievance #{complaint_id} updated to '{status}'.", 'success')
    return redirect(request.referrer or url_for('admin', tab='teachers'))

@app.route('/admin/update-status/<int:complaint_id>', methods=['POST'])
@admin_required
def update_status(complaint_id):
    status = request.form.get('status', '').strip()
    if status != 'Closed':
        flash('Work progress belongs to the assigned officer. Administration can perform verified final closure only.', 'warning')
        return redirect(request.referrer or url_for('admin'))

    complaint = execute_query(
        'SELECT student_id, assigned_faculty_id, student_feedback, feedback_sentiment, status FROM complaints WHERE id = %s',
        (complaint_id,),
        fetchone=True
    )
    if not complaint:
        return redirect(request.referrer or url_for('admin'))
    if complaint['status'] != 'Completed - Awaiting Feedback':
        flash('Only a completed grievance awaiting student feedback can be finalized by administration.', 'warning')
        return redirect(request.referrer or url_for('admin'))

    if not complaint.get('student_feedback') or not complaint['student_feedback'].strip():
        flash(f'Cannot close Grievance #{complaint_id}: Student has not submitted resolution feedback yet.', 'warning')
        return redirect(request.referrer or url_for('admin'))

    sentiment = complaint.get('feedback_sentiment')
    if not sentiment:
        sentiment, _ = predict_feedback_sentiment(complaint['student_feedback'])
        execute_query('UPDATE complaints SET feedback_sentiment = %s WHERE id = %s', (sentiment, complaint_id), commit=True)

    if sentiment != 'Positive':
        flash(f'Cannot close Grievance #{complaint_id}: student feedback must be positive before final closure.', 'danger')
        return redirect(request.referrer or url_for('admin'))

    execute_query('UPDATE complaints SET status = %s, closed_at = CURRENT_TIMESTAMP WHERE id = %s', ('Closed', complaint_id), commit=True)
    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, f'Admin ({ADMIN_USERNAME})', 'Admin', f"Administrative Closure: Case verified and permanently closed following positive student feedback ({sentiment})."),
        commit=True
    )
    add_notification(f'Grievance #{complaint_id} has been verified and permanently closed by Administration.', complaint_id, user_id=complaint['student_id'], faculty_id=complaint.get('assigned_faculty_id'))
    flash(f'Grievance #{complaint_id} verified and permanently closed following positive student feedback.', 'success')
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
    return redirect(request.referrer or url_for('admin'))

@app.route('/admin/mass-reply', methods=['POST'])
@admin_required
def admin_mass_reply():
    cluster_id = request.form.get('cluster_id', '').strip()
    status = request.form.get('status', 'Completed - Awaiting Feedback').strip()
    reply_message = request.form.get('reply_message', '').strip()

    if not cluster_id or not reply_message:
        flash('Cluster ID and reply message are required.', 'warning')
        return redirect(url_for('admin'))

    if status.lower() in ('completed', 'completed - awaiting feedback'):
        new_status = 'Completed - Awaiting Feedback'
    elif status.lower() in ('in progress', 'progress'):
        new_status = 'In Progress'
    else:
        new_status = 'Pending'

    complaints = execute_query(
        "SELECT id, student_id FROM complaints WHERE rag_cluster_id = %s AND status NOT IN ('Closed', 'Resolved')",
        (cluster_id,),
        fetchall=True
    ) or []

    if not complaints:
        flash(f'No active complaints found in Cluster {cluster_id}.', 'warning')
        return redirect(url_for('admin'))

    count = 0
    for comp in complaints:
        cid = comp['id']
        sid = comp['student_id']
        if new_status == 'Completed - Awaiting Feedback':
            execute_query(
                'UPDATE complaints SET status = %s, completion_note = %s WHERE id = %s',
                (new_status, reply_message[:500], cid),
                commit=True
            )
        else:
            execute_query('UPDATE complaints SET status = %s WHERE id = %s', (new_status, cid), commit=True)

        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (cid, f"Admin ({ADMIN_USERNAME}) [Mass Broadcast]", 'Admin', f"[Group Redressal]: {reply_message[:500]}"),
            commit=True
        )
        add_notification(
            f"Central Administrative Update on grievance #{cid} (Cluster {cluster_id}): {reply_message[:100]}",
            cid,
            user_id=sid
        )
        count += 1

    flash(f"⚡ Success! Central administrative mass resolution applied to {count} students in Cluster {cluster_id}!", 'success')
    return redirect(url_for('admin'))

@app.route('/admin/reassign/<int:complaint_id>', methods=['POST'])
@admin_required
def reassign_complaint(complaint_id):
    faculty_id = request.form.get('faculty_id')
    complaint = execute_query('SELECT id, student_id, predicted_department, status, student_feedback, feedback_sentiment, completion_note FROM complaints WHERE id = %s', (complaint_id,), fetchone=True)
    faculty = execute_query('SELECT id, name, department FROM faculty WHERE id = %s', (faculty_id,), fetchone=True)
    if complaint and faculty and complaint['status'] not in ('Closed', 'Resolved'):
        reopening_negative_case = (
            complaint['status'] == 'Completed - Awaiting Feedback'
            and complaint.get('feedback_sentiment') == 'Negative'
        )
        if complaint['status'] == 'Completed - Awaiting Feedback' and not reopening_negative_case:
            flash('Only a grievance with negative feedback can be reopened from the completed state.', 'warning')
            return redirect(request.referrer or url_for('admin'))

        execute_query(
            '''UPDATE complaints
               SET assigned_faculty_id = %s, predicted_department = %s, status = %s,
                   completion_note = %s, student_feedback = %s, feedback_sentiment = %s
               WHERE id = %s''',
            (
                faculty['id'],
                faculty['department'],
                'In Progress' if reopening_negative_case else complaint['status'],
                None if reopening_negative_case else complaint.get('completion_note'),
                None if reopening_negative_case else complaint.get('student_feedback'),
                None if reopening_negative_case else complaint.get('feedback_sentiment'),
                complaint_id
            ),
            commit=True
        )
        reassignment_message = f"Re-routed by Central Admin to {faculty['name']} ({faculty['department']})."
        if reopening_negative_case:
            reassignment_message = f"Reopened after negative student feedback and assigned to {faculty['name']} ({faculty['department']})."
            if complaint.get('completion_note'):
                reassignment_message += f" Previous resolution note: {complaint['completion_note']}"
        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, f'Admin ({ADMIN_USERNAME})', 'Admin', reassignment_message[:500]),
            commit=True
        )
        add_notification(
            f"Grievance #{complaint_id} has been re-assigned to {faculty['name']} ({faculty['department']}).",
            complaint_id,
            user_id=complaint['student_id'],
            faculty_id=faculty['id']
        )
    return redirect(request.referrer or url_for('admin'))

@app.route('/admin/export-csv')
@admin_required
def export_csv():
    complaints = execute_query('''
        SELECT c.id, u.roll_number, u.name AS student_name, c.description,
               c.predicted_department, c.priority_level, c.status,
               c.rag_acceptability, c.rag_policy_score, c.rag_policy_citation,
               c.rag_cluster_id, c.sla_deadline, c.sla_breach_risk, c.urgency_score,
               c.feedback_sentiment, c.created_at, c.closed_at,
               f.name AS faculty_name
        FROM complaints c
        LEFT JOIN users u ON c.student_id = u.id
        LEFT JOIN faculty f ON c.assigned_faculty_id = f.id
        ORDER BY c.id DESC
    ''', fetchall=True) or []

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Ticket ID', 'Roll Number', 'Student Name', 'Department', 'Priority',
        'Hazard Score', 'RAG Acceptability', 'Policy Citation', 'Compliance Score',
        'Cluster ID', 'SLA Deadline', 'SLA Risk', 'Status', 'Feedback Sentiment',
        'Assigned Faculty', 'Created At', 'Closed At', 'Description'
    ])
    for c in complaints:
        writer.writerow([
            c.get('id'),
            c.get('roll_number'),
            c.get('student_name'),
            c.get('predicted_department'),
            c.get('priority_level'),
            c.get('urgency_score') or 15,
            c.get('rag_acceptability') or 'Acceptable',
            c.get('rag_policy_citation') or '',
            c.get('rag_policy_score') or 85.0,
            c.get('rag_cluster_id') or '',
            c.get('sla_deadline') or '',
            c.get('sla_breach_risk') or 'Low',
            c.get('status'),
            c.get('feedback_sentiment') or 'Pending',
            c.get('faculty_name') or 'Unassigned',
            c.get('created_at'),
            c.get('closed_at') or 'Open',
            (c.get('description') or '').replace('\n', ' ')
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=campus_grievances_dossier.csv'}
    )


# ==============================================================================
# MODULE 4: RAG REST APIs (LIVE AS-YOU-TYPE PRECHECK & AI DRAFTING)
# ==============================================================================

@app.route('/api/rag/precheck', methods=['POST'])
def api_rag_precheck():
    data = request.get_json(silent=True) or {}
    description = data.get('description', '').strip()
    if not description or len(description) < 6:
        return jsonify({'valid': False, 'message': 'Description too short'})

    dept = predict_department(description)
    priority = get_priority_level(description)
    rag_eval = evaluate_complaint_acceptability(description, dept)
    hazard_score, hazard_reasons = calculate_hazard_risk_score(description)
    sla_h, _ = calculate_dynamic_sla(priority, dept)

    all_complaints = execute_query('SELECT id, description, predicted_department, priority_level, status FROM complaints LIMIT 200', fetchall=True) or []
    similar_list, cluster_id, is_duplicate = find_similar_complaints(description, all_complaints)

    return jsonify({
        'valid': True,
        'status': rag_eval['status'],
        'policy_score': rag_eval['policy_score'],
        'citation': rag_eval['citation'],
        'verdict_title': rag_eval['verdict_title'],
        'advisory': rag_eval['advisory_response'],
        'violation_reasons': rag_eval.get('violation_reasons', []),
        'predicted_department': dept,
        'priority': priority,
        'hazard_score': hazard_score,
        'hazard_reasons': hazard_reasons,
        'sla_hours': sla_h,
        'similar_count': len(similar_list),
        'cluster_id': cluster_id or f"CLUST-{dept[:4].upper()}-NEW",
        'is_duplicate': is_duplicate,
        'similar_preview': similar_list[0]['description'] if similar_list else None
    })

@app.route('/api/rag/draft-resolution', methods=['GET', 'POST'])
def api_rag_draft_resolution():
    cid = request.args.get('id')
    if not cid and request.is_json:
        data = request.get_json(silent=True) or {}
        cid = data.get('id') or data.get('complaint_id')
    if not cid:
        cid = request.form.get('id') or request.form.get('complaint_id')
    if not cid:
        return jsonify({'error': 'Missing id'}), 400
    comp = execute_query('SELECT id, description, predicted_department, rag_cluster_id FROM complaints WHERE id = %s', (cid,), fetchone=True)
    if not comp:
        return jsonify({'error': 'Not found'}), 404

    cluster_count = 1
    if comp.get('rag_cluster_id'):
        cnt_row = execute_query('SELECT COUNT(*) as c FROM complaints WHERE rag_cluster_id = %s', (comp['rag_cluster_id'],), fetchone=True)
        if cnt_row:
            cluster_count = cnt_row['c']

    officer_name = session.get('faculty_name') or session.get('admin_user') or "Assigned Authority"
    draft = generate_rag_resolution_draft(
        comp['description'],
        comp['predicted_department'],
        faculty_name=officer_name,
        similar_count=cluster_count
    )
    return jsonify({
        'success': True,
        'draft': draft,
        'cluster_id': comp.get('rag_cluster_id'),
        'cluster_size': cluster_count
    })

@app.route('/api/rag/policies')
def api_rag_policies():
    return jsonify({'policies': CAMPUS_POLICIES})

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
