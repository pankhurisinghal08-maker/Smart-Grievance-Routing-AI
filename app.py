from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps
import os
from model.rules import get_priority_level
from model.nlp_model import predict_department
from model.sentiment_model import predict_feedback_sentiment
from database.db import execute_query, get_backend_name

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or os.urandom(32)

# Admin access is disabled until explicit credentials are configured.
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', '')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', '')
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
            SELECT c.id, c.description, c.predicted_department, c.priority_level,
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

    # Intelligently assign to the faculty member in that department with the lowest active workload
    faculty = execute_query('''
        SELECT f.id, f.name, f.department, f.email, COUNT(c.id) AS active_load
        FROM faculty f
        LEFT JOIN complaints c ON f.id = c.assigned_faculty_id AND c.status NOT IN ('Closed', 'Resolved')
        WHERE f.department = %s
        GROUP BY f.id, f.name, f.department, f.email
        ORDER BY active_load ASC, f.id ASC
        LIMIT 1
    ''', (department,), fetchone=True)

    if not faculty:
        faculty = execute_query('''
            SELECT f.id, f.name, f.department, f.email, COUNT(c.id) AS active_load
            FROM faculty f
            LEFT JOIN complaints c ON f.id = c.assigned_faculty_id AND c.status NOT IN ('Closed', 'Resolved')
            GROUP BY f.id, f.name, f.department, f.email
            ORDER BY active_load ASC, f.id ASC
            LIMIT 1
        ''', fetchone=True)
        if faculty:
            department = faculty['department']

    complaint_id = execute_query(
        'INSERT INTO complaints (student_id, description, predicted_department, assigned_faculty_id, priority_level, status) VALUES (%s, %s, %s, %s, %s, %s)',
        (student['id'], description, department, faculty['id'] if faculty else None, priority, 'Pending'),
        commit=True
    )

    execute_query(
        'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
        (complaint_id, student['name'], 'Student', 'Grievance submitted for review.'),
        commit=True
    )

    faculty_text = f"{faculty['name']} ({faculty['department']})" if faculty else department
    add_notification(
        f"Grievance #{complaint_id} was routed to {faculty_text}. Priority: {priority}.",
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
               c.predicted_department, c.priority_level, c.status, c.created_at, 
               c.completion_note, c.student_feedback, c.feedback_sentiment, c.assigned_faculty_id, c.closed_at,
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

    return render_template(
        'assigned_dashboard.html',
        complaints=complaints,
        stats=stats,
        backend=get_backend_name(),
        current_faculty=current_faculty,
        selected_status=selected_status
    )

@app.route('/faculty/update-status/<int:complaint_id>', methods=['POST'])
@faculty_required
def faculty_update_status(complaint_id):
    status = request.form.get('status', '').strip()
    completion_note = request.form.get('completion_note', '').strip()
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
    elif status not in VALID_FACULTY_STATUSES:
        flash('Choose a valid progress stage. Final closure is managed by administration.', 'warning')
    elif complaint['status'] in ('Completed - Awaiting Feedback', 'Closed', 'Resolved'):
        flash('This grievance has already reached the final staff progress stage.', 'warning')
    elif complaint['status'] == 'Pending' and status == 'Completed - Awaiting Feedback':
        flash('Move the grievance to In Progress before marking the work complete.', 'warning')
    elif complaint['status'] == 'In Progress' and status == 'Pending':
        flash('An in-progress grievance cannot move back to Pending.', 'warning')
    elif status == 'Completed - Awaiting Feedback' and not completion_note:
        flash('Add a short resolution summary before marking the work complete.', 'warning')
    elif status != complaint['status']:
        if status == 'Completed - Awaiting Feedback':
            execute_query(
                'UPDATE complaints SET status = %s, completion_note = %s WHERE id = %s',
                (status, completion_note[:500], complaint_id),
                commit=True
            )
        else:
            execute_query('UPDATE complaints SET status = %s WHERE id = %s', (status, complaint_id), commit=True)

        if status == 'Completed - Awaiting Feedback':
            update_message = 'The assigned officer marked the work complete and requested your resolution feedback.'
            notification_message = f'Grievance #{complaint_id} is marked complete. Review the resolution summary and share your feedback.'
        else:
            update_message = f'Assigned officer updated the work status to: {status}'
            notification_message = f'The assigned officer updated grievance #{complaint_id} to {status}.'

        execute_query(
            'INSERT INTO complaint_comments (complaint_id, author_name, author_role, message) VALUES (%s, %s, %s, %s)',
            (complaint_id, 'System', 'System', update_message),
            commit=True
        )
        add_notification(
            notification_message,
            complaint_id,
            user_id=complaint['student_id']
        )
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
               COUNT(CASE WHEN c.status NOT IN ('Closed', 'Resolved') THEN c.id END) AS active_load,
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
        admin_name=ADMIN_USERNAME
    )

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

    # Require student feedback before any administrative closure.
    if not complaint.get('student_feedback') or not complaint['student_feedback'].strip():
        flash(f'Cannot close Grievance #{complaint_id}: Student has not submitted resolution feedback yet.', 'warning')
        return redirect(request.referrer or url_for('admin'))

    # Evaluate sentiment if it has not already been recorded.
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

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        threaded=True
    )
