from flask import Flask, render_template, request
import mysql.connector
from model.rules import get_priority_level
from model.nlp_model import predict_department

app = Flask(__name__)

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="", # Type your MySQL password here if you created one
        database="grievance_system" 
    )

@app.route('/')
def home():
    return render_template('submit_form.html')

@app.route('/submit', methods=['POST'])
def submit():
    roll_number = request.form['roll_number']
    description = request.form['description']
    
    # Use AI and Rules
    department = predict_department(description)
    priority = get_priority_level(description)
    
    # Save to database
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("SELECT id FROM users WHERE roll_number = %s", (roll_number,))
    student = cursor.fetchone()
    
    if student:
        student_id = student[0]
        sql = "INSERT INTO complaints (student_id, description, predicted_department, priority_level) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (student_id, description, department, priority))
        db.commit()
        result = f"Success! AI routed this to: {department}. Priority: {priority}"
    else:
        result = "Error: Roll number not found in the database. Use 123456 to test."
        
    cursor.close()
    db.close()
    
    return result

@app.route('/admin')
def admin():
    db = get_db()
    # dictionary=True makes it easy to read data by column names
    cursor = db.cursor(dictionary=True)
    
    # Get all complaints and match them with student roll numbers
    query = """
        SELECT c.id, u.roll_number, c.description, c.predicted_department, c.priority_level, c.status 
        FROM complaints c
        JOIN users u ON c.student_id = u.id
        ORDER BY c.id DESC
    """
    cursor.execute(query)
    complaints = cursor.fetchall()
    
    cursor.close()
    db.close()
    
    return render_template('admin_dashboard.html', complaints=complaints)

if __name__ == '__main__':
    app.run(debug=True)