from flask import Flask, render_template, request
import mysql.connector
from model.rules import get_priority_level
from model.nlp_model import predict_department

app = Flask(__name__)

# This function connects your Python code to your XAMPP database
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",      # XAMPP default password is empty
        database="grievance_system" 
    )

@app.route('/')
def home():
    return render_template('submit_form.html')

@app.route('/submit', methods=['POST'])
def submit():
    roll_number = request.form['roll_number']
    description = request.form['description']
    
    # 1. Ask the AI to guess the department
    department = predict_department(description)
    
    # 2. Check if the issue is an emergency
    priority = get_priority_level(description)
    
    # 3. Connect to the database
    db = get_db()
    cursor = db.cursor()
    
    # 4. Find the student in the database using their roll number
    cursor.execute("SELECT id FROM users WHERE roll_number = %s", (roll_number,))
    student = cursor.fetchone()
    
    # 5. Save the complaint if the student exists
    if student:
        student_id = student[0]
        sql = "INSERT INTO complaints (student_id, description, predicted_department, priority_level) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (student_id, description, department, priority))
        db.commit()
        result = f"Success! AI routed this to: {department}. Priority: {priority}"
    else:
        result = "Error: Roll number not found in the system. Please register first."
        
    cursor.close()
    db.close()
    
    return result

@app.route('/admin')
def admin():
    return render_template('admin_dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)