from flask import Flask, render_template, request

# Import the rules you just wrote
from model.rules import get_priority_level

app = Flask(__name__)

# Show the student form on the homepage
@app.route('/')
def home():
    return render_template('submit_form.html')

# Catch the data when the student clicks Submit
@app.route('/submit', methods=['POST'])
def submit():
    roll_number = request.form['roll_number']
    description = request.form['description']
    
    # Check if the complaint is an emergency
    priority = get_priority_level(description)
    
    # Print the result in your terminal to test if it works
    print("New Complaint Received!")
    print(f"Roll Number: {roll_number}")
    print(f"Text: {description}")
    print(f"Priority: {priority}")
    
    return f"Success! Your complaint is saved. System Priority: {priority}"

# Show the admin dashboard
@app.route('/admin')
def admin():
    return render_template('admin_dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)