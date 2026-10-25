from flask import Flask

# Create the web server
app = Flask(__name__)

# Set up the main homepage
@app.route('/')
def home():
    return "Hello! The Student Grievance System server is running."

# Turn on the server
if __name__ == '__main__':
    app.run(debug=True)