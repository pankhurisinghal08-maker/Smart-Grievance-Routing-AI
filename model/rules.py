# This function checks if a complaint is an emergency
def get_priority_level(text):
    # A list of serious words that need immediate attention
    urgent_keywords = ["emergency", "harassment", "fire", "ragging", "medical", "safety", "danger", "urgent"]
    
    # Convert the student's text to lowercase so we don't miss capitalized words
    lower_text = text.lower()
    
    # Check if any urgent word is inside the student's text
    for word in urgent_keywords:
        if word in lower_text:
            return "High Priority"
            
    # If no urgent words are found, it is a normal issue
    return "Normal"