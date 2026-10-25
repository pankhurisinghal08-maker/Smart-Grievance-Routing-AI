def get_priority_level(text):
    # Words that mean an emergency
    urgent_keywords = ["emergency", "harassment", "fire", "ragging", "medical", "safety", "danger", "urgent"]
    
    # Change text to small letters to check easily
    lower_text = text.lower()
    
    # If any emergency word is found, tag as High Priority
    for word in urgent_keywords:
        if word in lower_text:
            return "High Priority"
            
    return "Normal"