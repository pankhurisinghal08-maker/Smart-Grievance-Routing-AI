// Select the text box and set the maximum length
const descriptionBox = document.querySelector('textarea[name="description"]');
const maxChars = 500;

// Update the counter every time the student types a letter
if (descriptionBox) {
    descriptionBox.addEventListener('input', function() {
        const currentLength = descriptionBox.value.length;
        const remaining = maxChars - currentLength;
        
        // Find the counter element on the page and update the number
        const counterText = document.getElementById('char-count');
        if (counterText) {
            counterText.textContent = remaining + " character(s) left";
        }
    });
}

// Show a popup warning if the text box is empty when clicking submit
const form = document.querySelector('form');
if (form) {
    form.addEventListener('submit', function(event) {
        if (descriptionBox.value.trim() === "") {
            alert("Please describe your problem before submitting.");
            event.preventDefault(); // Stop the form from sending data to the server
        }
    });
}