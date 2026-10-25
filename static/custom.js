const descriptionBox = document.querySelector('textarea[name="description"]');
const maxChars = 500;

if (descriptionBox) {
    descriptionBox.addEventListener('input', function() {
        const currentLength = descriptionBox.value.length;
        const remaining = maxChars - currentLength;
        const counterText = document.getElementById('char-count');
        if (counterText) {
            counterText.textContent = remaining + " character(s) left";
        }
    });
}

const form = document.querySelector('form');
if (form) {
    form.addEventListener('submit', function(event) {
        if (descriptionBox.value.trim() === "") {
            alert("Please describe your problem before submitting.");
            event.preventDefault(); 
        }
    });
}