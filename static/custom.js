document.addEventListener('DOMContentLoaded', function() {
    const descriptionBox = document.querySelector('textarea[name="description"]');
    const rollBox = document.querySelector('input[name="roll_number"]');
    const form = document.querySelector('form.form-container');
    const counterText = document.getElementById('char-count');
    const maxChars = 500;

    function updateCounter() {
        if (!descriptionBox || !counterText) return;
        const currentLength = descriptionBox.value.length;
        const remaining = maxChars - currentLength;
        
        counterText.textContent = remaining + " character(s) left";
        if (remaining < 0) {
            counterText.classList.remove('text-muted');
            counterText.classList.add('text-danger', 'fw-bold');
        } else {
            counterText.classList.remove('text-danger', 'fw-bold');
            counterText.classList.add('text-muted');
        }
    }

    if (descriptionBox) {
        descriptionBox.addEventListener('input', updateCounter);
        // Initial count check on page load
        updateCounter();
    }

    if (form) {
        form.addEventListener('submit', function(event) {
            const desc = descriptionBox ? descriptionBox.value.trim() : '';
            const roll = rollBox ? rollBox.value.trim() : '';

            if (roll === "") {
                alert("Please enter your roll number.");
                if (rollBox) rollBox.focus();
                event.preventDefault();
                return;
            }

            if (desc === "") {
                alert("Please describe your problem before submitting.");
                if (descriptionBox) descriptionBox.focus();
                event.preventDefault();
                return;
            }

            // Provide visual feedback during AI inference
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Analyzing with AI...';
            }
        });
    }
});
