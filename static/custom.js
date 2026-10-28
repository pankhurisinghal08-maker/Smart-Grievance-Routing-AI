document.addEventListener('DOMContentLoaded', function () {
    // Description character counter
    const descriptionBox = document.querySelector('textarea[name="description"]');
    const counterText = document.getElementById('char-count');
    const maxChars = 500;

    function updateCounter() {
        if (!descriptionBox || !counterText) return;
        const currentLength = descriptionBox.value.length;
        const remaining = maxChars - currentLength;

        counterText.textContent = remaining + " character(s) left";
        if (remaining <= 50) {
            counterText.classList.remove('text-muted');
            counterText.classList.add('text-warning', 'fw-semibold');
        } else if (remaining < 0) {
            counterText.classList.remove('text-muted', 'text-warning');
            counterText.classList.add('text-danger', 'fw-bold');
        } else {
            counterText.classList.remove('text-danger', 'text-warning', 'fw-bold', 'fw-semibold');
            counterText.classList.add('text-muted');
        }
    }

    if (descriptionBox) {
        descriptionBox.addEventListener('input', updateCounter);
        updateCounter();
    }

    // Sample prompt chips click-to-populate (both .sample-chip and .lpu-chip)
    const chips = document.querySelectorAll('.sample-chip, .lpu-chip');
    chips.forEach(chip => {
        chip.addEventListener('click', function () {
            if (descriptionBox) {
                const text = this.getAttribute('data-prompt') || this.innerText.trim();
                descriptionBox.value = text;
                descriptionBox.focus();
                updateCounter();
            }
        });
    });

    // Form submission feedback
    const grievanceForm = document.getElementById('grievanceForm') || document.querySelector('form[action="/submit"]');
    if (grievanceForm) {
        grievanceForm.addEventListener('submit', function (event) {
            const desc = descriptionBox ? descriptionBox.value.trim() : '';

            if (desc === '') {
                alert('Please describe your grievance before submitting.');
                if (descriptionBox) descriptionBox.focus();
                event.preventDefault();
                return;
            }

            const submitBtn = grievanceForm.querySelector('button[type="submit"]');
            if (submitBtn) {
                setTimeout(() => {
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Registering & AI Routing...';
                }, 10);
            }
        });
    }


});
