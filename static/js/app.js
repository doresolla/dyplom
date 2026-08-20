document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-password-toggle]').forEach((button) => {
        const input = document.getElementById(button.dataset.passwordToggle);
        if (!input) return;
        button.addEventListener('click', () => {
            const visible = input.type === 'text';
            input.type = visible ? 'password' : 'text';
            button.textContent = visible ? '◉' : '◎';
            button.setAttribute('aria-label', visible ? 'Показать пароль' : 'Скрыть пароль');
        });
    });
    document.querySelectorAll('[data-confirm]').forEach((form) => {
        form.addEventListener('submit', (event) => {
            if (!window.confirm(form.dataset.confirm)) event.preventDefault();
        });
    });
});
