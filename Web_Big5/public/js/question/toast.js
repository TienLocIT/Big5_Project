function toast({ Title, Message, Type, Duration }) {
    const main = document.getElementById("toast");
    if (main) {
        const toast = document.createElement("div");
        const icons = {
            success: "fas fa-check-circle",
            error: "fas fa-exclamation-circle"
        }
        const icon = icons[Type];
        const duration = (Duration / 1000).toFixed(2);
        const Timeout = Duration + 1000;
        toast.classList.add("toast", `toast--${Type}`);
        toast.innerHTML = `
    <div class="toast__icon">
        <i class="${icon}"></i>
    </div>
    <div class="toast__desc">
        <h3 class="toast__status">${Title}</h3>
        <p class="toast__inforStatus">${Message}</p>
    </div>
    <div class="toast__close">
        <i class="fas fa-times"></i>    
    </div>
    `
        main.appendChild(toast);
        toast.style.animation = `Fadein 0.5s ease,Close 1s ${duration}s forwards`;
        const autoRemoveId = setTimeout(function () {
            main.removeChild(toast)
        }, Timeout)
        main.onclick = function (e) {
            if (e.target.closest(".toast__close")) {
                main.removeChild(toast);
                clearTimeout(autoRemoveId);
            }
        }
    }
}
export default toast;