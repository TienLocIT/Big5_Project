const preloading = document.getElementById("loadingio");
const center = document.querySelector(".center");
const login = document.getElementById("center__btn");
const username = document.getElementById("username");
const password = document.getElementById("password");
import localHostEnv from "./config.js";
login.onclick = async () => {
        center.style.display = "none";
        preloading.style.display = "block";
        const data = {
                "username": username.value,
                "password": password.value
        }
        const response = await fetch("http://"+localHostEnv+"/login", {
                method: "POST",
                mode: "cors",
                cache: "no-cache",
                credentials: "same-origin",
                headers: {
                        "Content-type": "application/json"
                },
                redirect: "follow",
                referrerPolicy: "no-referrer",
                body: JSON.stringify(data)
        })
        const dataResponse = await response.json();
        if (dataResponse.error == "No") {
                toast({
                        Title: "Success",
                        Message: "Login successfully",
                        Type: "success",
                        Duration: 2000
                })
                window.location.href = "http://"+localHostEnv+"/admin";
        }
        else {
                center.style.display = "flex";
                preloading.style.display = "none";
                toast({
                        Title: "Error",
                        Message: dataResponse.error,
                        Type: "error",
                        Duration: 2000
                })
        }
}
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
