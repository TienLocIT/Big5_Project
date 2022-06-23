userSendModalBody.onclick = (event) => {
    event.stopPropagation();
}
user__choooseOption_cancel.onclick = () => {
    bodySendModal.innerHTML = "";
    usermodel.style.display = "none";
}