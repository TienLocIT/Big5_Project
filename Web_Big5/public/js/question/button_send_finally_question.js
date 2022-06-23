import localHostEnv from "../config.js";
import toast from "./toast.js"
buttonSendFinally.onclick = async () => {
    // const userBodyModalChooose=document.querySelector(".usersend-modal__body-chooose");
    const loadiniomodal = document.querySelector(".usersend-modal__body .loadingiomodal");
    var idCheckBoxQuestion;
    var arrayTrait = [];
    const options = [...document.getElementsByClassName("user__want--active")];
    const HighLevelOption = document.querySelector(".Highlevel");
    const questionCheckBoxs = [...document.getElementsByName("questionCheckBox")];
    questionCheckBoxs.forEach(data => {
        if (data.checked) {
            idCheckBoxQuestion = data.id;
        }
    })
    arrayTrait.push(HighLevelOption.innerText);
    if (options.length != 0) {
        options.forEach(data => {
            arrayTrait.push(data.innerText);
        })
        // toast({
        //     Title:"Error",
        //     Message:"You need choose trait",
        //     Type:"error",
        //     Duration:2000
        // });
        // return;
    }
    loadiniomodal.style.display = "block";
    const data = {
        "arrayTrait": arrayTrait,
        "id": idCheckBoxQuestion
    }
    loadingio.style.display = "block";
    try {
        const response = await fetch("http://"+localHostEnv+"/admin/sendquestion", {
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
        });
        const dataResponse = await response.json();
        if (dataResponse.erorr == "no") {
            usermodel.style.display = "none";
            loadiniomodal.style.display = "none";
            bodySendModal.innerHTML = "";
            toast({
                Title: "Success",
                Message: "Send Question Success",
                Type: "success",
                Duration: 2000
            })
        }
        else {
            loadiniomodal.style.display = "none";
            toast({
                Title: "Error",
                Message: "Send Question Error",
                Type: "error",
                Duration: 2000
            })
        }
    }
    catch {
        loadiniomodal.style.display = "none";
        toast({
            Title: "Error",
            Message: "Maybe your network is error",
            Type: "error",
            Duration: 3000
        })
    }

}