import toast from "./toast.js";
buttonSend.onclick = () => {
    var checkBoxOption = false;
    var dataCheckBox;
    var checkBoxs = [...document.getElementsByName("questionCheckBox")];
    checkBoxs.forEach(data => {
        if (data.checked) {
            checkBoxOption = true;
            dataCheckBox = data;
        }
    })
    if (checkBoxOption == true) {
        const dataCheckBoxHTML = document.querySelector(".qa" + dataCheckBox.id);
        const childrensDataCheckBoxHTML = dataCheckBoxHTML.children;
        var dataQuestionCheck = {
            "Openness": childrensDataCheckBoxHTML[3].innerHTML,
            "Conscientious": childrensDataCheckBoxHTML[4].innerHTML,
            "Extraversion": childrensDataCheckBoxHTML[5].innerHTML,
            "Agreeable": childrensDataCheckBoxHTML[6].innerHTML,
            "Neuroticism": childrensDataCheckBoxHTML[7].innerHTML
        };
        bodySendModal.innerHTML = bodySendModal.innerHTML + `
        <div class="${dataQuestionCheck.Openness}level leveltrait">
            Openness
        </div>
        <div class="${dataQuestionCheck.Conscientious}level leveltrait">
            Conscientious
        </div>
        <div class="${dataQuestionCheck.Extraversion}level leveltrait">
            Extraversion
        </div>
        <div class="${dataQuestionCheck.Agreeable}level leveltrait">
            Agreeable
        </div>
        <div class="${dataQuestionCheck.Neuroticism}level leveltrait">
            Neuroticism
        </div>
        `
        usermodel.style.display = "flex";
        const mediumLevel = [...document.querySelectorAll(".Mediumlevel")];
        const lowLevel = [...document.querySelectorAll(".Lowlevel")];
        const leveltrait = [...document.getElementsByClassName("leveltrait")];
        mediumLevel.forEach(data => {
            data.onclick = () => {
                mediumLevel.forEach(data => {
                    data.classList.toggle("user__want--active");
                })
            }
        })
        lowLevel.forEach(data => {
            data.onclick = () => {
                if (data.classList.contains("user__want--active")) {
                    lowLevel.forEach(data => {
                        data.classList.remove("user__want--active");
                    })
                }
                else {
                    leveltrait.forEach(data => {
                        if (!data.classList.contains("Highlevel")) {
                            if (!data.classList.contains("user__want--active")) {
                                data.classList.add("user__want--active")
                            }
                        }
                    })
                }
            }
        })
    }
    else {
        toast({
            Title: "Erorr",
            Message: "Bạn phải chọn câu hỏi",
            Type: "error",
            Duration: 2000
        });
    }

}