import localHostEnv from "../config";

var questionCheckBox;
const user__choooseOption_cancel = document.getElementById("user-modal__button-cancel--question");
// const user__choooseOption=[...document.getElementsByClassName("user__chooose--option")];
const userSendModalBody = document.querySelector(".usersend-modal__body");
const userChooseModalBody = document.querySelector(".usersend-modal__body");
const usermodel = document.querySelector(".usersend-modal");
const userChooseModel = document.querySelector(".userchoose-modal");
const buttonSend = document.querySelector("#send__question");
const buttonChooseBig5EnvIndicator = document.querySelector("#choose__big5envindicator");
const buttonSendFinally = document.querySelector("#user-modal__button-send--question");
var e = document.getElementById("trait__Option");
const tbody = document.querySelector(".tbody");
const loadingio = document.getElementById("loadingio");
const bodySendModal = document.querySelector(".usersend-modal__body-chooose");
const levels = [...document.getElementsByClassName("level")];
// SelectBig5EnvIndicator

// Close
async function refreshQuestion() {
    const hr = [...document.querySelectorAll(".tbody hr")];
    const loadingio = document.getElementById("loadingio");
    loadingio.classList.add("loadingio__active");
    buttonSend.style.display = "none";
    buttonChooseBig5EnvIndicator.style.display = "none";
    hr.forEach(data => {
        data.remove()
    })
    var traitString = e.options[e.selectedIndex].value;
    try {
        const tbodyflex = [...document.getElementsByClassName("tbody--flex")];
        tbodyflex.forEach(data => {
            data.remove();
        })

        const response = await fetch("http://172.25.23.225:5000/admin/getquestions?trait=" + traitString);
        const datas = await response.json();
        loadingio.classList.remove("loadingio__active");
        buttonSend.style.display = "flex";
        buttonChooseBig5EnvIndicator.style.display = "flex";
        datas.forEach(async (data) => {
            tbody.innerHTML = tbody.innerHTML + `<div class="tbody--flex qa${data._id}">
            <div><label><input type="checkbox" id=${data._id} value=${data._id} name="questionCheckBox" /></label></div>
            <div class="question">${data.question}</div>
            <div style="font-weight:bold">${data.indicator}</div>
            <div class=${data.personality[0].Openness}>${data.personality[0].Openness}</div>
            <div class=${data.personality[1].Conscientious}>${data.personality[1].Conscientious}</div>
            <div class=${data.personality[2].Extraversion}>${data.personality[2].Extraversion}</div>
            <div class=${data.personality[3].Agreeable}>${data.personality[3].Agreeable}</div>
            <div class=${data.personality[4].Neuroticism}>${data.personality[4].Neuroticism}</div>
        </div> <hr style="margin-top:5px;z-index: 999;"/>`
        });
    }
    catch {
        toast({
            Title: "Error",
            Message: "Maybe error because sever",
            Type: "error",
            Duration: 2000
        })
    }
    questionCheckBox = [...document.getElementsByName("questionCheckBox")];
    function onSelectCheckBox(id) {
        questionCheckBox.forEach(data => {
            data.checked = false;
        });
        id.checked = true;
    }
    questionCheckBox.forEach(data => {
        data.onclick = () => {
            if (data.checked) {
                onSelectCheckBox(data);
            }
            else {
                data.checked = false;
            }

        }
    })
}
// function getpoint(data,trait){
//     var point;
//         switch (trait){
//             case 'Openness':
//                 if(data.personality[0].Openness=="High"){
//                     point=data.point.pointHigh;
//                 }
//                 else if(data.personality[0].Openness=="Medium"){
//                     point=data.point.pointMedium;
//                 }
//                 else{
//                     point=data.point.pointLow
//                 }
//                 break;
//             case 'Conscientious':
//                     if(data.personality[1].Conscientious=="High"){
//                         point=data.point.pointHigh;
//                     }
//                     else if(data.personality[1].Conscientious=="Medium"){
//                         point=data.point.pointMedium;
//                     }
//                     else{
//                         point=data.point.pointLow
//                     }
//                     break;
//             case 'Extraversion':
//                         if(data.personality[2].Extraversion=="High"){
//                             point=data.point.pointHigh;
//                         }
//                         else if(data.personality[2].Extraversion=="Medium"){
//                             point=data.point.pointMedium;
//                         }
//                         else{
//                             point=data.point.pointLow
//                         }
//                         break;
//             case 'Agreeable':
//                             if(data.personality[3].Agreeable=="High"){
//                                 point=data.point.pointHigh;
//                             }
//                             else if(data.personality[3].Agreeable=="Medium"){
//                                 point=data.point.pointMedium;
//                             }
//                             else{
//                                 point=data.point.pointLow
//                             }
//                             break;
//             default:
//                         if(data.personality[4].Neuroticism=="High"){
//                             point=data.point.pointHigh;
//                         }
//                         else if(data.personality[4].Neuroticism=="Medium"){
//                             point=data.point.pointMedium;
//                         }
//                         else{
//                             point=data.point.pointLow
//                         }
//                         break;         

//         }
//         return point;
// }
window.onload = async () => {
    refreshQuestion();
}
e.onchange = async () => {
    const tbodyflex = [...document.getElementsByClassName("tbody--flex")];
    const hrTag = [...document.querySelectorAll(".tbody hr")];
    const loadingio = document.getElementById("loadingio");
    tbodyflex.forEach(data => {
        data.remove();
    })
    hrTag.forEach(data => {
        data.remove();
    })
    loadingio.style.display = "block";
    buttonSend.style.display = "none";
    var traitString = e.options[e.selectedIndex].value;
    const response = await fetch("172.25.23.225:5000" + traitString);
    const datas = await response.json();
    loadingio.style.display = "none";
    buttonSend.style.display = "flex";
    datas.forEach((data) => {
        tbody.innerHTML = tbody.innerHTML + `<div class="tbody--flex qa${data._id}">
        <div><label><input type="checkbox" id=${data._id} value=${data._id} name="questionCheckBox" /></label></div>
        <div class="question">${data.question}</div>
        <div style="font-weight:bold">${data.indicator}</div>
        <div class=${data.personality[0].Openness}>${data.personality[0].Openness}</div>
        <div class=${data.personality[1].Conscientious}>${data.personality[1].Conscientious}</div>
        <div class=${data.personality[2].Extraversion}>${data.personality[2].Extraversion}</div>
        <div class=${data.personality[3].Agreeable}>${data.personality[3].Agreeable}</div>
        <div class=${data.personality[4].Neuroticism}>${data.personality[4].Neuroticism}</div>
    </div> <hr style="margin-top:5px;"/>`
    });
    questionCheckBox = [...document.getElementsByName("questionCheckBox")];
    function onSelectCheckBox(id) {
        questionCheckBox.forEach(data => {
            data.checked = false;
        });
        id.checked = true;
    }
    questionCheckBox.forEach(data => {
        data.onclick = () => {

            if (data.checked) {
                onSelectCheckBox(data);
            }
            else {
                data.checked = true;
            }

        }
    })
}
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
// button chonj big5Indicator mở
buttonChooseBig5EnvIndicator.onclick = () => {
    const big5EnvIndicatorFacetContent = document.querySelector(".big5EnvIndicator_content_facet");
    const selectIndicatorOption = document.querySelector("#select_indicator_option");
    const selectFacetOption = document.querySelector("#select_facet_option");
    const selectKeywordOption = document.querySelector("#select_keyword_option");
    const searchKeywords = document.querySelector("#search_keywords");
    const big5EnvIndicatorKeywordContent = document.querySelector(".big5EnvIndicator_content_keyword");
    const buttonChooseQuestion = document.querySelector("#user-modal__button-choose--question");
    const buttonCancleChooseQuestion = document.querySelector("#user-modal__button-cancel-choose--question");
    userChooseModel.style.display = "flex";
    const bigFiveDimensions = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"];
    const facets = [
        {
            big5Trait: "Extraversion",
            facet: ["Gregariousness", "Assertiveness", "Activity", "Excitement-seeking", "Positive emotions", "Warmth"]
        },
        {
            big5Trait: "Agreeableness",
            facet: ["Trust", "Straightforwardness", "Altruism", "Compliance", "Modesty", "Tender-mindedness"]
        },
        {
            big5Trait: "Conscientiousness",
            facet: ["Competence", "Order", "Dutifulness", "Achievement striving", "Self-discipline", "Deliberation"]
        },
        {
            big5Trait: "Neuroticism",
            facet: ["Anxiety", "Angry hostility", "Depression", "Self-consciousness", "Impulsiveness", "Vulnerability"]
        },
        {
            big5Trait: "Openness",
            facet: ["Ideas", "Fantasy", "Aesthetics", "Actions", "Feelings", "Values"]
        }
    ]
    const keywords = ["electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest", "electric", "forest",]

    // Select option keywords innerHTML
    var selectKeywordOptionsString = keywords.reduce((total, currentValue) => {
        return total + `<option value=${currentValue}>${currentValue}</option>`;
    }, "")
    selectKeywordOption.innerHTML += selectKeywordOptionsString;
    // selectKeywordOption.selectedIndex = 0;
    selectKeywordOption.size = 6;
    // Close
    // Change size select Facet option when open modal one time
    selectIndicatorOption.size = 6;
    // Close

    // Search keyword change
    searchKeywords.addEventListener("keyup", (e) => {
        var defaultValue;
        var selectKeywordOptionWithSearch = keywords.filter(value => value.indexOf(e.target.value) !== -1);
        if (e.target.value == "") {
            defaultValue = `<option value="all">All</option>`
        }
        else {
            defaultValue = "";
        }
        selectKeywordOption.innerHTML = selectKeywordOptionWithSearch.reduce((total, currentValue) => {
            return total + `<option value=${currentValue}>${currentValue}</option>`
        }, defaultValue);
        selectKeywordOption.selectedIndex = 0;
        if (selectKeywordOption.options[selectKeywordOption.selectedIndex].value == "all") {
            big5EnvIndicatorKeywordContent.innerText = ""
        }
        else {
            big5EnvIndicatorKeywordContent.innerText = selectKeywordOption.options[selectKeywordOption.selectedIndex].value;
        }

    })

    // Close

    // Select_keywords_change
    selectKeywordOption.addEventListener("change", () => {
        var valueSelectKeywordOption = selectKeywordOption.options[selectKeywordOption.selectedIndex].value;
        if (valueSelectKeywordOption == "all") {
            big5EnvIndicatorKeywordContent.innerText = "";
        }
        else {
            big5EnvIndicatorKeywordContent.innerText = valueSelectKeywordOption;
        }
    })
    // Close
    // Button choose
    buttonChooseQuestion.onclick = () => {
        var valueFacetContent = big5EnvIndicatorFacetContent.innerText;
        var valueKeywordContent = big5EnvIndicatorKeywordContent.innerText;
        (valueFacetContent == "" && valueKeywordContent == "") ? () => {
            toast({
                Title: "Success",
                Message: "Fetch all question",
                Type: "success",
                Duration: 2000
            });
            userChooseModel.style.display = "none";
        } : (valueFacetContent == "") ? toast({
            Title: "Error",
            Message: "Please choose facet",
            Type: "error",
            Duration: 2000
        }) : (valueKeywordContent == "") ? toast({
            Title: "Error",
            Message: "Please choose keyword",
            Type: "error",
            Duration: 2000
        }) : () => {
                toast({
                Title: "Success",
                Message: `Fetch question topic ${valueFacetContent} ${valueKeywordContent}`,
                Type: "success",
                Duration: 2000
            })
            userChooseModel.style.display = "none";
        }

    }
    // buttonChooseQuestion.addEventListener("click", () => {



    // })
    // Close

    //Cancle choose question
    buttonCancleChooseQuestion.onclick = () => {
        userChooseModel.style.display = "none";
    }
    //Close
    var selectIndicatorOptions = "";
    var selectFacetOptions = "";
    bigFiveDimensions.forEach(bigFiveDimension => {
        selectIndicatorOptions += `<option value=${bigFiveDimension}>${bigFiveDimension}</option>`
    })
    selectIndicatorOption.innerHTML += selectIndicatorOptions;
    selectIndicatorOption.addEventListener("change", () => {

        selectFacetOptions = "";
        var valueSelectIndicatorOption = selectIndicatorOption.options[selectIndicatorOption.selectedIndex].value;
        if (valueSelectIndicatorOption === "all") {
            selectFacetOption.value = "all";
            selectFacetOption.innerHTML = `<option value="all">All</option>`;
            selectFacetOption.size = 0;
            big5EnvIndicatorFacetContent.innerText = "";
        }
        else {
            var big5TraitFacets = facets.find(facet => facet.big5Trait === valueSelectIndicatorOption);
            selectFacetOption.size = big5TraitFacets.facet.length;


            big5TraitFacets.facet.forEach(value => {
                selectFacetOptions += `
                <option value=${value}>${value}</option>
                `
            })
            selectFacetOption.innerHTML = "";
            selectFacetOption.innerHTML += selectFacetOptions;
            selectFacetOption.selectedIndex = 0;
            big5EnvIndicatorFacetContent.innerText = selectFacetOption.options[selectFacetOption.selectedIndex].value;
            // console.log(selectFacetOption);
        }
        selectIndicatorOption.size = 0;
    })
    selectFacetOption.addEventListener("change", () => {
        const valueFacetOption = selectFacetOption.options[selectFacetOption.selectedIndex].value;
        big5EnvIndicatorFacetContent.innerText = valueFacetOption;
    })

    // close

    usermodel.onclick = () => {
        usermodel.style.display = "none";
    }
    userSendModalBody.onclick = (event) => {
        event.stopPropagation();
    }
    // user__choooseOption.forEach(data=>{
    //     data.onclick=()=>{
    //         data.classList.toggle("user__want--active")
    //     }
    // })

    user__choooseOption_cancel.onclick = () => {
        bodySendModal.innerHTML = "";
        usermodel.style.display = "none";
    }

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
            const response = await fetch("http://"+localHostEnv+":5000/admin/sendquestion", {
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
    const saveButton = document.getElementById("save__question");
    const levelOpenness = document.getElementById("levelOpenness");
    const levelConscientious = document.getElementById("levelConciencetious");
    const levelExtraversion = document.getElementById("levelExtraversion");
    const levelAgreeable = document.getElementById("levelAgreeable");
    const levelNeuroticism = document.getElementById("levelNeuroticism");
    const pointHigh = document.getElementById("high__point");
    const pointMedium = document.getElementById("medium__point");
    const pointLow = document.getElementById("low__point");
    const question = document.getElementById("input__question");
    const loadingioQuestion = document.getElementById("loadingioquestion");
    const indicator = document.getElementById("input__indicator");
    saveButton.onclick = async () => {
        if (question.value == "") {
            toast({
                Title: "Error",
                Message: "You need input question",
                Type: "error",
                Duration: 2000
            })
        }
        else if (indicator.value == "") {
            toast({
                Title: "Error",
                Message: "You need input indicator",
                Type: "error",
                Duration: 2000
            })
        }
        else {
            const dataPointHigh = parseInt(pointHigh.value);
            const dataPointMedium = parseInt(pointMedium.value);
            const dataPointLow = parseInt(pointLow.value);
            const data = {
                "Openness": levelOpenness.options[levelOpenness.selectedIndex].text,
                "Conscientious": levelConscientious.options[levelConscientious.selectedIndex].text,
                "Extraversion": levelExtraversion.options[levelExtraversion.selectedIndex].text,
                "Agreeable": levelAgreeable.options[levelAgreeable.selectedIndex].text,
                "Neuroticism": levelNeuroticism.options[levelNeuroticism.selectedIndex].text,
                "point": {
                    "pointHigh": dataPointHigh,
                    "pointMedium": dataPointMedium,
                    "pointLow": dataPointLow
                },
                "question": question.value,
                "indicator": indicator.value
            }
            try {
                loadingioQuestion.style.display = "block";
                const response = await fetch("http://172.25.23.225:5000/admin/addquestion", {
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
                if (dataResponse.error == "no") {
                    loadingioQuestion.style.display = "none";
                    toast({
                        Title: "Success",
                        Message: "You was input question success",
                        Type: 'success',
                        Duration: 2000,
                    })
                    newData();
                    refreshQuestion();
                }
                else {
                    loadingioQuestion.style.display = "none";
                    toast({
                        Title: "Error",
                        Message: dataResponse.error,
                        Type: 'error',
                        Duration: 2000,
                    })
                }
            }
            catch {
                loadingioQuestion.style.display = "none";
                toast({
                    Title: "Error",
                    Message: "Maybe your fault because sever",
                    Type: 'error',
                    Duration: 2000,
                })
            }
        }

    }

    function newData() {

        question.value = "";
        indicator.value = "";
        levelOpenness.value = 2;
        levelNeuroticism.value = 2;
        levelExtraversion.value = 2;
        levelConscientious.value = 2;
        levelAgreeable.value = 2;
        pointHigh.value = 20;
        pointMedium.value = 15;
        pointLow.value = 10;
    }
    levels.forEach(level => {
        level.onchange = () => {
            var arrayValue = [];
            levels.forEach(data => {
                arrayValue.push(data.value);
            })
            // if(arrayValue.find(1)!=-1){
            //     toast({
            //             Title:"Error",
            //             Message:"The question just have one level high",
            //             Type:"error",
            //             Duration:2000
            //         })
            // }
            arrayValue = arrayValue.filter(element => (element != "2") && (element != "3"));
            if (arrayValue.length > 1) {
                toast({
                    Title: "Error",
                    Message: "The question just have one level high",
                    Type: "error",
                    Duration: 2000
                })
                arrayValue = [];
                level.value = 2;
            }
            return;
        }
    })
}

// function toast({Title,Message,Type,Duration}){
//     const main=document.getElementById("toast");
//     if(main){
//         const toast=document.createElement("div");
//         const icons={
//             success:"fas fa-check-circle",
//             error:"fas fa-exclamation-circle"
//         }
//         const icon=icons[Type];
//         const duration=(Duration/1000).toFixed(2);
//         const Timeout=Duration+1000;
//         toast.classList.add("toast",`toast--${Type}`);
//         toast.innerHTML=`
//         <div class="toast__icon">
//             <i class="${icon}"></i>
//         </div>
//         <div class="toast__desc">
//             <h3 class="toast__status">${Title}</h3>
//             <p class="toast__inforStatus">${Message}</p>
//         </div>
//         <div class="toast__close">
//             <i class="fas fa-times"></i>
//         </div>
//         `
//         main.appendChild(toast);
//         toast.style.animation=`Fadein 0.5s ease,Close 1s ${duration}s forwards`;
//         const autoRemoveId=setTimeout(function(){
//            main.removeChild(toast)
//         },Timeout)
//         main.onclick=function(e){
//            if(e.target.closest(".toast__close")){
//                main.removeChild(toast);
//                clearTimeout(autoRemoveId);
//            }
//        }
//        }
// }
