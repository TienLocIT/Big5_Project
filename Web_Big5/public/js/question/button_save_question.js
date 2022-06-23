import newData from "./new_data_input_question.js"
import toast from "./toast.js"
import refreshQuestion from "./refresh_question.js"
import fetchAPI from "../fetchAPI.js"
import localHostEnv from "../config.js"
saveButton.onclick = async () => {
    if (question.value == "") {
        toast({
            Title: "Error",
            Message: "You need input question",
            Type: "error",
            Duration: 2000
        })
    }
    else if (big5Envindicator.innerText == "") {
        toast({
            Title: "Error",
            Message: "You need chooose big5 env indicator",
            Type: "error",
            Duration: 2000
        })
    }
    else {
        loadingioQuestion.style.display="block";
        const responseCheckBig5EnvIndicatorWithQuestion = await fetchAPI(`http://${localHostEnv}/admin/questions/getByBig5EnvIndicator?big5EnvIndicator=${big5Envindicator.innerHTML}`);
        const responseCheckBig5EnvIndicatorWithIndicator = await fetchAPI("http://"+localHostEnv+"/admin/adminCheckIndicator", {
            big5EnvIndicator: big5Envindicator.innerHTML
        }, "POST");
        if (responseCheckBig5EnvIndicatorWithQuestion.Status == "Success" ) {
            // responseCheckBig5EnvIndicatorWithIndicator.Status == "Error" &&
            // responseCheckBig5EnvIndicatorWithIndicator.typeCheckWhen == "Save"
            if (responseCheckBig5EnvIndicatorWithQuestion.questions.length == 0) {
                console.log(responseCheckBig5EnvIndicatorWithIndicator);
                if(responseCheckBig5EnvIndicatorWithIndicator.Status == "Error" &&
                responseCheckBig5EnvIndicatorWithIndicator.typeCheckWhen == "Save Question"){
                    await saveQuestion();
                }
                else{
                    toast({
                        Title: "Error",
                        Message: "You don't have this big5 env indicator",
                        Type: "error",
                        Duration: 2000
                    })
                    loadingioQuestion.style.display="none";
                }
            }
            else{
                await saveQuestion();
            }
        }
        else{
            toast({
                Title: "Error",
                Message: "Maybe have some fault",
                Type: "error",
                Duration: 2000
            })
            loadingioQuestion.style.display="none";
        }
    }

}
async function saveQuestion(){
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
        "big5EnvIndicator": big5Envindicator.innerText
    }
        const responseSaveQuestion = await fetchAPI("http://"+localHostEnv+"/admin/addquestion",data,"POST");
        
        if (responseSaveQuestion.error == "no") {
            const responseCheckAndDeleteQuestion=await fetchAPI("http://"+localHostEnv+"/admin/keyword/checkAndDelete",{big5EnvIndicator:big5Envindicator.innerText},"DELETE");
            if(responseCheckAndDeleteQuestion.Status=="Success"){
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
            else{
                toast({
                    Title: "Error",
                    Message: "Maybe have some fault",
                    Type: 'error',
                    Duration: 2000,
                })
            }
           
        }
        else {
            loadingioQuestion.style.display = "none";
            toast({
                Title: "Error",
                Message: responseSaveQuestion.Notification,
                Type: 'error',
                Duration: 2000,
            })
        }
}