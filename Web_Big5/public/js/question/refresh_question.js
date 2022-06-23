
import fetchAPI from "../fetchAPI.js";
import toast from "./toast.js";
import BornHtml from "./refresquestion/born_html.js";
import localHostEnv from "../config.js";
async function refreshQuestion() {
    
    const hr = [...document.querySelectorAll(".tbody hr")];
    const loadingio = document.getElementById("loadingio");
    questionAndIndicatorListFromDB=[];
    loadingio.classList.add("loadingio__active");
    buttonSend.style.display = "none";
    buttonChooseBig5EnvIndicator.style.display = "none";
    hr.forEach(data => {
        data.remove()
    })
    // var traitString = e.options[e.selectedIndex].value;
    try {
        const tbodyflex = [...document.getElementsByClassName("tbody--flex")];
        tbodyflex.forEach(data => {
            data.remove();
        })
        

        const responseGetQuestion = await fetchAPI("http://"+localHostEnv+"/admin/getquestions");
        const responseGetBig5EnvIndicator=await fetchAPI("http://"+localHostEnv+"/admin/getBig5EnvIndicator");
        responseGetBig5EnvIndicator.big5EnvIndicatorList.forEach(value=>{
            questionAndIndicatorListFromDB.push(value);
        })
        responseGetQuestion.questions.forEach(value=>{
            questionAndIndicatorListFromDB.push(value);
        })
        console.log(questionAndIndicatorListFromDB);
        // const datas = await responseGe.json();
        loadingio.classList.remove("loadingio__active");
        buttonSend.style.display = "flex";
        buttonChooseBig5EnvIndicator.style.display = "flex";
        questionAndIndicatorListFromDB.forEach((data) => {
            tbody.innerHTML = tbody.innerHTML+ BornHtml(data);
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
export default refreshQuestion;