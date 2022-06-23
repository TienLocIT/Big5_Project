import localHostEnv from "../config";

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
    const response = await fetch("http://"+localHostEnv+"/admin/getquestions?trait=" + traitString);
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