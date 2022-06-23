function BornHtml(data){
    if(data.question){
        return `<div class="tbody--flex qa${data._id} question_big5env_indicator">
        <div><label><input type="checkbox" id=${data._id} value=${data._id} name="questionCheckBox" /></label></div>
        <div class="question">${data.question}</div>
        <div style="font-weight:bold"
        onMouseOver="this.style.opacity=0.2,this.style.cursor='pointer'"
        onMouseOut="this.style.opacity=1,this.style.cursor='default'"
        onClick="Big5EnvIndicatorOnClick('${data.big5EnvIndicator}')"
        >${data.big5EnvIndicator}</div>
        <div class=${data.personality[0].Openness}>${data.personality[0].Openness}</div>
        <div class=${data.personality[1].Conscientious}>${data.personality[1].Conscientious}</div>
        <div class=${data.personality[2].Extraversion}>${data.personality[2].Extraversion}</div>
        <div class=${data.personality[3].Agreeable}>${data.personality[3].Agreeable}</div>
        <div class=${data.personality[4].Neuroticism}>${data.personality[4].Neuroticism}</div>
    </div> <hr style="margin-top:5px;z-index: 999;"/>`
    }
    else{
        return `<div class="tbody--flex question_big5env_indicator">
        <div></div>
        <div class="question"></div>
        <div style="font-weight:bold"
        onMouseOver="this.style.opacity=0.2,this.style.cursor='pointer'"
        onMouseOut="this.style.opacity=1,this.style.cursor='default'"
        onClick="Big5EnvIndicatorOnClick('${data.big5EnvIndicator}')"
        >${data.big5EnvIndicator}</div>
        <div></div>
        <div></div>
        <div></div>
        <div></div>
        <div></div>
    </div> <hr style="margin-top:5px;z-index: 999;"/>`
    }
}
export default BornHtml;