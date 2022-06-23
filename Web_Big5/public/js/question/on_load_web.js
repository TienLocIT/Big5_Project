// var questionCheckBox;
// const buttonSend = document.querySelector("#send__question");
// const buttonChooseBig5EnvIndicator = document.querySelector("#choose__big5envindicator");
// const toast=require("./toast");
import refreshQuestion from "./refresh_question.js"
window.onload = async () => {
    refreshQuestion();
}