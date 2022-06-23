import BornHtml from "./refresquestion/born_html.js";
searchKeywordInput.addEventListener("keyup", (e) => {
    tbody.innerHTML = "";
    const questionAndIndicatorListFromDBWasFilter = questionAndIndicatorListFromDB.filter(value => value.big5EnvIndicator.indexOf(e.target.value) !== -1);
    console.log(questionAndIndicatorListFromDBWasFilter);
    questionAndIndicatorListFromDBWasFilter.forEach((data) => {
        tbody.innerHTML = tbody.innerHTML + BornHtml(data);
    });
})