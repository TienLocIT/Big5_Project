import localHostEnv from "../../config.js";
import fetchAPI from "../../fetchAPI.js"
import SearchKeywordsKeyUp from "./search_keywords_key_up.js";
async function getKeywordsToSelect() {
    const keywords = await fetchAPI("http://"+localHostEnv+"/admin/keywords/getAll", null);
    var arrayKeyword=[];
    keywords.forEach(keyword=>{
        arrayKeyword.push(keyword.key_word);
    })
    // Select option keywords innerHTML
    var selectKeywordOptionsString = keywords.reduce((total, currentValue) => {
        return total + `<option value='${currentValue.key_word}'>${currentValue.key_word}</option>`
    }, "")
    selectKeywordOption.innerHTML += selectKeywordOptionsString;
    selectKeywordOption.size = 6;
    SearchKeywordsKeyUp(arrayKeyword);
}
function onChangeSelectKeyWord(){
    selectKeywordOption.addEventListener("change", () => {
        var valueSelectKeywordOption = selectKeywordOption.options[selectKeywordOption.selectedIndex].value;
        if (valueSelectKeywordOption == "all") {
            big5EnvIndicatorKeywordContent.innerText = "";
        }
        else {
            big5EnvIndicatorKeywordContent.innerText = valueSelectKeywordOption;
        }
    })
}
export {
    getKeywordsToSelect,onChangeSelectKeyWord
}