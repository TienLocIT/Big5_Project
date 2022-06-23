function SearchKeywordsKeyUp(keywords){
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
}
export default SearchKeywordsKeyUp;