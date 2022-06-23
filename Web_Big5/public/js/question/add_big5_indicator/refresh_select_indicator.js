function refresh_select_indicator(){
    selectIndicatorOption.size = 6;
    selectKeywordOption.innerHTML=`<option value="all" selected>All</option>`;
    selectIndicatorOption.innerHTML=`<option value="all" selected>All</option>`;
    bigFiveDimensions.forEach(bigFiveDimension => {
        selectIndicatorOption.innerHTML += `<option value=${bigFiveDimension}>${bigFiveDimension}</option>`
    })
    selectKeywordOption.selectedIndex=0;
    selectFacetOption.innerHTML=`<option value="all" selected>All</option>`;
    selectFacetOption.size=0;
    big5EnvIndicatorList=[];
    big5EnvIndicatorFacetContent.innerHTML="";
    big5EnvIndicatorKeywordContent.innerHTML="";
    big5EnvIndicatorContentList.innerHTML="";
}
export default refresh_select_indicator;