function SelectIndicatorOptionChange(){
selectIndicatorOption.addEventListener("change", () => {
    // selectFacetOptions = "";
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
        selectFacetOption.innerHTML = "";
        big5TraitFacets.facet.forEach(value => {
            selectFacetOption.innerHTML += `
            <option value=${value}>${value}</option>
            `
        })
        
        // selectFacetOption.innerHTML += selectFacetOptions;
        selectFacetOption.selectedIndex = 0;
        big5EnvIndicatorFacetContent.innerText = selectFacetOption.options[selectFacetOption.selectedIndex].value;
        // console.log(selectFacetOption);
    }
    selectIndicatorOption.size = 0;
})
}
export default SelectIndicatorOptionChange;
