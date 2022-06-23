function SelectFacetOptionChange(){
    selectFacetOption.addEventListener("change", () => {
        const valueFacetOption = selectFacetOption.options[selectFacetOption.selectedIndex].value;
        big5EnvIndicatorFacetContent.innerText = valueFacetOption;
    })
}
export default SelectFacetOptionChange;
