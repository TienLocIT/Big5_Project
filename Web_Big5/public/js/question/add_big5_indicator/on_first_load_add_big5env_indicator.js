function OnFirstLoad(){
    selectIndicatorOption.size = 6;
    var selectIndicatorOptions = "";
    bigFiveDimensions.forEach(bigFiveDimension => {
        selectIndicatorOptions += `<option value=${bigFiveDimension}>${bigFiveDimension}</option>`
    })
    selectIndicatorOption.innerHTML += selectIndicatorOptions;
}
export default OnFirstLoad;