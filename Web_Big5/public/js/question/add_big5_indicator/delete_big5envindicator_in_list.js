function RemoveBig5EnvIndicatorInList(value){
    const elementBig5EnvIndicatorInList=document.querySelector(`div[data-value='${value}']`);
    elementBig5EnvIndicatorInList.remove();
    big5EnvIndicatorList=big5EnvIndicatorList.filter(big5EnvIndicator=>big5EnvIndicator!==value);
}