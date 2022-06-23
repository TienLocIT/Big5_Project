import toast from "./toast.js";
import BtnCreateBig5EnvIndicator from "./add_big5_indicator/btn_create_big5envindicator_click.js";
import SelectIndicatorOptionChange from "./add_big5_indicator/select_indicator_option_change.js";
import OnFirstLoad from "./add_big5_indicator/on_first_load_add_big5env_indicator.js";
import { getKeywordsToSelect,onChangeSelectKeyWord} from "./add_big5_indicator/Select_keyword_option.js"
import BtnAddBig5IndicatorToList from "./add_big5_indicator/add_big5envindicator_to_list.js";
import refresh_select_indicator from "./add_big5_indicator/refresh_select_indicator.js";
import SelectFacetOptionChange from "./add_big5_indicator/Select_Facet_Option_Change.js";
buttonChooseBig5EnvIndicator.onclick = async () => {
    // const big5EnvIndicatorFacetContent = document.querySelector(".big5EnvIndicator_content_facet");
    // const selectIndicatorOption = document.querySelector("#select_indicator_option");
    // const selectFacetOption = document.querySelector("#select_facet_option");
    // const selectKeywordOption = document.querySelector("#select_keyword_option");
    // const searchKeywords = document.querySelector("#search_keywords");
    // const big5EnvIndicatorKeywordContent = document.querySelector(".big5EnvIndicator_content_keyword");
    // const buttonChooseQuestion = document.querySelector("#user-modal__button-choose--question");
    // const buttonCancleChooseQuestion = document.querySelector("#user-modal__button-cancel-choose--question");
    // const loadingioBig5Modal=document.querySelector(".userchoose-modal__body .loadingiomodal")
    userChooseModel.style.display = "flex";
    loadingioBig5Modal.style.display="block";
    
    // Select_keyword_option
    OnFirstLoad();
    await getKeywordsToSelect();
    onChangeSelectKeyWord();
    btnAddBig5IndicatorToList.onclick=async ()=>{
        var big5EnvIndicatorString=await BtnAddBig5IndicatorToList();
        if(big5EnvIndicatorString!=""){
            big5EnvIndicatorContentList.innerHTML+=`
            <div data-value='${big5EnvIndicatorString}'>
            <span style="text-align:center;margin_left:5%">${big5EnvIndicatorString}</span>
            <span style="text-decoration:underline;color:blue;" 
            onMouseOver="this.style.opacity=0.2;this.style.cursor='pointer'"
            onMouseOut="this.style.opacity=1;this.style.cursor='auto'"
            onClick="RemoveBig5EnvIndicatorInList('${big5EnvIndicatorString}')";
            > Deleted </span>
            </div>
            `;
            big5EnvIndicatorList.push(big5EnvIndicatorString);
            console.log(big5EnvIndicatorList);
        }
    }
    
    // const keywords = await fetchAPI("http://192.168.1.9:5000/admin/keywords/getAll",null);
    // console.log(keywords);
    // Select option keywords innerHTML
    // var selectKeywordOptionsString = keywords.reduce((total, currentValue) => {
    //     return total + `<option value=${currentValue}>${currentValue.key_word}</option>`;
    // }, "")
    // selectKeywordOption.innerHTML += selectKeywordOptionsString;
    // selectKeywordOption.selectedIndex = 0; //
    // selectKeywordOption.size = 6;
    // Close
    // Change size select Facet option when open modal one time
    // selectIndicatorOption.size = 6; tí mở lại
    // Close

    // Search keyword change
    // searchKeywords.addEventListener("keyup", (e) => {
    //     var defaultValue;
    //     var selectKeywordOptionWithSearch = keywords.filter(value => value.indexOf(e.target.value) !== -1);
    //     if (e.target.value == "") {
    //         defaultValue = `<option value="all">All</option>`
    //     }
    //     else {
    //         defaultValue = "";
    //     }
    //     selectKeywordOption.innerHTML = selectKeywordOptionWithSearch.reduce((total, currentValue) => {
    //         return total + `<option value=${currentValue}>${currentValue}</option>`
    //     }, defaultValue);
    //     selectKeywordOption.selectedIndex = 0;
    //     if (selectKeywordOption.options[selectKeywordOption.selectedIndex].value == "all") {
    //         big5EnvIndicatorKeywordContent.innerText = ""
    //     }
    //     else {
    //         big5EnvIndicatorKeywordContent.innerText = selectKeywordOption.options[selectKeywordOption.selectedIndex].value;
    //     }

    // }) mở lại

    // Close

    // Select_keywords_change
    // selectKeywordOption.addEventListener("change", () => {
    //     var valueSelectKeywordOption = selectKeywordOption.options[selectKeywordOption.selectedIndex].value;
    //     if (valueSelectKeywordOption == "all") {
    //         big5EnvIndicatorKeywordContent.innerText = "";
    //     }
    //     else {
    //         big5EnvIndicatorKeywordContent.innerText = valueSelectKeywordOption;
    //     }
    // })
    // Close
    // Button choose
    // buttonChooseQuestion.onclick = () => {
    //     var valueFacetContent = big5EnvIndicatorFacetContent.innerText;
    //     var valueKeywordContent = big5EnvIndicatorKeywordContent.innerText;
    //     (valueFacetContent == "" && valueKeywordContent == "") ? () => {
    //         toast({
    //             Title: "Success",
    //             Message: "Fetch all question",
    //             Type: "success",
    //             Duration: 2000
    //         });
    //         userChooseModel.style.display = "none";
    //     } : (valueFacetContent == "") ? toast({
    //         Title: "Error",
    //         Message: "Please choose facet",
    //         Type: "error",
    //         Duration: 2000
    //     }) : (valueKeywordContent == "") ? toast({
    //         Title: "Error",
    //         Message: "Please choose keyword",
    //         Type: "error",
    //         Duration: 2000
    //     }) : () => {
    //             toast({
    //             Title: "Success",
    //             Message: `Fetch question topic ${valueFacetContent} ${valueKeywordContent}`,
    //             Type: "success",
    //             Duration: 2000
    //         })
    //         userChooseModel.style.display = "none";
    //     }

    // }mở lại
    BtnCreateBig5EnvIndicator();
    // buttonChooseQuestion.addEventListener("click", () => {



    // })
    // Close

    //Cancle choose question
    buttonCancleChooseQuestion.onclick = () => {
        refresh_select_indicator();
        userChooseModel.style.display = "none";
    }
    //Close
    // var selectIndicatorOptions = "";
    // var selectFacetOptions = "";
    // bigFiveDimensions.forEach(bigFiveDimension => {
    //     selectIndicatorOptions += `<option value=${bigFiveDimension}>${bigFiveDimension}</option>`
    // })
    // selectIndicatorOption.innerHTML += selectIndicatorOptions; //tí mở lại
    // selectIndicatorOption.addEventListener("change", () => {
    //     // selectFacetOptions = "";
    //     var valueSelectIndicatorOption = selectIndicatorOption.options[selectIndicatorOption.selectedIndex].value;
    //     if (valueSelectIndicatorOption === "all") {
    //         selectFacetOption.value = "all";
    //         selectFacetOption.innerHTML = `<option value="all">All</option>`;
    //         selectFacetOption.size = 0;
    //         big5EnvIndicatorFacetContent.innerText = "";
    //     }
    //     else {
    //         var big5TraitFacets = facets.find(facet => facet.big5Trait === valueSelectIndicatorOption);
    //         selectFacetOption.size = big5TraitFacets.facet.length;
    //         selectFacetOption.innerHTML = "";
    //         big5TraitFacets.facet.forEach(value => {
    //             selectFacetOption.innerHTML += `
    //             <option value=${value}>${value}</option>
    //             `
    //         })
            
    //         // selectFacetOption.innerHTML += selectFacetOptions;
    //         selectFacetOption.selectedIndex = 0;
    //         big5EnvIndicatorFacetContent.innerText = selectFacetOption.options[selectFacetOption.selectedIndex].value;
    //         // console.log(selectFacetOption);
    //     }
    //     selectIndicatorOption.size = 0;
    // }) mơ lại
    SelectIndicatorOptionChange();
    // selectFacetOption.addEventListener("change", () => {
    //     const valueFacetOption = selectFacetOption.options[selectFacetOption.selectedIndex].value;
    //     big5EnvIndicatorFacetContent.innerText = valueFacetOption;
    // }) mở lại
    SelectFacetOptionChange();
    loadingioBig5Modal.style.display="none";
}
