import fetchAPI from "../../fetchAPI.js";
import toast from "../toast.js";
import refresh_select_indicator from "./refresh_select_indicator.js";
import refreshQuestion from "../refresh_question.js";
import localHostEnv from "../../config.js";
function BtnCreateBig5EnvIndicator() {
    buttonChooseQuestion.onclick = async () => {
        loadingioBig5Modal.style.display="block";
        if (big5EnvIndicatorList.length == 0) {
            toast({
                Title: "Error",
                Message: "Please choose at least 1 big5 environment indicator",
                Type: "error",
                Duration: 2000
            })
        }
        else {
            const big5EnvIndicatorListJson=[];
            big5EnvIndicatorList.forEach(value=>{
                big5EnvIndicatorListJson.push({
                    "big5EnvIndicator":value
                })
            })
            var response = await fetchAPI("http://"+localHostEnv+"/admin/addBig5EnvIndicator",
                {big5EnvIndicatorList:big5EnvIndicatorListJson}, "POST")
            if (response.Notification == "Error") {
                toast({
                    Title: "Error",
                    Message: "Sorry ,you can't create big5 env indicator right now",
                    Type: "error",
                    Duration: 2000
                })
            }
            toast({
                Title: "Success",
                Message: "Create big5 env indicator success",
                Type: "success",
                Duration: 2000
            })
            // var valueFacetContent = big5EnvIndicatorFacetContent.innerText;
            // var valueKeywordContent = big5EnvIndicatorKeywordContent.innerText;
            // (valueFacetContent == "" && valueKeywordContent == "") ? () => {
            //     toast({
            //         Title: "Success",
            //         Message: "Fetch all question",
            //         Type: "success",
            //         Duration: 2000
            //     });
            //     userChooseModel.style.display = "none";
            // } : (valueFacetContent == "") ? toast({
            //     Title: "Error",
            //     Message: "Please choose facet",
            //     Type: "error",
            //     Duration: 2000
            // }) : (valueKeywordContent == "") ? toast({
            //     Title: "Error",
            //     Message: "Please choose keyword",
            //     Type: "error",
            //     Duration: 2000
            // }) : () => {
            //         toast({
            //         Title: "Success",
            //         Message: `Fetch question topic ${valueFacetContent} ${valueKeywordContent}`,
            //         Type: "success",
            //         Duration: 2000
            //     })
            refresh_select_indicator();
            refreshQuestion();
            userChooseModel.style.display = "none";
        }
    }
}
export default BtnCreateBig5EnvIndicator;