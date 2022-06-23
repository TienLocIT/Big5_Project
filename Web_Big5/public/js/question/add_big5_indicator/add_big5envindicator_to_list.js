import toast from "../toast.js";
import fetchAPI from "../../fetchAPI.js";
import localHostEnv from "../../config.js";
async function BtnAddBig5IndicatorToList() {
    var valueFacetContent = big5EnvIndicatorFacetContent.innerText;
    var valueKeywordContent = big5EnvIndicatorKeywordContent.innerText;
    var big5EnvIndicator = valueKeywordContent + " " + valueFacetContent;
    if (valueFacetContent == "" && valueKeywordContent == "") {
        toast({
            Title: "Error",
            Message: "You need choose big5 environment indicator",
            Type: "error",
            Duration: 2000
        });
    }
    else if (valueFacetContent == "") {
        toast({
            Title: "Error",
            Message: "You need choose big5  trait facet",
            Type: "error",
            Duration: 2000
        });
    }
    else if (valueKeywordContent == "") {
        toast({
            Title: "Error",
            Message: "You need choose keyword",
            Type: "error",
            Duration: 2000
        });
    }
    else {
        if (big5EnvIndicatorList.includes(big5EnvIndicator)) {
            toast({
                Title: "Error",
                Message: "Big 5 indicator have on list",
                Type: "error",
                Duration: 2000
            })
        }
        else {
            const checkBig5EnvIndicator = await fetchAPI("http://"+localHostEnv+"/admin/adminCheckIndicator", {
                big5EnvIndicator: valueKeywordContent + " " + valueFacetContent
            }, "POST");
            if (checkBig5EnvIndicator.Status == "Error") {
                toast({
                    Title: "Error",
                    Message: checkBig5EnvIndicator.Notification,
                    Type: "error",
                    Duration: 2000
                })
            }
            else {
                if (checkBig5EnvIndicator.Status == "Success") {
                    const responseCheckBig5EnvIndicatorWithQuestion = await fetchAPI(`http://${localHostEnv}/admin/questions/getByBig5EnvIndicator?big5EnvIndicator=${valueKeywordContent + " " + valueFacetContent}`);
                    if (responseCheckBig5EnvIndicatorWithQuestion.questions.length == 0) {
                        toast({
                            Title: "Success",
                            Message: "Add big5 env indicator in to list success",
                            Type: "success",
                            Duration: 2000
                        })
                        return valueKeywordContent + " " + valueFacetContent;
                    }
                    else{
                        toast({
                            Title: "Error",
                            Message: "Big 5 indicator was have on list",
                            Type: "error",
                            Duration: 2000
                        })
                    }
                }
                else {
                    toast({
                        Title: "Error",
                        Message: "Big 5 indicator was have on list",
                        Type: "error",
                        Duration: 2000
                    })
                }

            }
        }

    }
    return "";
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
    // }
}
export default BtnAddBig5IndicatorToList;