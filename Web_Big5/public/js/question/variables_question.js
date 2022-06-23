var questionCheckBox;
var questionAndIndicatorListFromDB=[];
var questionAndIndicatorListFromDB;
const searchKeywordInput=document.getElementById("search_big5_env_indicator");
const user__choooseOption_cancel = document.getElementById("user-modal__button-cancel--question");
// const user__choooseOption=[...document.getElementsByClassName("user__chooose--option")];
const userSendModalBody = document.querySelector(".usersend-modal__body");
const userChooseModalBody = document.querySelector(".usersend-modal__body");
const usermodel = document.querySelector(".usersend-modal");
const userChooseModel = document.querySelector(".userchoose-modal");
const buttonSend = document.querySelector("#send__question");
const buttonChooseBig5EnvIndicator = document.querySelector("#choose__big5envindicator");
const buttonSendFinally = document.querySelector("#user-modal__button-send--question");
// var e = document.getElementById("trait__Option");
const tbody = document.querySelector(".tbody");
const loadingio = document.getElementById("loadingio");
const bodySendModal = document.querySelector(".usersend-modal__body-chooose");
const levels = [...document.getElementsByClassName("level")];
const bigFiveDimensions = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"];
    const facets = [
        {
            big5Trait: "Extraversion",
            facet: ["Gregariousness", "Assertiveness", "Activity", "Excitement-seeking", "Positive emotions", "Warmth"]
        },
        {
            big5Trait: "Agreeableness",
            facet: ["Trust", "Straightforwardness", "Altruism", "Compliance", "Modesty", "Tender-mindedness"]
        },
        {
            big5Trait: "Conscientiousness",
            facet: ["Competence", "Order", "Dutifulness", "Achievement striving", "Self-discipline", "Deliberation"]
        },
        {
            big5Trait: "Neuroticism",
            facet: ["Anxiety", "Angry hostility", "Depression", "Self-consciousness", "Impulsiveness", "Vulnerability"]
        },
        {
            big5Trait: "Openness",
            facet: ["Ideas", "Fantasy", "Aesthetics", "Actions", "Feelings", "Values"]
        }
    ]