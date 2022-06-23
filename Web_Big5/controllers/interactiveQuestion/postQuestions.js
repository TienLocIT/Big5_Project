const client = require("../../mongoDB");
const shortid = require("shortid");
const checkQuestionConcide = require("../checkQuestionConcide/concideQuestion");
const checkQuestion = require("../checkQuestion/checkQuestion");
module.exports = {
    postQuestions: async (phoneNumber) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5")
        async function updateQuestionWithNull(questionArray, phoneNumber) {
            var id = shortid.generate();
            var questions = [];
            var QuestionsAll = [];

            // while (questions.length < 5) {
            //     var random = Math.floor(Math.random() * questionArray.length);
            //     var questionContent = questionArray[random].question;

            //     if (questions.length == 0) {
            //         var questionBody = {
            //             "question": questionArray[random].question
            //         }
            //         questions.push(questionBody);
            //     }
            //     else {
            //         if (checkQuestionConcide.concideQuestion(questionContent, questions) == false) {
            //             var questionBody = {
            //                 "question": questionArray[random].question
            //             }
            //             questions.push(questionBody);
                       
            //         }
            //     }
            // }
            for(var i=0; i<5 ;i++){
                    var questionBody = {
                        question:questionArray[random].question
                    }
                    questions.push(questionBody);
            }
            var currentdate = new Date();
            // var date=(currentdate.getMonth()+1) + "/"+ currentdate.getDate() + "/" + currentdate.getFullYear();
            // var time=currentdate.getHours()+":"+currentdate.getMinutes()+":"+currentdate.getSeconds();

            var body = {
                "id": id,
                "dateTime": currentdate,
                "questionsDate": questions,
            }
            QuestionsAll.push(body);
            await db.collection("users").updateOne({ "phoneNumber": phoneNumber }, { $set: { "Questions": QuestionsAll } });
        }
        // async function updateQuestion(questionArray, phoneNumber) {
        //     var questions = [];
        //     var QuestionsAll = [];
        //     var id = shortid.generate();
        //     while (questions.length < 5) {
        //         var random = Math.floor(Math.random() * questionArray.length);
        //         var questionContent = questionArray[random].question;
        //         var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
        //         if (questions.length == 0) {
        //             if (checkQuestionConcide.coincideQuestionDB(questionContent, user.Questions) == false) {
        //                 var questionBody = {
        //                     "question": questionArray[random].question
        //                 }
        //                 questions.push(questionBody);
        //             }
        //         }
        //         else {
        //             if (checkQuestionConcide.coincideQuestionDB(questionContent, user.Questions) == false && checkQuestionConcide.concideQuestion(questionContent, questions) == false) {
        //                 var questionBody = {
        //                     "question": questionArray[random].question
        //                 }
        //                 questions.push(questionBody);
        //             }
        //         }
        //     }
        //     const currentdate = new Date();
        //     // var date=(currentdate.getMonth()+1) + "/"+ currentdate.getDate()  + "/" + currentdate.getFullYear()
        //     // var time=currentdate.getHours()+":"+currentdate.getMinutes()+":"+currentdate.getSeconds();
        //     // var date = (currentdate.getMonth() + 1) + "/" + currentdate.getDate() + "/" + currentdate.getFullYear()
        //     var body = {
        //         "id": id,
        //         "dateTime": currentdate,
        //         "questionsDate": questions,
        //     }
        //     QuestionsAll.push(body);
        //     await db.collection("users").updateOne({ "phoneNumber": phoneNumber }, {
        //         $push: {
        //             "Questions": {
        //                 $each: QuestionsAll
        //             }
        //         }
        //     });
        // }
        // function compareTime(time1,time2){
        // var regex = new RegExp(':', 'g')
        // if(time2[0]=="2"&&time2[1]=="4"){
        //         time2="0"+time2.substr(2,t2.length)
        //     }
        // if(parseInt(time1.replace(regex, ''), 10) <= parseInt(time2.replace(regex, ''), 10)){
        // return true;
        // } 
        // return false
        // }
        const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber })
        const questionArray = await db.collection("questions").find({}).toArray();


        // var time=currentdate.getHours()+":"+currentdate.getMinutes()+":"+currentdate.getSeconds();
       
        if (user.Questions === null || user.Questions.length == 0) {
            updateQuestionWithNull(questionArray, phoneNumber)
        }
        else{
            
            return;
        }
        // else {
        //     const dateDb = new Date(user.Questions[(user.Questions).length - 1].dateTime);
        //     const currentdate = new Date();
        //     var datePost = new Date();
        //     if (currentdate - dateDb > 86400000) {
        //         updateQuestion(questionArray, phoneNumber);
        //     }
        //     else if (datePost.getTime() - dateDb.getTime() > 86400000) {
        //         updateQuestion(questionArray, phoneNumber);
        //         res.sendStatus(200);
        //     }
        // }
    }
}