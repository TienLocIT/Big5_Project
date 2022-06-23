const client = require("../../mongoDB");
const bcrypt = require("bcrypt");
const shortid = require("shortid");
const checkQuestionConcide = require("../checkQuestionConcide/concideQuestion");
module.exports = {
    register: async (fullName, phoneNumber, displayName, password, age, location, gender) => {
        function randomNumber(min, max) {
            return Math.random() * (max - min) + min;
        }
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5")
        const questionArray = await db.collection("questions").find({}).toArray();
        const saltRounds = 10
      
        async function updateQuestionWithNull(questionArray, phoneNumber) {
            var id = shortid.generate();
            var questions = [];
            var QuestionsAll = [];
            // while(questions.length<5){
            //     var random=Math.floor(Math.random()*questionArray.length);
            //     var questionContent=questionArray[random].question;
            //     if(questions.length==0){
            //             var questionBody={
            //                 "question":questionArray[random].question
            //             }
            //             questions.push(questionBody);

            //     }
            //     else{
            //         if(checkQuestionConcide.concideQuestion(questionContent,questions)==false){
            //             var questionBody={
            //                 "question":questionArray[random].question
            //             }
            //             questions.push(questionBody);

            //         }
            //     }
            // }
            for (var i = 0; i < 5; i++) {
                var questionBody = {
                    question: questionArray[i].question
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
        // var phoneNumber=req.body.phoneNumber;
        // var digits="0123456789";
        // var OTP="";

        // var inputInformation;
        const keySeceret = shortid.generate();
        var body = {
            "phoneNumber": phoneNumber,
            "fullName": fullName,
            "displayName": displayName,
            "gender": gender,
            "age": age,
            "location": location,
            "addressMap": [],
            "keySeceret": keySeceret,
        }
        await db.collection("users").insertOne(body);
        updateQuestionWithNull(questionArray, phoneNumber);
        await bcrypt.genSalt(saltRounds, (err, salt) => {
            bcrypt.hash(password, salt, (err, hash) => {
                db.collection("users").updateOne({ "phoneNumber": phoneNumber }, { $set: { "password": hash } });
            })
        })
        // res.status(200).json({
        //     "error":"No"
        // })


        // for(var i=0;i<4;i++){
        //     OTP+=digits[Math.floor(Math.random()*10)];
        //  }
        //  var userWithPhoneNumber=await db.collection("users").findOne({"phoneNumber":phoneNumber});
        //  if(userWithPhoneNumber){
        //    res.status(200).json({
        //        "error":"Phone Number is coincide",
        //    });
        //  }
        //  else{
        //      await db.collection("users").insertOne()
        //  }
        //  const bodyRegister={
        //      "phoneNumber":phoneNumber,
        //      "OTP":OTP,
        //      "inputInformation":inputInformation
        //  }  
    },
}