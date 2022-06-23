const client = require("../mongoDB");
const shortid = require("shortid");
const { ObjectId } = require("mongodb");
module.exports = {
    expertSendQuestionToAll: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        async function updateQuestion(phoneNumber, question) {
            var questions = [{ "question": question }];
            var QuestionsAll = [];
            var id = shortid.generate();
            var currentdate = new Date();

            // var date=(currentdate.getMonth()+1) + "/"+ currentdate.getDate()  + "/" + currentdate.getFullYear()
            // var time=currentdate.getHours()+":"+currentdate.getMinutes()+":"+currentdate.getSeconds();

            var body = {
                "id": id,
                "dateTime": currentdate,
                "questionsDate": questions,
            }
            QuestionsAll.push(body);
            await db.collection("users").updateOne({ "phoneNumber": phoneNumber }, {
                $push: {
                    "Questions": {
                        $each: QuestionsAll
                    }
                }
            });
        }
        // async function updateDefineQuestionWithNull(question,phoneNumber){
        //     var id=shortid.generate();
        //     var currentdate = new Date(); 
        //     var QuestionsAll=[];
        //         // var date=(currentdate.getMonth()+1) + "/"+ currentdate.getDate() + "/" + currentdate.getFullYear();
        //         // var time=currentdate.getHours()+":"+currentdate.getMinutes()+":"+currentdate.getSeconds();

        //         var body={
        //             "id":id,
        //             "dateTime":currentdate,
        //             "question":question,
        //         }
        //         QuestionsAll.push(body);
        //         await db.collection("users").updateOne({"phoneNumber":phoneNumber},{ $set: {"defindQuestion":QuestionsAll}});
        // }
        // async function updateDefineQuestion(question,phoneNumber){
        //     var id=shortid.generate();
        //     var currentdate = new Date(); 
        //     var QuestionsAll=[];
        //         // var date=(currentdate.getMonth()+1) + "/"+ currentdate.getDate() + "/" + currentdate.getFullYear();
        //         // var time=currentdate.getHours()+":"+currentdate.getMinutes()+":"+currentdate.getSeconds();

        //         var body={
        //             "id":id,
        //             "dateTime":currentdate,
        //             "question":question,
        //         }
        //         QuestionsAll.push(body);
        //         await db.collection("users").updateOne({"phoneNumber":phoneNumber},{ $push: {"defindQuestion":{
        //             $each:QuestionsAll
        //         }}});
        // }
        try {

            const question = req.body.question;
            const key = req.body.key;
            const questionObject = await db.collection("questions").findOne({ "question": question });
            const users = await db.collection("users").find({}).toArray();
            if (key !== process.env.KEY_EXPERT_SEND_QUESTION) {
                return res.status(200).json({ "erorr": "Access-Denined" });
            }
            else {
                await db.collection("questions").updateOne({ "question": question }, {
                    $set: {
                        numberUser: users.length
                    }
                })
                users.forEach(data => {
                    updateQuestion(data.phoneNumber, questionObject.question);
                })
                return res.status(200).json({ "erorr": "no" });
            }
        }
        catch (err) {
            return res.status(400).json({ "erorr": err });
        }

    },
    expertGetQuestion: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        const key = req.query.key;
        const question = req.query.question;
        try {
            if (key !== process.env.KEY_EXPERT_GET_QUESTION) {
                return res.status(200).json({ "err": "Acess-denined" });
            }
            const questionFind = await db.collection("questions").findOne({ question });
            return res.status(200).json({ "err": "no", question: questionFind });
        }
        catch (err) {
            return res.status(400).json({ "err": err });
        }
    },
    testAPI: (req, res, next) => {
        return res.status(200).send({ "Output_modal": "No" })
    },
    getPointForMachineLearning: async (req, res, next) => {
        function pointOpenness(pointOpenness, pointMultyplyOpenness) {
            var pointOpennessReturn;
            if (pointMultyplyOpenness !== undefined) {
                pointOpennessReturn = pointOpenness * pointMultyplyOpenness;
            }
            else {
                pointOpennessReturn = pointOpenness;
            }
            return pointOpennessReturn;
        }
        function pointConscientious(pointConscientious, pointMultyplyConscientious) {
            var pointConscientiousReturn;
            if (pointMultyplyConscientious !== undefined) {
                pointConscientiousReturn = pointConscientious * pointMultyplyConscientious;
            }
            else {
                pointConscientiousReturn = pointConscientious;
            }
            return pointConscientiousReturn;
        }
        function pointExtraversion(pointExtraversion, pointMultyplyExtraversion) {
            var pointExtraversionReturn;
            if (pointMultyplyExtraversion !== undefined) {
                pointExtraversionReturn = pointExtraversion * pointMultyplyExtraversion;
            }
            else {
                pointExtraversionReturn = pointExtraversion;
            }
            return pointExtraversionReturn;
        }
        function pointAgreeable(pointAgreeable, pointMultyplyAgreeable) {
            var pointAgreeableReturn;
            if (pointMultyplyAgreeable !== undefined) {
                pointAgreeableReturn = pointAgreeable * pointMultyplyAgreeable;
            }
            else {
                pointAgreeableReturn = pointAgreeable;
            }
            return pointAgreeableReturn;
        }
        function pointNeuroticism(pointNeuroticism, pointMultyplyNeuroticism) {
            var pointNeuroticismReturn;
            if (pointMultyplyNeuroticism !== undefined) {
                pointNeuroticismReturn = pointNeuroticism * pointMultyplyNeuroticism;
            }
            else {
                pointNeuroticismReturn = pointNeuroticism;
            }
            return pointNeuroticismReturn;
        }
        function randomNumber(min, max) {
            return Math.random() * (max - min) + min;
        }
        const question = req.body.question;
        const gender = Number.parseInt(req.body.gender);
        const location = Number.parseInt(req.body.location);
        const age = Number.parseInt(req.body.age);
        const key = req.body.key;
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        try {
            let pointO = 0;
            let pointC = 0;
            let pointE = 0;
            let pointA = 0;
            let pointN = 0;
            const users = await db.collection("users").find({ gender, location, age }).toArray();
            if (key !== process.env.KEY_EXPERT_GET_POINT) {
                return res.status(200).json({ "err": "Access-denine" });
            }
            else {
                if (location === null || age === null || gender === null) {
                    return res.status(400).json({ "err": "Required body" });
                }
                else {
                    for (var i = 0; i < users.length; i++) {
                        for (var j = 1; j < users[i].Questions.length; j++) {
                            if (users[i].Questions[j].questionsDate[0].question === question) {
                                if (users[i].Questions[j].questionsDate[0].answer) {
                                    pointO += pointOpenness(users[i].Questions[j].pointOpenness, users[i].Questions[j].pointMultyplyOpenness);
                                    pointC += pointConscientious(users[i].Questions[j].pointConscientious, users[i].Questions[j].pointMultyplyConscientious);
                                    pointE += pointExtraversion(users[i].Questions[j].pointExtraversion, users[i].Questions[j].pointMultyplyExtraversion);
                                    pointA += pointAgreeable(users[i].Questions[j].pointAgreeable, users[i].Questions[j].pointMultyplyAgreeable);
                                    pointN += pointNeuroticism(users[i].Questions[j].pointNeuroticism, users[i].Questions[j].pointMultyplyNeuroticism);
                                }
                            }
                        }
                    }
                }

                const body = [Number.parseInt(age), Number.parseInt(gender), Number.parseInt(location), randomNumber(1, 1000000), pointO, pointC, pointE, pointA, pointN]
                return res.status(200).json(body);
            }
        }
        catch (err) {
            return res.status(400).json(err);
        }

    },
    cloneDocument: async (req, res, next) => {
        const age = req.body.age;
        const location = req.body.location;
        const gender = req.body.location;
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        // try {

        //     String.prototype.shuffle = function () {
        //         var a = this.split(""),
        //             n = a.length;

        //         for (var i = n - 1; i > 0; i--) {
        //             var j = Math.floor(Math.random() * (i + 1));
        //             var tmp = a[i];
        //             a[i] = a[j];
        //             a[j] = tmp;
        //         }
        //         return a.join("");
        //     }
        //     let users = [];
        //     for (var i = 0; i < 60; i++) {
        //         let phoneNumber = "0" + "123456789123456789".shuffle().slice(0, 10);
        //         var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
        //         while (user !== null) {
        //             phoneNumber = "0" + "123456789123456789".shuffle().slice(0, 10);
        //             user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
        //         }
        //         const body =/** 
        //         * Paste one or more documents here
        //         */
        //         {
                   
        //             "phoneNumber":phoneNumber,
        //             "fullName":`Simualator${i}`,
        //             "displayName":`Simualator${i}`,
        //             "gender":1,
        //             "age":2,
        //             "location":1,
        //             "addressMap":[
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "33 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "33 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "33 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam",
        //                "19 Chơn Tâm 8, Đà Nẵng 550000, Việt Nam"
        //             ],
        //             "keySeceret":"MnO03sceg",
        //             "Questions":[
        //                {
        //                   "id":"FXAbhkQFDZ",
        //                   "dateTime":{
                             
        //                   },
        //                   "questionsDate":[
        //                      {
        //                         "question":"Is there a rice field in texas?",
        //                         "answer":"Yes"
        //                      },
        //                      {
        //                         "question":"Are you comfortable having a forest fire?",
        //                         "answer":"Yes"
        //                      },
        //                      {
        //                         "question":"Do trees bring value to you?",
        //                         "answer":"No"
        //                      },
        //                      {
        //                         "question":"Is there a bad party in the world?",
        //                         "answer":"Yes"
        //                      },
        //                      {
        //                         "question":"Do you care about forest fires?",
        //                         "answer":"Yes"
        //                      }
        //                   ],
        //                   "pointAgreeable":60,
        //                   "pointConscientious":60,
        //                   "pointOpenness":60,
        //                   "pointExtraversion":70,
        //                   "pointNeuroticism":70,
        //                   "pointMultyplyNeuroticism":2,
        //                   "pointMultyplyAgreeable":2,
        //                   "pointMultyplyConscientious":2,
        //                   "pointMultyplyExtraversion":2
        //                },
        //                {
        //                   "id":"QESvd4Js-T",
        //                   "dateTime":{
                            
        //                   },
        //                   "questionsDate":[
        //                      {
        //                         "question":"Are you a person who buys animal feed?",
        //                         "answer":"Yes"
        //                      }
        //                   ],
        //                   "pointConscientious":10,
        //                   "pointOpenness":10,
        //                   "pointNeuroticism":10,
        //                   "pointAgreeable":10,
        //                   "pointExtraversion":15
        //                },
        //                {
        //                   "id":"WWhUTgRdc0",
        //                   "dateTime":{
                            
        //                   },
        //                   "questionsDate":[
        //                      {
        //                         "question":"Is there such thing as a process processor?",
        //                         "answer":"No"
        //                      }
        //                   ],
        //                   "pointOpenness":10,
        //                   "pointConscientious":10,
        //                   "pointExtraversion":10,
        //                   "pointNeuroticism":20,
        //                   "pointAgreeable":10
        //                },
        //                {
        //                   "id":"zFDqt_cpQS",
        //                   "dateTime":{
                            
        //                   },
        //                   "questionsDate":[
        //                      {
        //                         "question":"Is there a rice field in ww2?",
        //                         "answer":"No"
        //                      }
        //                   ],
        //                   "pointNeuroticism":10,
        //                   "pointConscientious":10,
        //                   "pointOpenness":20,
        //                   "pointExtraversion":10,
        //                   "pointAgreeable":10
        //                }
        //             ],
        //             "password":"$2b$10$dCM5fek6OJahLByezXegW.K0mIgX55yMQ2VDnNQMkkgi2aWKfDhMi",
        //             "seen":false,
        //             isClone:true
        //          }
        //         users.push(body);
        //     }
        //     await db.collection("users").insertMany(users);
        try{
            const users = await db.collection("users").find({}).toArray();
            users.forEach(async (user) => {
                console.log(user._id)
                await db.collection("users").updateOne({
                    _id:ObjectId(user._id)
                },{
                    $set:{
                        isClone:true
                    }
                })
            })
            return res.status(200).send("Good");
            
        }
        catch (err) {
            console.log(err);
            return res.status(400).send(err);
        }

    }
}