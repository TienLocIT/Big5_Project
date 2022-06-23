const client = require("../mongoDB");
const shortid = require("shortid");
const checkQuestion = require("./checkQuestion/checkQuestion");
const ObjectId = require("mongodb").ObjectId;
function getpoint(data, trait) {
    var point;
    switch (trait) {
        case 'O':
            if (data.personality[0].Openness == "High") {
                point = data.point.pointHigh;
            }
            else if (data.personality[0].Openness == "Medium") {
                point = data.point.pointMedium;
            }
            else {
                point = data.point.pointLow
            }
            break;
        case 'C':
            if (data.personality[1].Conscientious == "High") {
                point = data.point.pointHigh;
            }
            else if (data.personality[1].Conscientious == "Medium") {
                point = data.point.pointMedium;
            }
            else {
                point = data.point.pointLow
            }
            break;
        case 'E':
            if (data.personality[2].Extraversion == "High") {
                point = data.point.pointHigh;
            }
            else if (data.personality[2].Extraversion == "Medium") {
                point = data.point.pointMedium;
            }
            else {
                point = data.point.pointLow
            }
            break;
        case 'A':
            if (data.personality[3].Agreeable == "High") {
                point = data.point.pointHigh;
            }
            else if (data.personality[3].Agreeable == "Medium") {
                point = data.point.pointMedium;
            }
            else {
                point = data.point.pointLow
            }
            break;
        default:
            if (data.personality[4].Neuroticism == "High") {
                point = data.point.pointHigh;
            }
            else if (data.personality[4].Neuroticism == "Medium") {
                point = data.point.pointMedium;
            }
            else {
                point = data.point.pointLow
            }
            break;

    }
    return point;
}
function pointOpennessProbe(data) {
    var pointOpenness = 0;
    if (data.pointOpenness == null || data.pointOpenness == 0) {

    }
    else {
        if (data.pointMultyplyOpenness == null || data.pointMultyplyOpenness == 0) {
            pointOpenness += data.pointOpenness
        }
        else {
            pointOpenness += (data.pointOpenness * 2);
            for (var j = 4; j <= data.pointMultyplyOpenness; j += 2) {
                pointOpenness *= 2
            }
        }
    }
    return pointOpenness;
}
function pointConscientiousProbe(data) {
    var pointConscientious = 0;
    if (data.pointConscientious == null || data.pointConscientious == 0) {

    }
    else {
        if (data.pointMultyplyConscientious == null || data.pointMultyplyConscientious == 0) {
            pointConscientious += data.pointConscientious
        }
        else {
            pointConscientious += (data.pointConscientious * 2);
            for (var j = 4; j <= data.pointMultyplyConscientious; j += 2) {
                pointConscientious *= 2
            }
        }
    }
    return pointConscientious;
}
function pointExtraversionProbe(data) {
    var pointExtraversion = 0;
    if (data.pointExtraversion == null || data.pointExtraversion == 0) {

    }
    else {
        if (data.pointMultyplyExtraversion == null || data.pointMultyplyExtraversion == 0) {
            pointExtraversion += data.pointExtraversion
        }
        else {
            pointExtraversion += (data.pointExtraversion * 2);
            for (var j = 4; j <= data.pointMultyplyExtraversion; j += 2) {
                pointExtraversion *= 2
            }
        }
    }
    return pointExtraversion;
}
function pointAgreeableProbe(data) {
    var pointAgreeable = 0;
    if (data.pointAgreeable == null || data.pointAgreeable == 0) {

    }
    else {
        if (data.pointMultyplyAgreeable == null || data.pointMultyplyAgreeable == 0) {
            pointAgreeable += data.pointAgreeable
        }
        else {
            pointAgreeable += (data.pointAgreeable * 2);
            for (var j = 4; j <= data.pointMultyplyAgreeable; j += 2) {
                pointAgreeable *= 2;
            }
        }
    }
    return pointAgreeable;
}
function pointNeuroticismProbe(data) {
    var pointNeuroticism = 0;
    if (data.pointNeuroticism == null || data.pointNeuroticism == 0) {

    }
    else {
        if (data.pointMultyplyNeuroticism == null || data.pointMultyplyNeuroticism == 0) {
            pointNeuroticism += data.pointNeuroticism
        }
        else {
            pointNeuroticism += (data.pointNeuroticism * 2);
            for (var j = 4; j <= data.pointMultyplyNeuroticism; j += 2) {
                pointNeuroticism *= 2;
            };
        }
    }
    return pointNeuroticism;
}
function pointOpenness(user) {
    var pointOpennessAll=0;
    
    for(var i=0;i<user.Questions.length;i++){
        if(user.Questions[i].pointOpenness==null || user.Questions[i].pointOpenness==0){
            break;
        }
        else{
            if(user.Questions[i].pointMultyplyOpenness==null ||user.Questions[i].pointMultyplyOpenness==0){
                pointOpennessAll+=user.Questions[i].pointOpenness
            }
            else{
                var pointOpenness=(user.Questions[i].pointOpenness*2);
                for(var j=4;j<=user.Questions[i].pointMultyplyOpenness;j+=2){
                    pointOpenness*=2
                }
                pointOpennessAll+=pointOpenness;
            }
        }
    }
    return pointOpennessAll;
}
function pointConscientious(user) {
    var pointConscientiousAll=0;
    for(var i=0;i<user.Questions.length;i++){
        if(user.Questions[i].pointConscientious==null || user.Questions[i].pointConscientious==0){
            break;
        }
        else{
            if(user.Questions[i].pointMultyplyConscientious==null ||user.Questions[i].pointMultyplyConscientious==0){
                pointConscientiousAll+=user.Questions[i].pointConscientious
            }
            else{
                var pointConscientious=(user.Questions[i].pointConscientious*2);
                for(var j=4;j<=user.Questions[i].pointMultyplyConscientious;j+=2){
                    pointConscientious*=2
                }
                pointConscientiousAll+=pointConscientious;
            }
        }
    }
    return pointConscientiousAll;
}
function pointExtraversion(user) {
    var pointExtraversionAll=0;
    for(var i=0;i<user.Questions.length;i++){
        if(user.Questions[i].pointExtraversion==null || user.Questions[i].pointExtraversion==0){
            break;
        }
        else{
            if(user.Questions[i].pointMultyplyExtraversion==null ||user.Questions[i].pointMultyplyExtraversion==0){
                pointExtraversionAll+=user.Questions[i].pointExtraversion
            }
            else{
                var pointExtraversion=(user.Questions[i].pointExtraversion*2);
                for(var j=4;j<=user.Questions[i].pointMultyplyExtraversion;j+=2){
                    pointExtraversion*=2
                }
                pointExtraversionAll+=pointExtraversion;
            }
        }
    }
    return pointExtraversionAll;
}
function pointAgreeable(user) {
    var pointAgreeableAll=0;
    for(var i=0;i<user.Questions.length;i++){
        if(user.Questions[i].pointAgreeable==null || user.Questions[i].pointAgreeable==0){
            break;
        }
        else{
            if(user.Questions[i].pointMultyplyAgreeable==null ||user.Questions[i].pointMultyplyAgreeable==0){
                pointAgreeableAll+=user.Questions[i].pointAgreeable
            }
            else{
                var pointAgreeable=(user.Questions[i].pointAgreeable*2);
                for(var j=4;j<=user.Questions[i].pointMultyplyAgreeable;j+=2){
                    pointAgreeable*=2
                }
                pointAgreeableAll+=pointAgreeable;
            }
        }
    }
    return pointAgreeableAll;
}
function pointNeuroticism(user) {
    var pointNeuroticismAll = 0;
    for (var i = 0; i < user.Questions.length; i++) {
        if (user.Questions[i].pointNeuroticism == null || user.Questions[i].pointNeuroticism == 0) {
            continue;
        }
        else {
            if (user.Questions[i].pointMultyplyNeuroticism == null || user.Questions[i].pointMultyplyNeuroticism == 0) {
                pointNeuroticismAll += user.Questions[i].pointNeuroticism
            }
            else {
                var pointNeuroticism = (user.Questions[i].pointNeuroticism * 2);
                for (var j = 4; j <= user.Questions[i].pointMultyplyNeuroticism; j += 2) {
                    pointNeuroticism *= 2
                };
                return pointNeuroticism;
                // pointNeuroticismAll +=user.Questions[i].pointNeuroticism*user.Questions[i].pointMultyplyNeuroticism;
            }
        }
    }
    return pointNeuroticismAll;
}
function getMaxTrait(arrayPoint) {
    var dem = 0;
    var max = arrayPoint[0];
    for (var i = 0; i < arrayPoint.length; i++) {
        if (arrayPoint[i] > max) {
            max = arrayPoint[i];
            dem = i;
        }
    }
    return dem;
}

function getMaxTraitToString(arrayPoint) {
    var dem = 0;
    var max = arrayPoint[0].point;
    for (var i = 0; i < arrayPoint.length; i++) {
        if (arrayPoint[i].point > max) {
            max = arrayPoint[i].point;
            dem = i;
        }
    }
    return arrayPoint[dem].personality;
}
function wasHasPoint(pointO, PointC, PointE, PointA, PointN) {
    if (pointO == 0 && PointC == 0 && PointE == 0 && PointA == 0 && PointN == 0) {
        return false;
    }
    return true;
}

module.exports = {
    adminHome: async (req, res) => {
        // res.setHeader("Content-Type", "text/html");
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = await dbs.db("green_big_5");
        const sumUser = await db.collection("users").find({}).toArray();
        const sumUserFilter = sumUser.filter(user => user.isClone === false || user.isClone === undefined)
        const sumQuestion = await db.collection("questions").find({}).count();
        res.render("index", {
            "sumUser": sumUserFilter.length,
            "sumQuestion": sumQuestion
        })
    },
    getTraitNumberUser: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = await dbs.db("green_big_5");
        var numberquestionO = 0;
        var numberquestionC = 0;
        var numberquestionE = 0;
        var numberquestionA = 0;
        var numberquestionN = 0;
        const questions = await db.collection("questions").find({}).toArray();
        const users = await db.collection("users").find({}).toArray();
        const usersFilter = users.filter(user => user.isClone === false || user.isClone === undefined)
        var arrayUser = [0, 0, 0, 0, 0];
        usersFilter.forEach((user) => {
            const arrayPointUser = [pointOpenness(user), pointConscientious(user), pointExtraversion(user), pointAgreeable(user), pointNeuroticism(user)];
            const dem = getMaxTrait(arrayPointUser);
            arrayUser[dem]++;
        })
        questions.forEach(question => {
            switch (checkQuestion.check(question)) {
                case 'Openness':
                    numberquestionO++;
                    break;
                case 'Conscientious':
                    numberquestionC++;
                    break;
                case 'Extraversion':
                    numberquestionE++;
                    break;
                case 'Agreeable':
                    numberquestionA++;
                    break;
                default:
                    numberquestionN++;
                    break;
            }
        })
        var arrayQuestion = [numberquestionO, numberquestionC, numberquestionE, numberquestionA, numberquestionN];
        const dataResponse = {
            "totalUser": usersFilter.length,
            "arrayUser": arrayUser,
            "arrayQuestion": arrayQuestion,
        }
        res.status(200).json(dataResponse);
    },
    adminUserHome: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = await dbs.db("green_big_5");
        const id = req.query.id;
        var haveUser = false;
        var haveAccuracy = true;
        var dataprobes = [];
        var dataAccuracys = [];
        var pointOProbe, pointCProbe, pointEProbe, pointAProbe, pointNProbe;
        var pointOAccuracy, pointCAccuracy, pointEAccuracy, pointAAccuracy, pointNAccuracy;
        var name, numberQuestionAnswer = 0, phoneNumber, trait, numberQuestion = 0;
        if (id == undefined) {
            haveUser = false;
        }
        else {
            haveUser = true;
            const user = await db.collection("users").findOne({ "_id": ObjectId(id) });
            name = user.fullName;
            phoneNumber = user.phoneNumber;
            const arrayPersonality = [
                {
                    "point": pointOpenness(user),
                    "personality": "Openness"
                },
                {
                    "point": pointConscientious(user),
                    "personality": "Conscientious"
                },
                {
                    "point": pointExtraversion(user),
                    "personality": "Extraversion"
                },
                {
                    "point": pointAgreeable(user),
                    "personality": "Agreeable"
                },
                {
                    "point": pointNeuroticism(user),
                    "personality": "Neuroticism"
                },
            ]
            trait = getMaxTraitToString(arrayPersonality);
            user.Questions.forEach(data => {
                for (var i = 0; i < data.questionsDate.length; i++) {
                    if (data.questionsDate[i].answer != null) {
                        numberQuestionAnswer++;
                    }
                    numberQuestion++;
                }
            })
            for (var i = 0; i < user.Questions[0].questionsDate.length; i++) {
                var question, answer, pointO, pointC, pointE, pointA, pointN;
                console.log(user.Questions[0].questionsDate[i].question)
                var questionFromDB = await db.collection("questions").findOne({ "question": user.Questions[0].questionsDate[i].question });
                question = user.Questions[0].questionsDate[i].question;
                if (user.Questions[0].questionsDate[i].answer == null) {
                    answer = 'N/A';
                }
                else {
                    answer = user.Questions[0].questionsDate[i].answer;
                }
                pointO = {
                    'point': getpoint(questionFromDB, "O"),
                    'trait': questionFromDB.personality[0].Openness
                }
                pointC = {
                    'point': getpoint(questionFromDB, "C"),
                    'trait': questionFromDB.personality[1].Conscientious
                }
                pointE = {
                    'point': getpoint(questionFromDB, "E"),
                    'trait': questionFromDB.personality[2].Extraversion
                }
                pointA = {
                    'point': getpoint(questionFromDB, "A"),
                    'trait': questionFromDB.personality[3].Agreeable
                }
                pointN = {
                    'point': getpoint(questionFromDB, "N"),
                    'trait': questionFromDB.personality[4].Neuroticism
                }
                var dataProbe = {
                    'question': question,
                    'answer': answer,
                    'pointO': pointO,
                    'pointC': pointC,
                    'pointE': pointE,
                    'pointA': pointA,
                    'pointN': pointN
                }
                dataprobes.push(dataProbe);
            }
            if (user.Questions[1] != null) {
                haveAccuracy = true;
                for (var i = 1; i < user.Questions.length; i++) {
                    for (var j = 0; j < user.Questions[i].questionsDate.length; j++) {
                        var question, answer, pointO, pointC, pointE, pointA, pointN;
                        var questionFromDB = await db.collection("questions").findOne({ "question": user.Questions[i].questionsDate[j].question });
                        question = user.Questions[i].questionsDate[j].question;
                        if (user.Questions[i].questionsDate[j].answer == null) {
                            answer = 'N/A'
                        }
                        else {
                            answer = user.Questions[i].questionsDate[j].answer;
                        }
                        pointO = {
                            'point': getpoint(questionFromDB, "O"),
                            'trait': questionFromDB.personality[0].Openness
                        }
                        pointC = {
                            'point': getpoint(questionFromDB, "C"),
                            'trait': questionFromDB.personality[1].Conscientious
                        }
                        pointE = {
                            'point': getpoint(questionFromDB, "E"),
                            'trait': questionFromDB.personality[2].Extraversion
                        }
                        pointA = {
                            'point': getpoint(questionFromDB, "A"),
                            'trait': questionFromDB.personality[3].Agreeable
                        }
                        pointN = {
                            'point': getpoint(questionFromDB, "N"),
                            'trait': questionFromDB.personality[4].Neuroticism
                        }
                        var dataAccuracy = {
                            'question': question,
                            'answer': answer,
                            'pointO': pointO,
                            'pointC': pointC,
                            'pointE': pointE,
                            'pointA': pointA,
                            'pointN': pointN
                        }
                        dataAccuracys.push(dataAccuracy);
                    }
                }
            }
            pointOProbe = pointOpennessProbe(user.Questions[0])
            pointCProbe = pointConscientiousProbe(user.Questions[0])
            pointEProbe = pointExtraversionProbe(user.Questions[0])
            pointAProbe = pointAgreeableProbe(user.Questions[0])
            pointNProbe = pointNeuroticismProbe(user.Questions[0]);
            user.Questions.shift();
            pointOAccuracy = pointOpenness(user);
            pointCAccuracy = pointConscientious(user);
            pointEAccuracy = pointExtraversion(user);
            pointAAccuracy = pointAgreeable(user);
            pointNAccuracy = pointNeuroticism(user);
        }
        const dataRender = {
            'haveUser': haveUser,
            'haveAccuracy': haveAccuracy,
            'pointOProbe': pointOProbe,
            'pointCProbe': pointCProbe,
            'pointEProbe': pointEProbe,
            'pointAProbe': pointAProbe,
            'pointNProbe': pointNProbe,
            'dataprobes': dataprobes,
            'dataAccuracys': dataAccuracys,
            'pointOAccuracy': pointOAccuracy,
            'pointCAccuracy': pointCAccuracy,
            'pointEAccuracy': pointEAccuracy,
            'pointAAccuracy': pointAAccuracy,
            'pointNAccuracy': pointNAccuracy,
            'name': name,
            'numberQuestion': numberQuestion,
            'numberQuestionAnswer': numberQuestionAnswer,
            'phone': phoneNumber,
            'trait': trait
        }

        res.render("pages/user", {
            'dataRender': dataRender
        });
    },
    adminQuestionHome: (req, res) => {
        res.setHeader("Content-Type", "text/html");
        res.render("pages/question");
    },
    adminGetUsers: async (req, res) => {
        const dbs=await client.connect().catch(err=>{
            console.log(err);
        })
        const db=dbs.db("green_big_5");
        const trait=req.query.trait;
        var arrayUser=[];
        var arrayUserOpenness=[];
        var arrayUserConscientious=[];
        var arrayUserExtraversion=[];
        var arrayUserAgreeable=[];
        var arrayUserNeuroticism=[];
        const users= await db.collection("users").find({}).toArray();
        users.forEach(data=>{
            var body={
                "_id":data._id,
                "userName":data.displayName,
                "arrayPersonality":[
                {
                    "point":pointOpenness(data),
                    "personality":"Openness"
                },
                {
                    "point":pointConscientious(data),
                    "personality":"Conscientious"
                },
                {
                    "point":pointExtraversion(data),
                    "personality":"Extraversion"
                },
                {
                    "point":pointAgreeable(data),
                    "personality":"Agreeable"
                },
                {
                    "point":pointNeuroticism(data),
                    "personality":"Neuroticism"
                },
            ],
            isClone:data.isClone === true ?true :false
        }
            arrayUser.push(body);
            switch(getMaxTraitToString(body.arrayPersonality)){
                case "Openness":
                    arrayUserOpenness.push(body);
                    break;
                case "Conscientious":
                    arrayUserConscientious.push(body);
                    break;
                case "Extraversion":
                    arrayUserExtraversion.push(body);
                    break;
                case "Agreeable":
                    arrayUserAgreeable.push(body);
                    break;
                default: 
                    arrayUserNeuroticism.push(body);
                    break;
            }
        })
        switch(trait){
            case 'All':
                res.status(200).json(arrayUser);
                break;
            case 'Openness':
                res.status(200).json(arrayUserOpenness);
                break;
            case 'Conscientious':
                res.status(200).json(arrayUserConscientious);
                break;
            case 'Extraversion':
                res.status(200).json(arrayUserExtraversion);
                break;
            case 'Agreeable':
                res.status(200).json(arrayUserAgreeable);
                break;
            default:
                res.status(200).json(arrayUserNeuroticism);
                break;
            
        }
    },  
    adminGetQuestions: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        if (!dbs) {
            console.log("error")
        }
        else {
            const db = await dbs.db("green_big_5");
            // const query=req.query.trait;
            // var questions;
            try {
                const questions = await db.collection("questions").find({}).toArray();
                return res.status(200).json({ "Status": "Success", questions });
                // switch(query){
                //     case "Openness":
                //         questions=await db.collection("questions").find({personality:{
                //             $elemMatch:{"Openness":"High"}
                //          }
                //         }).toArray();

                //         break;
                //     case "Conscientious":
                //         questions=await db.collection("questions").find({personality:{
                //                 $elemMatch:{"Conscientious":"High"}
                //              }
                //             }).toArray();

                //             break;
                //     case "Extraversion":
                //         questions=await db.collection("questions").find({personality:{
                //             $elemMatch:{"Extraversion":"High"}
                //          }
                //         }).toArray();

                //         break;
                //     case "Agreeable":
                //         questions=await db.collection("questions").find({personality:{
                //             $elemMatch:{"Agreeable":"High"}
                //          }
                //         }).toArray();

                //         break;
                //     case "Neuroticism":
                //         questions=await db.collection("questions").find({personality:{
                //             $elemMatch:{"Neuroticism":"High"}
                //          }
                //         }).toArray();
                //         break;
                //     default:
                //         questions=await db.collection("questions").find({}).toArray();
                //         break;
                // }
            }
            catch {
                err =>
                    res.status(200).json({ "Status": "Error", "Notification": "Error" })
            }
        }
    },
    adminGetQuestionsByBig5EnvIndcator: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        if (!dbs) {
            return res.status(200).json({ "Status": "Error", "Notification": "My database is very hot" });
        }
        else {
            const db = await dbs.db("green_big_5");
            try {
                const questionsByBig5IndicatorEnv = await db.collection("questions").find({ big5EnvIndicator: req.query.big5EnvIndicator }).toArray();
                return res.status(200).json({ Status: "Success", questions: questionsByBig5IndicatorEnv })
            }
            catch {
                err =>
                    res.status(200).json({ Status: "Erorr", Notification: err });
            }
        }
    },
    adminGetQuestionsByIndicator: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        if (!dbs) {
            console.log("error")
        }
        else {
            const db = await dbs.db("green_big_5");
            const query = req.query.big5envindicator;
            var questions;
            try {
                questions = await db.collection("questions").find({ big5envindicator: query }).toArray();
            }
            catch (err) {
                res.status(200).json({ "Notification": "Error" });
            } finally {
                res.status(200).json({ "Notification": "Success", questions });
            }
        }
    },
    adminLogout: (req, res) => {
        res.clearCookie("admin_id");
        res.redirect("/login")
    },
    adminSendQuestion: async (req, res) => {
        // res.setHeader("Content-Type", "text/html");
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Error when connnect" });
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
        const idQuestion = req.body.id;
        const arrayTrait = req.body.arrayTrait;
        const questionObject = await db.collection("questions").findOne({ "_id": ObjectId(idQuestion) });
        const users = await db.collection("users").find({}).toArray();
        var arrayUser = [];

        users.forEach(data => {
            var arrayPointObject = [
                {
                    "point": pointOpenness(data),
                    "personality": "Openness"
                },
                {
                    "point": pointConscientious(data),
                    "personality": "Conscientious"
                },
                {
                    "point": pointExtraversion(data),
                    "personality": "Extraversion"
                },
                {
                    "point": pointAgreeable(data),
                    "personality": "Agreeable"
                },
                {
                    "point": pointNeuroticism(data),
                    "personality": "Neuroticism"
                },
            ]
            // console.log(getMaxTraitToString(arrayPointObject))
            if (arrayTrait.includes(getMaxTraitToString(arrayPointObject)) == true) {
                if (wasHasPoint(pointOpenness(data), pointConscientious(data), pointExtraversion(data), pointAgreeable(data), pointNeuroticism(data)) == false) {

                }
                else {
                    arrayUser.push(data);
                }

            }
        })
        arrayUser.forEach(data => {
            // if(data.defindQuestion==null){
            //     updateDefineQuestionWithNull(questionObject.question,data.phoneNumber);
            // }
            // else{
            //     updateDefineQuestion(questionObject.question,data.phoneNumber);
            // }
            updateQuestion(data.phoneNumber, questionObject.question);
        })
        res.status(200).json({ "erorr": "no" });
    },
    adminAddQuestionPost: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const question = req.body.question;
        const Openness = req.body.Openness;
        const Conscientious = req.body.Conscientious;
        const Extraversion = req.body.Extraversion;
        const Agreeable = req.body.Agreeable;
        const Neuroticism = req.body.Neuroticism;
        const point = req.body.point;
        const big5EnvIndicator = req.body.big5EnvIndicator;
        const db = dbs.db("green_big_5");

        try {
            const questionFind = await db.collection("questions").findOne({ "question": question.trim() });
            if (questionFind) {
                return res.status(200).json({ "error": "yes", Notification: "Question is concide" });
            }
            else {
                await db.collection("questions").insertOne({
                    "question": question.trim(),
                    "big5EnvIndicator": big5EnvIndicator.trim(),
                    "personality": [
                        { "Openness": Openness },
                        { "Conscientious": Conscientious },
                        { "Extraversion": Extraversion },
                        { "Agreeable": Agreeable },
                        { "Neuroticism": Neuroticism }
                    ],
                    point: point
                })
                return res.status(200).json({ "error": "no" });
            }


            // res.redirect("/listQuestion?key=MBe2IdLb-9dhasuidad02316156156adasf231223423523sdg")
        }
        catch (err) { return res.status(200).json({ "error": err }) }
    },
    adminAddBig5EnvIndicatorPost: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            return res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const db = dbs.db("green_big_5");
        try {
            await db.collection("indicators_dont_have_question").insertMany(
                req.body.big5EnvIndicatorList)
            return res.status(200).json({ "Notification": "Success" })
        }
        catch {
            return res.status(200).json({ "Notification": "Erorr" })
        }
    },
    adminDeleteAllBig5EnvIndicator: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const db = dbs.db("green_big_5");
        await db.collection("indicators_dont_have_question").remove({});
    },
    adminCheckIndicator: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            return res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const db = dbs.db("green_big_5");
        try {
            const big5EnvIndicator = await db.collection("indicators_dont_have_question").findOne({
                big5EnvIndicator: req.body.big5EnvIndicator
            });
            if (!big5EnvIndicator) {
                return res.status(200).json({ "Status": "Success", "Notification": "Successs" });
            }
            else {
                return res.status(200).json({ "Status": "Error", "Notification": "Big5 env indicator was have", typeCheckWhen: "Save Question" });
            }
        }
        catch { err => res.status(200).json({ "Status": "Error", "Notification": err }); }
    },
    adminGetIndicator: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            return res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const db = dbs.db("green_big_5");
        try {
            const big5EnvIndicatorList = await db.collection("indicators_dont_have_question").find({}).toArray();
            return res.status(200).json({ "Status": "Success", big5EnvIndicatorList });
        }
        catch { err => res.status(200).json({ "Status": "Error", "Notification": err }); }
    },
    adminCheckAndDelteBig5EnvIndicator: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            return res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const db = dbs.db("green_big_5");
        try {
            const big5EnvIndicator = await db.collection("indicators_dont_have_question").findOne({ big5EnvIndicator: req.body.big5EnvIndicator });
            if (big5EnvIndicator) {
                await db.collection("indicators_dont_have_question").deleteOne({
                    big5EnvIndicator: req.body.big5EnvIndicator
                })
                return res.status(200).json({ "Status": "Success" });
            }
            else {
                return res.status(200).json({ "Status": "Success" });
            }
        }
        catch (err) {
            return res.status(200).json({ "Status": "Error" });
        }

    },
    adminGetKeyword: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            return res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const db = dbs.db("green_big_5");
        try {
            const keywordList = await db.collection("keywords").find({}).toArray();
            return res.status(200).json(keywordList);
        }
        catch { err => res.status(400).json(err); }
    },
    adminGetAddress: async (req,res,next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            return res.status(200).json({ "error": "Maybe your network is hot" });
        })
        const db = dbs.db("green_big_5");
        const id =req.query.id;
        try {
            const user = await db.collection("users").findOne({_id:ObjectId(id)});
            return res.status(200).json(user.addressMap);
        }
        catch { err => res.status(400).json(err); }
    }
}