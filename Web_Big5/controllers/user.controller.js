const client = require("../mongoDB");
const bcrypt = require("bcrypt");
const shortid = require("shortid");
const updateQuestionPersonality = require("./UpdateQuestionPersonality/UpdateQuestionPersonality");
const startUserDoRegister = require("./startUserDo/register");
const itereactiveQuestionPost = require("./interactiveQuestion/postQuestions");
const itereactiveQuestionPostAnswer = require("./interactiveQuestion/postAnswerQuestion");
const changeInformation = require("./userDo/changeInformation");
module.exports = {
    register: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            return res.status(200).json({ "error": "Maybe your network it hot" });
        })
        try {
            
            const db = dbs.db("green_big_5")
            const fullName = req.body.fullName;
            const phoneNumber = req.body.phoneNumber;
            const age = req.body.age;
            const location = req.body.location;
            const gender = req.body.gender;
            const displayName = req.body.displayName;
            const password = req.body.password;
            const key = req.body.key;
          
            //const inputInformation;
            if (key != process.env.Key) {
                res.status(400);
            }
            else {
                const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
                if (user) {
                    res.status(200).json({
                        "error": "Phone Number is Concide"
                    })
                }
                else {
                    startUserDoRegister.register(fullName, phoneNumber, displayName, password, age, location, gender);
                    return res.status(200).json({
                        "error": "No"
                    })
                }
            }
        }
        catch (err) {
            return res.status(400).json({ err })
        }

        // for(var i=0;i<4;i++){
        //     OTP+=digits[Math.floor(Math.random()*10)];
        //  }
        // const userWithPhoneNumber=await db.collection("users").findOne({"phoneNumber":phoneNumber});
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
    login: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Maybe your network it hot" });
        })
        const db = dbs.db("green_big_5")
        const phoneNumber = req.body.phoneNumber;
        const key = req.body.key;
        const password = req.body.password;
        if (key != process.env.Key) {
            res.status(400);
        }
        else {

            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user) {
                bcrypt.compare(password, user.password, (err, result) => {
                    if (result) {
                        res.status(200).json({
                            "error": "No"
                        })
                    }
                    else {
                        res.status(200).json({
                            "error": "Wrong password"
                        })
                    }
                })
            }
            else {
                res.status(200).json({
                    "error": "Wrong phone number"
                })
            }
        }
    },
    postQuestions: async (req, res) => {
        const phoneNumber = req.body.phoneNumber;
        const key = req.body.key;
        if (key != process.env.keyGetQuestions) {
            res.status(400);
        }
        else {
          
            itereactiveQuestionPost.postQuestions(phoneNumber);
            res.status(200).send({"err":"no"});
        }
    },
    getUser: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Maybe your network it hot" });
        })
        const db = dbs.db("green_big_5")
        const phoneNumber = req.query.phoneNumber;
        const key = req.query.key;
        if (key != process.env.KeyGetUser) {
            res.status(400);
        }
        else {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            res.status(200).json(user);
        }
    },
    postAnswerQuestion: async (req, res) => {
        const phoneNumber = req.body.phoneNumber;
        const questionPost = req.body.question;
        const answer = req.body.answer;
        const key = req.body.key;
        const id = req.body.id;
        const dem = parseInt(req.body.dem);
        const index = parseInt(req.body.index);
        if (key != process.env.KeyPostAnswerQuestion) {
            res.sendStatus(400);
        }
        else {
            itereactiveQuestionPostAnswer.postAnswerQuestion(phoneNumber, questionPost, answer, id, dem, index);
            res.sendStatus(200);
        }
    },
    changeDisplayName: (req, res) => {
        const phoneNumber = req.body.phoneNumber;
        const displayName = req.body.displayName;
        const key = req.body.key;
        if (key != process.env.KeyChangeDisplayName) {
            res.status(200).json({ "error": "No Key" });
        }
        else {
            changeInformation.changeDisplayName(phoneNumber, displayName);
            res.status(200).json({ "error": "No" });
        }
    },
    changePhoneNumber: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Maybe your network it hot" });
        })
        const db = dbs.db("green_big_5");
        const phoneNumber = req.body.phoneNumber;
        const phoneNumberChange = req.body.phoneNumberChange;
        const key = req.body.key;
        if (key != process.env.KeyChangePhoneNumber) {
            res.sendStatus(400);
        }
        else {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumberChange })
            if (user) {
                res.json({
                    "error": "Phone is concide"
                })
            }
            else {
                changeInformation.changePhoneNumber(phoneNumber, phoneNumberChange);
                res.json({ "error": "No" });
            }
        }
    },
    listQuestion: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Maybe your network it hot" });
        })
        const db = dbs.db("green_big_5");
        const listQueston = await db.collection("questions").find({}).toArray();
        if (req.query.key != process.env.KeyGetListQuestion) {
            res.sendStatus(400);
        }
        else {
            res.render("user/listQuestion", {
                "questions": listQueston
            });
        }
    },
    updateQuestionPersonalityLevel: async (req, res) => {
        const question = req.body.question;
        const Openness = req.body.Openness;
        const Conscientious = req.body.Conscientious;
        const Extraversion = req.body.Extraversion;
        const Agreeable = req.body.Agreeable;
        const Neuroticism = req.body.Neuroticism;
        const key = req.body.key;
        if (key != process.env.KeyPostUpdatePersonalityLevel) {
            res.sendStatus(400);
        }
        else {
            updateQuestionPersonality.updateQuestionPersonalityLevel(question, Openness, Conscientious, Extraversion, Agreeable, Neuroticism);
            res.sendStatus(200);
        }
    },
    updateQuestionPersonalityPoint: async (req, res) => {
        const OpennessPoint = req.body.OpennessPoint;
        const ConscientiousPoint = req.body.ConscientiousPoint;
        const ExtraversionPoint = req.body.ExtraversionPoint;
        const AgreeablePoint = req.body.AgreeablePoint;
        const NeuroticismPoint = req.body.NeuroticismPoint;
        const question = req.body.question;
        const key = req.body.key;
        if (key != process.env.KeyPostUpdatePersonalityPoint) {
            res.sendStatus(400);
        }
        else {
            updateQuestionPersonality.updateQuestionPersonalityPoint(question, OpennessPoint, ConscientiousPoint, ExtraversionPoint, AgreeablePoint, NeuroticismPoint);
            res.sendStatus(200);
        }

    },
    updateQuestionLevelPoint: async (req, res) => {
        const High = req.body.High;
        const Medium = req.body.Medium;
        const Low = req.body.Low;
        const question = req.body.question;
        const key = req.body.key;
        if (key != process.env.KeyPostUpdateQuestionLevelPoint) {
            res.sendStatus(400);
        }
        else {
            updateQuestionPersonality.updateQuestionLevelPoint(question, High, Medium, Low);
            res.sendStatus(200);
        }
    },
    updateAddress: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
            res.status(200).json({ "error": "Maybe your network it hot" });
        })
        try{
           
            const db = dbs.db("green_big_5");
            const key = req.body.key;
            const address = req.body.address;
            const phoneNumber = req.body.phoneNumber;
            if(key!= process.env.KeyPostUpdateQuestion){
                return res.status(200).json({err:"Access denined"});
            }
            else{
                const userProfile = await db.collection("users").findOne({phoneNumber:phoneNumber});
                console.log(userProfile.addressMap);
                console.log(address)
                const userProfileUpdate = db.collection("users").updateOne({phoneNumber:phoneNumber},{
                    $set:{
                        addressMap:[...userProfile.addressMap,address]
                    }
                })
                
                return res.status(200).json({"err":"no",userProfileUpdate});
            }
        }
        catch(err){
            console.log(err)
            return res.status(400).json({err});
        }
    }
}