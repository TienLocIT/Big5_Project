const client = require("../../mongoDB");
const checkQuestion = require("../checkQuestion/checkQuestion");
module.exports = {
    postAnswerQuestion: async (phoneNumber, questionPost, answer, id, dem, index) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5")
        const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
        const questionsDate = user.Questions[index].questionsDate;
        const questionFind = await db.collection("questions").findOne({ "question": questionPost });
        if(questionFind.totalAnswer === undefined) {
            await db.collection("questions").updateOne({"question":questionPost},{
                $set:{
                    totalAnswer:1,
                }
            })
        }
        else{
            await db.collection("questions").updateOne({"question":questionPost},{
                $set:{
                    totalAnswer:questionFind.totalAnswer + 1,
                }
            })
        }
        await db.collection("users").updateOne({"phoneNumber":phoneNumber},{
                $set:{
                   seen:false,
                }
        })
       
        //Done fixbug lần 1
        async function updatePointMultipyOpenness(phoneNumber, id, index, question, answer, questionDB) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (questionDB.answer == answer) {
                return;
            }
            if (answer == "Yes") {
                if (user.Questions[index].pointMultyplyOpenness == null) {
                    if (checkQuestion.check(question) == "Openness") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyOpenness": 2 }
                        })
                    }
                }
                else {
                    if (checkQuestion.check(question) == "Openness") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyOpenness": user.Questions[index].pointMultyplyOpenness + 2 }
                        })
                    }
                }
            }
            else {
                if (user.Questions[index].pointMultyplyOpenness == null) {
                    return;
                }
                else {
                    if (checkQuestion.check(question) == "Openness") {
                        if (user.Questions[index].pointMultyplyOpenness == 0) {
                            return;
                        }
                        else {
                            if (questionDB.answer == "Yes") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointMultyplyOpenness": user.Questions[index].pointMultyplyOpenness - 2 }
                                })
                            }
                            else {
                                return;
                            }
                        }
                    }
                }
            }

        }
        //Done fixbug lần 1
        async function updatePointMultipyConscientious(phoneNumber, id, index, question, answer, questionDB) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (questionDB.answer == answer) {
                return;
            }
            if (answer == "Yes") {
                if (user.Questions[index].pointMultyplyConscientious == null) {
                    if (checkQuestion.check(question) == "Conscientious") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyConscientious": 2 }
                        })
                    }
                }
                else {
                    if (checkQuestion.check(question) == "Conscientious") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyConscientious": user.Questions[index].pointMultyplyConscientious + 2 }
                        })
                    }
                }
            }
            else {
                if (user.Questions[index].pointMultyplyConscientious == null) {
                    return;
                }
                else {
                    if (checkQuestion.check(question) == "Conscientious") {
                        if (user.Questions[index].pointMultyplyConscientious == 0) {
                            return;
                        }
                        else {
                            if (questionDB.answer == "Yes") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointMultyplyConscientious": user.Questions[index].pointMultyplyConscientious - 2 }
                                })
                            }
                            else {
                                return;
                            }

                        }
                    }
                }
            }

        }
        //Done fixbug lần 1
        async function updatePointMultipyExtraversion(phoneNumber, id, index, question, answer, questionDB) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (questionDB.answer == answer) {
                return;
            }
            if (answer == "Yes") {
                if (user.Questions[index].pointMultyplyExtraversion == null) {
                    if (checkQuestion.check(question) == "Extraversion") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyExtraversion": 2 }
                        })
                    }
                }
                else {
                    if (checkQuestion.check(question) == "Extraversion") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyExtraversion": user.Questions[index].pointMultyplyExtraversion + 2 }
                        })
                    }
                }
            }
            else {
                if (user.Questions[index].pointMultyplyExtraversion == null) {
                    return;
                }
                else {
                    if (checkQuestion.check(question) == "Extraversion") {
                        if (user.Questions[index].pointMultyplyExtraversion == 0) {
                            return;
                        }
                        else {
                            if (questionDB.answer == "Yes") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointMultyplyExtraversion": user.Questions[index].pointMultyplyExtraversion - 2 }
                                })
                            }
                            else {
                                return;
                            }

                        }
                    }
                }
            }

        }
        //Done fixbug lần 1
        async function updatePointMultipyAgreeable(phoneNumber, id, index, question, answer, questionDB) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (questionDB.answer == answer) {
                return;
            }
            if (answer == "Yes") {
                if (user.Questions[index].pointMultyplyAgreeable == null) {
                    if (checkQuestion.check(question) == "Agreeable") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyAgreeable": 2 }
                        })
                    }
                }
                else {
                    if (checkQuestion.check(question) == "Agreeable") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyAgreeable": user.Questions[index].pointMultyplyAgreeable + 2 }
                        })
                    }
                }
            }
            else {
                if (user.Questions[index].pointMultyplyAgreeable == null) {
                    return;
                }
                else {
                    if (checkQuestion.check(question) == "Agreeable") {
                        if (user.Questions[index].pointMultyplyAgreeable == 0) {
                        }
                        else {
                            if (questionDB.answer == "Yes") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointMultyplyAgreeable": user.Questions[index].pointMultyplyAgreeable - 2 }
                                })
                            }
                            else {
                                return;
                            }

                        }
                    }
                }
            }

        }
        //Done fixbug lần 1
        async function updatePointMultipyNeuroticism(phoneNumber, id, index, question, answer, questionDB) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (questionDB.answer == answer) {
                return;
            }
            if (answer == "Yes") {
                if (user.Questions[index].pointMultyplyNeuroticism == null) {
                    if (checkQuestion.check(question) == "Neuroticism") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyNeuroticism": 2 }
                        })
                    }
                }
                else {
                    if (checkQuestion.check(question) == "Neuroticism") {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                            "$set": { "Questions.$.pointMultyplyNeuroticism": user.Questions[index].pointMultyplyNeuroticism + 2 }
                        })
                    }
                }
            }
            else {
                if (user.Questions[index].pointMultyplyNeuroticism == null) {
                    return;
                }
                else {
                    if (checkQuestion.check(question) == "Neuroticism") {
                        if (user.Questions[index].pointMultyplyNeuroticism == 0) {
                            return;
                        }
                        else {
                            if (questionDB.answer == "Yes") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointMultyplyNeuroticism": user.Questions[index].pointMultyplyNeuroticism - 2 }
                                })
                            }
                            else {
                                return;
                            }
                        }
                    }
                }
            }
        }
        async function updatePointOpenness(phoneNumber, id, index, question) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointOpenness == null) {
                if (checkQuestion.check(question) == "Openness") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointOpenness": question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Openness") == true) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointOpenness": question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointOpenness": question.point.pointLow }
                    })
                }
            }
            else {
                if (checkQuestion.check(question) == "Openness") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Openness") == true) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + question.point.pointLow }
                    })
                }
            }
        }
        async function updatePointConscientious(phoneNumber, id, index, question) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointConscientious == null) {
                if (checkQuestion.check(question) == "Conscientious") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointConscientious": question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Conscientious") == true) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointConscientious": question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointConscientious": question.point.pointLow }
                    })
                }
            }
            else {
                if (checkQuestion.check(question) == "Conscientious") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Conscientious") == true) {

                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + question.point.pointLow }
                    })
                }
            }
        }
        async function updatePointExtraversion(phoneNumber, id, index, question) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointExtraversion == null) {
                if (checkQuestion.check(question) == "Extraversion") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointExtraversion": question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Extraversion") == true) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointExtraversion": question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointExtraversion": question.point.pointLow }
                    })
                }
            }
            else {
                if (checkQuestion.check(question) == "Extraversion") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Extraversion") == true) {

                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + question.point.pointLow }
                    })
                }
            }
        }
        async function updatePointAgreeable(phoneNumber, id, index, question) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointAgreeable == null) {
                if (checkQuestion.check(question) == "Agreeable") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointAgreeable": question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Agreeable") == true) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointAgreeable": question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointAgreeable": question.point.pointLow }
                    })
                }
            }
            else {
                if (checkQuestion.check(question) == "Agreeable") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Agreeable") == true) {

                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + question.point.pointLow }
                    })
                }
            }
        }
        async function updatePointNeuroticism(phoneNumber, id, index, question) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointNeuroticism == null) {
                if (checkQuestion.check(question) == "Neuroticism") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointNeuroticism": question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Neuroticism") == true) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointNeuroticism": question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointNeuroticism": question.point.pointLow }
                    })
                }
            }
            else {
                if (checkQuestion.check(question) == "Neuroticism") {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + question.point.pointHigh }
                    })
                }
                else if ((checkQuestion.checkMedium(question)).includes("Neuroticism") == true) {

                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + question.point.pointMedium }
                    })
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                        "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + question.point.pointLow }
                    })
                }
            }
        }
        async function UpdateAnswer(phoneNumber, id, question, answer) {
            await db.collection("users").updateOne({
                "phoneNumber": phoneNumber, "Questions": {
                    "$elemMatch": {
                        id, "questionsDate.question": question
                    }
                }
            }, {
                "$set": {
                    "Questions.$[outer].questionsDate.$[inner].answer": answer
                }
            }, {
                "arrayFilters": [
                    { "outer.id": id },
                    { "inner.question": question }
                ]
            })
        }
        
        if (user.Questions[index].questionsDate[dem].answer == null) {
            updatePointOpenness(phoneNumber, id, index, questionFind);
            updatePointConscientious(phoneNumber, id, index, questionFind);
            updatePointExtraversion(phoneNumber, id, index, questionFind);
            updatePointAgreeable(phoneNumber, id, index, questionFind);
            updatePointNeuroticism(phoneNumber, id, index, questionFind);
            updatePointMultipyOpenness(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyConscientious(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyExtraversion(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyAgreeable(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyNeuroticism(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
        }
        else {
            updatePointMultipyOpenness(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyConscientious(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyExtraversion(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyAgreeable(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
            updatePointMultipyNeuroticism(phoneNumber, id, index, questionFind, answer, questionsDate[dem]);
        }
        UpdateAnswer(phoneNumber, id, questionPost, answer);
    },
}