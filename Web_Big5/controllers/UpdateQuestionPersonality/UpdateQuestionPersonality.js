const client = require("../../mongoDB");
const checkQuestion = require("../checkQuestion/checkQuestion");
module.exports = {
    updateQuestionPersonalityLevel: async (question, Openness, Conscientious, Extraversion, Agreeable, Neuroticism) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        async function doUpdateQuestionOpennessPointNo(Openness, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Openness") {
                if (Openness == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "High" },
                        { "$set": { "personality.$.Openness": "Medium" } })
                }
                else if (Openness == "Low") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "High" },
                        { "$set": { "personality.$.Openness": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Openness") {
                if (Openness == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Medium" },
                        { "$set": { "personality.$.Openness": "High" } })
                }
                else if (Openness == "Low") {

                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Medium" },
                        { "$set": { "personality.$.Openness": "Low" } })
                }
            }
            else {
                if (Openness == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Low" },
                        { "$set": { "personality.$.Openness": "High" } })
                }
                else if (Openness == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Low" },
                        { "$set": { "personality.$.Openness": "Medium" } })
                }
            }
        }
        //Point Yes xong
        async function doUpdateQuestionOpennessPointYes(Openness, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Openness") {
                if (Openness == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {//test ổn        
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        //test ổn cả 2
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {

                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) { //test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {//test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        //test ổn
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "High" },
                        { "$set": { "personality.$.Openness": "Medium" } })
                }
                else if (Openness == "Low") {//test ổn
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "High" },
                        { "$set": { "personality.$.Openness": "Low" } })
                }
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                    { "$set": { "Questions.$.pointMultyplyOpenness": user.Questions[index].pointMultyplyOpenness - 2 } })
            }
            else if (checkQuestion.checkMedium(questionFind) == "Openness") {
                if (Openness == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Medium" },
                        { "$set": { "personality.$.Openness": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyOpenness": user.Questions[index].pointMultyplyOpenness + 2 } })
                }
                else if (Openness == "Low") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Medium" },
                        { "$set": { "personality.$.Openness": "Low" } })
                }
            }
            else {
                if (Openness == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Low" },
                        { "$set": { "personality.$.Openness": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyOpenness": user.Questions[index].pointMultyplyOpenness + 2 } })
                }
                else if (Openness == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Low" },
                        { "$set": { "personality.$.Openness": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionOpeenessPointNull(question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Openness") {
                if (Openness == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "High" },
                        { "$set": { "personality.$.Openness": "Medium" } })
                }
                else if (Openness == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "High" },
                        { "$set": { "personality.$.Openness": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Openness") {
                if (Openness == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Medium" },
                        { "$set": { "personality.$.Openness": "High" } })
                }
                else if (Openness == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Medium" },
                        { "$set": { "personality.$.Openness": "Low" } })
                }
            }
            else {
                if (Openness == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Low" },
                        { "$set": { "personality.$.Openness": "High" } })
                }
                else if (Openness == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Openness": "Low" },
                        { "$set": { "personality.$.Openness": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionOpennessPoint(Openness, question, objectQuestionFind, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointOpenness == null || user.Questions[index].pointOpenness == "" || user.Questions[index].pointOpenness == 0) {
                doUpdateQuestionOpeenessPointNull(question);
            }
            else {
                if (objectQuestionFind.point.answer == "Yes") {
                    doUpdateQuestionOpennessPointYes(Openness, question, index, id, phoneNumber);
                }
                else if (objectQuestionFind.point.answer == "No") {
                    doUpdateQuestionOpennessPointNo(Openness, question, index, id, phoneNumber);
                }
                else {
                    doUpdateQuestionOpeenessPointNull(question);
                }
            }
        }
        async function updateQuestionOpennessPoint(Openness, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionOpennessPoint(Openness, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionOpenness(Openness, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.personality[0].Openness == Openness) {
                return;
            }
            else {
                updateQuestionOpennessPoint(Openness, question);
            }
        }
        //Test ổn hết

        // Update Questions point Conscientious
        async function doUpdateQuestionConscientiousPointNo(Conscientious, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Conscientious") {
                if (Conscientious == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "High" },
                        { "$set": { "personality.$.Conscientious": "Medium" } })
                }
                else if (Conscientious == "Low") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "High" },
                        { "$set": { "personality.$.Conscientious": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Conscientious") {
                if (Conscientious == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Medium" },
                        { "$set": { "personality.$.Conscientious": "High" } })
                }
                else if (Conscientious == "Low") {

                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Medium" },
                        { "$set": { "personality.$.Conscientious": "Low" } })
                }
            }
            else {
                if (Conscientious == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Low" },
                        { "$set": { "personality.$.Conscientious": "High" } })
                }
                else if (Conscientious == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Low" },
                        { "$set": { "personality.$.Conscientious": "Medium" } })
                }
            }
        }
        //Point Yes xong
        async function doUpdateQuestionConscientiousPointYes(Conscientious, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Conscientious") {
                if (Conscientious == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {//test ổn        
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        //test ổn cả 2
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {

                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) { //test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {//test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        //test ổn
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "High" },
                        { "$set": { "personality.$.Conscientious": "Medium" } })
                }
                else if (Conscientious == "Low") {//test ổn
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "High" },
                        { "$set": { "personality.$.Conscientious": "Low" } })
                }
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                    { "$set": { "Questions.$.pointMultyplyConscientious": user.Questions[index].pointMultyplyConscientious - 2 } })
            }
            else if (checkQuestion.checkMedium(questionFind) == "Conscientious") {
                if (Conscientious == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Medium" },
                        { "$set": { "personality.$.Conscientious": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyConscientious": user.Questions[index].pointMultyplyConscientious + 2 } })
                }
                else if (Conscientious == "Low") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Medium" },
                        { "$set": { "personality.$.Conscientious": "Low" } })
                }
            }
            else {
                if (Conscientious == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Low" },
                        { "$set": { "personality.$.Conscientious": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyConscientious": user.Questions[index].pointMultyplyConscientious + 2 } })
                }
                else if (Conscientious == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Low" },
                        { "$set": { "personality.$.Conscientious": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionOpeenessPointNull(question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Conscientious") {
                if (Conscientious == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "High" },
                        { "$set": { "personality.$.Conscientious": "Medium" } })
                }
                else if (Conscientious == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "High" },
                        { "$set": { "personality.$.Conscientious": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Conscientious") {
                if (Conscientious == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Medium" },
                        { "$set": { "personality.$.Conscientious": "High" } })
                }
                else if (Conscientious == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Medium" },
                        { "$set": { "personality.$.Conscientious": "Low" } })
                }
            }
            else {
                if (Conscientious == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Low" },
                        { "$set": { "personality.$.Conscientious": "High" } })
                }
                else if (Conscientious == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Conscientious": "Low" },
                        { "$set": { "personality.$.Conscientious": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionConscientiousPoint(Conscientious, question, objectQuestionFind, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointConscientious == null || user.Questions[index].pointConscientious == "" || user.Questions[index].pointConscientious == 0) {
                doUpdateQuestionOpeenessPointNull(question);
            }
            else {
                if (objectQuestionFind.point.answer == "Yes") {
                    doUpdateQuestionConscientiousPointYes(Conscientious, question, index, id, phoneNumber);
                }
                else if (objectQuestionFind.point.answer == "No") {
                    doUpdateQuestionConscientiousPointNo(Conscientious, question, index, id, phoneNumber);
                }
                else {
                    doUpdateQuestionOpeenessPointNull(question);
                }
            }
        }
        async function updateQuestionConscientiousPoint(Conscientious, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionConscientiousPoint(Conscientious, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionConscientious(Conscientious, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.personality[0].Conscientious == Conscientious) {
                return;
            }
            else {
                updateQuestionConscientiousPoint(Conscientious, question);
            }
        }
        // Update point Extraversion
        async function doUpdateQuestionExtraversionPointNo(Extraversion, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Extraversion") {
                if (Extraversion == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "High" },
                        { "$set": { "personality.$.Extraversion": "Medium" } })
                }
                else if (Extraversion == "Low") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "High" },
                        { "$set": { "personality.$.Extraversion": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Extraversion") {
                if (Extraversion == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Medium" },
                        { "$set": { "personality.$.Extraversion": "High" } })
                }
                else if (Extraversion == "Low") {

                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Medium" },
                        { "$set": { "personality.$.Extraversion": "Low" } })
                }
            }
            else {
                if (Extraversion == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Low" },
                        { "$set": { "personality.$.Extraversion": "High" } })
                }
                else if (Extraversion == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Low" },
                        { "$set": { "personality.$.Extraversion": "Medium" } })
                }
            }
        }
        //Point Yes xong
        async function doUpdateQuestionExtraversionPointYes(Extraversion, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Extraversion") {
                if (Extraversion == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {//test ổn        
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        //test ổn cả 2
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {

                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) { //test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {//test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        //test ổn
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "High" },
                        { "$set": { "personality.$.Extraversion": "Medium" } })
                }
                else if (Extraversion == "Low") {//test ổn
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "High" },
                        { "$set": { "personality.$.Extraversion": "Low" } })
                }
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                    { "$set": { "Questions.$.pointMultyplyExtraversion": user.Questions[index].pointMultyplyExtraversion - 2 } })
            }
            else if (checkQuestion.checkMedium(questionFind) == "Extraversion") {
                if (Extraversion == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Medium" },
                        { "$set": { "personality.$.Extraversion": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyExtraversion": user.Questions[index].pointMultyplyExtraversion + 2 } })
                }
                else if (Extraversion == "Low") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Medium" },
                        { "$set": { "personality.$.Extraversion": "Low" } })
                }
            }
            else {
                if (Extraversion == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Low" },
                        { "$set": { "personality.$.Extraversion": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyExtraversion": user.Questions[index].pointMultyplyExtraversion + 2 } })
                }
                else if (Extraversion == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Low" },
                        { "$set": { "personality.$.Extraversion": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionOpeenessPointNull(question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Extraversion") {
                if (Extraversion == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "High" },
                        { "$set": { "personality.$.Extraversion": "Medium" } })
                }
                else if (Extraversion == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "High" },
                        { "$set": { "personality.$.Extraversion": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Extraversion") {
                if (Extraversion == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Medium" },
                        { "$set": { "personality.$.Extraversion": "High" } })
                }
                else if (Extraversion == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Medium" },
                        { "$set": { "personality.$.Extraversion": "Low" } })
                }
            }
            else {
                if (Extraversion == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Low" },
                        { "$set": { "personality.$.Extraversion": "High" } })
                }
                else if (Extraversion == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Extraversion": "Low" },
                        { "$set": { "personality.$.Extraversion": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionExtraversionPoint(Extraversion, question, objectQuestionFind, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointExtraversion == null || user.Questions[index].pointExtraversion == "" || user.Questions[index].pointExtraversion == 0) {
                doUpdateQuestionOpeenessPointNull(question);
            }
            else {
                if (objectQuestionFind.point.answer == "Yes") {
                    doUpdateQuestionExtraversionPointYes(Extraversion, question, index, id, phoneNumber);
                }
                else if (objectQuestionFind.point.answer == "No") {
                    doUpdateQuestionExtraversionPointNo(Extraversion, question, index, id, phoneNumber);
                }
                else {
                    doUpdateQuestionOpeenessPointNull(question);
                }
            }
        }
        async function updateQuestionExtraversionPoint(Extraversion, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionExtraversionPoint(Extraversion, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionExtraversion(Extraversion, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.personality[0].Extraversion == Extraversion) {
                return;
            }
            else {
                updateQuestionExtraversionPoint(Extraversion, question);
            }
        }

        // Update Point Agreeable
        async function doUpdateQuestionAgreeablePointNo(Agreeable, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Agreeable") {
                if (Agreeable == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "High" },
                        { "$set": { "personality.$.Agreeable": "Medium" } })
                }
                else if (Agreeable == "Low") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "High" },
                        { "$set": { "personality.$.Agreeable": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Agreeable") {
                if (Agreeable == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Medium" },
                        { "$set": { "personality.$.Agreeable": "High" } })
                }
                else if (Agreeable == "Low") {

                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Medium" },
                        { "$set": { "personality.$.Agreeable": "Low" } })
                }
            }
            else {
                if (Agreeable == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Low" },
                        { "$set": { "personality.$.Agreeable": "High" } })
                }
                else if (Agreeable == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Low" },
                        { "$set": { "personality.$.Agreeable": "Medium" } })
                }
            }
        }
        //Point Yes xong
        async function doUpdateQuestionAgreeablePointYes(Agreeable, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Agreeable") {
                if (Agreeable == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {//test ổn        
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        //test ổn cả 2
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {

                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) { //test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {//test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        //test ổn
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "High" },
                        { "$set": { "personality.$.Agreeable": "Medium" } })
                }
                else if (Agreeable == "Low") {//test ổn
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "High" },
                        { "$set": { "personality.$.Agreeable": "Low" } })
                }
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                    { "$set": { "Questions.$.pointMultyplyAgreeable": user.Questions[index].pointMultyplyAgreeable - 2 } })
            }
            else if (checkQuestion.checkMedium(questionFind) == "Agreeable") {
                if (Agreeable == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Medium" },
                        { "$set": { "personality.$.Agreeable": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyAgreeable": user.Questions[index].pointMultyplyAgreeable + 2 } })
                }
                else if (Agreeable == "Low") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Medium" },
                        { "$set": { "personality.$.Agreeable": "Low" } })
                }
            }
            else {
                if (Agreeable == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Low" },
                        { "$set": { "personality.$.Agreeable": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyAgreeable": user.Questions[index].pointMultyplyAgreeable + 2 } })
                }
                else if (Agreeable == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Low" },
                        { "$set": { "personality.$.Agreeable": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionOpeenessPointNull(question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Agreeable") {
                if (Agreeable == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "High" },
                        { "$set": { "personality.$.Agreeable": "Medium" } })
                }
                else if (Agreeable == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "High" },
                        { "$set": { "personality.$.Agreeable": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Agreeable") {
                if (Agreeable == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Medium" },
                        { "$set": { "personality.$.Agreeable": "High" } })
                }
                else if (Agreeable == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Medium" },
                        { "$set": { "personality.$.Agreeable": "Low" } })
                }
            }
            else {
                if (Agreeable == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Low" },
                        { "$set": { "personality.$.Agreeable": "High" } })
                }
                else if (Agreeable == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Agreeable": "Low" },
                        { "$set": { "personality.$.Agreeable": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionAgreeablePoint(Agreeable, question, objectQuestionFind, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointAgreeable == null || user.Questions[index].pointAgreeable == "" || user.Questions[index].pointAgreeable == 0) {
                doUpdateQuestionOpeenessPointNull(question);
            }
            else {
                if (objectQuestionFind.point.answer == "Yes") {
                    doUpdateQuestionAgreeablePointYes(Agreeable, question, index, id, phoneNumber);
                }
                else if (objectQuestionFind.point.answer == "No") {
                    doUpdateQuestionAgreeablePointNo(Agreeable, question, index, id, phoneNumber);
                }
                else {
                    doUpdateQuestionOpeenessPointNull(question);
                }
            }
        }
        async function updateQuestionAgreeablePoint(Agreeable, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionAgreeablePoint(Agreeable, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionAgreeable(Agreeable, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.personality[0].Agreeable == Agreeable) {
                return;
            }
            else {
                updateQuestionAgreeablePoint(Agreeable, question);
            }
        }
        // Update Question Neuroticism
        async function doUpdateQuestionNeuroticismPointNo(Neuroticism, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Neuroticism") {
                if (Neuroticism == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "High" },
                        { "$set": { "personality.$.Neuroticism": "Medium" } })
                }
                else if (Neuroticism == "Low") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "High" },
                        { "$set": { "personality.$.Neuroticism": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Neuroticism") {
                if (Neuroticism == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Medium" },
                        { "$set": { "personality.$.Neuroticism": "High" } })
                }
                else if (Neuroticism == "Low") {

                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Medium" },
                        { "$set": { "personality.$.Neuroticism": "Low" } })
                }
            }
            else {
                if (Neuroticism == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Low" },
                        { "$set": { "personality.$.Neuroticism": "High" } })
                }
                else if (Neuroticism == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Low" },
                        { "$set": { "personality.$.Neuroticism": "Medium" } })
                }
            }
        }
        //Point Yes xong
        async function doUpdateQuestionNeuroticismPointYes(Neuroticism, question, index, id, phoneNumber) {
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            var questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Neuroticism") {
                if (Neuroticism == "Medium") {
                    if (questionFind.point.pointHigh == null && questionFind.point.pointMedium != null) {//test ổn        
                        if (20 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (20 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointMedium - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium == null) {
                        //test ổn cả 2
                        if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - 15) } });
                        }
                        else if (questionFind.point.pointHigh < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (15 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointMedium != null) {

                        if (questionFind.point.pointHigh > questionFind.point.pointMedium) { //test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointMedium) {//test ổn
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        //test ổn
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "High" },
                        { "$set": { "personality.$.Neuroticism": "Medium" } })
                }
                else if (Neuroticism == "Low") {//test ổn
                    if (questionFind.point.pointHigh == null && questionFind.point.pointLow != null) {
                        if (20 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (20 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointLow - 20) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - 10) } });
                        }
                        else if (questionFind.point.pointHigh < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (10 - questionFind.point.pointHigh) } });
                        }
                    }
                    else if (questionFind.point.pointHigh != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointHigh > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointHigh < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "High" },
                        { "$set": { "personality.$.Neuroticism": "Low" } })
                }
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                    { "$set": { "Questions.$.pointMultyplyNeuroticism": user.Questions[index].pointMultyplyNeuroticism - 2 } })
            }
            else if (checkQuestion.checkMedium(questionFind) == "Neuroticism") {
                if (Neuroticism == "High") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointHigh != null) {
                        if (15 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (15 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointHigh - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointMedium > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - 20) } });
                        }
                        else if (questionFind.point.pointMedium < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (20 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Medium" },
                        { "$set": { "personality.$.Neuroticism": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyNeuroticism": user.Questions[index].pointMultyplyNeuroticism + 2 } })
                }
                else if (Neuroticism == "Low") {
                    if (questionFind.point.pointMedium == null && questionFind.point.pointLow != null) {
                        if (15 > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (15 - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointLow - 15) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow == null) {
                        if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - 10) } });
                        }
                        else if (questionFind.point.pointMedium < 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (10 - questionFind.point.pointMedium) } });
                        }
                    }
                    else if (questionFind.point.pointMedium != null && questionFind.point.pointLow != null) {
                        if (questionFind.point.pointMedium > questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                        else if (questionFind.point.pointMedium < questionFind.point.pointLow) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Medium" },
                        { "$set": { "personality.$.Neuroticism": "Low" } })
                }
            }
            else {
                if (Neuroticism == "High") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointHigh != null) {
                        if (10 > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (10 - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointHigh > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointHigh - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh == null) {
                        if (questionFind.point.pointLow > 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - 20) } });
                        }
                        else if (questionFind.point.pointLow < 20) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (20 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointHigh != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointHigh) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointHigh) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointHigh - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + 10 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Low" },
                        { "$set": { "personality.$.Neuroticism": "High" } })
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointMultyplyNeuroticism": user.Questions[index].pointMultyplyNeuroticism + 2 } })
                }
                else if (Neuroticism == "Medium") {
                    if (questionFind.point.pointLow == null && questionFind.point.pointMedium != null) {
                        if (10 > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (10 - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointMedium > 10) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (questionFind.point.pointMedium - 10) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium == null) {
                        if (questionFind.point.pointLow > 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - 15) } });
                        }
                        else if (questionFind.point.pointLow < 15) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (15 - questionFind.point.pointLow) } });
                        }
                    }
                    else if (questionFind.point.pointLow != null && questionFind.point.pointMedium != null) {
                        if (questionFind.point.pointLow > questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - questionFind.point.pointMedium) } });
                        }
                        else if (questionFind.point.pointLow < questionFind.point.pointMedium) {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                                { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - questionFind.point.pointLow) } });
                        }
                    }
                    else {
                        await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                            { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + 5 } });
                    }
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Low" },
                        { "$set": { "personality.$.Neuroticism": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionOpeenessPointNull(question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (checkQuestion.check(questionFind) == "Neuroticism") {
                if (Neuroticism == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "High" },
                        { "$set": { "personality.$.Neuroticism": "Medium" } })
                }
                else if (Neuroticism == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "High" },
                        { "$set": { "personality.$.Neuroticism": "Low" } })
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Neuroticism") {
                if (Neuroticism == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Medium" },
                        { "$set": { "personality.$.Neuroticism": "High" } })
                }
                else if (Neuroticism == "Low") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Medium" },
                        { "$set": { "personality.$.Neuroticism": "Low" } })
                }
            }
            else {
                if (Neuroticism == "High") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Low" },
                        { "$set": { "personality.$.Neuroticism": "High" } })
                }
                else if (Neuroticism == "Medium") {
                    await db.collection("questions").updateOne({ "question": question, "personality.Neuroticism": "Low" },
                        { "$set": { "personality.$.Neuroticism": "Medium" } })
                }
            }
        }
        async function doUpdateQuestionNeuroticismPoint(Neuroticism, question, objectQuestionFind, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (user.Questions[index].pointNeuroticism == null || user.Questions[index].pointNeuroticism == "" || user.Questions[index].pointNeuroticism == 0) {
                doUpdateQuestionOpeenessPointNull(question);
            }
            else {
                if (objectQuestionFind.point.answer == "Yes") {
                    doUpdateQuestionNeuroticismPointYes(Neuroticism, question, index, id, phoneNumber);
                }
                else if (objectQuestionFind.point.answer == "No") {
                    doUpdateQuestionNeuroticismPointNo(Neuroticism, question, index, id, phoneNumber);
                }
                else {
                    doUpdateQuestionOpeenessPointNull(question);
                }
            }
        }
        async function updateQuestionNeuroticismPoint(Neuroticism, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionNeuroticismPoint(Neuroticism, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionNeuroticism(Neuroticism, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.personality[0].Neuroticism == Neuroticism) {
                return;
            }
            else {
                updateQuestionNeuroticismPoint(Neuroticism, question);
            }
        }
        updateQuestionOpenness(Openness, question);
        updateQuestionConscientious(Conscientious, question);
        updateQuestionExtraversion(Extraversion, question);
        updateQuestionAgreeable(Agreeable, question);
        updateQuestionNeuroticism(Neuroticism, question);
    },
    updateQuestionLevelPoint: async (question, High, Medium, Low) => {
        const db = client.db("green_big_5");
        //Update Point High
        async function doUpdateQuestionWithPointHighLoop(High, question, objectQuestionFind, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (objectQuestionFind.point.answer == null || objectQuestionFind.point.answer == "") {

            }
            else {
                if (questionFind.point.pointHigh > High) {
                    switch (checkQuestion.check(questionFind)) {
                        case "Openness":
                            if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (user.Questions[index].pointOpenness - High) }
                                })
                            }
                            break;
                        case "Conscientious":
                            if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (user.Questions[index].pointConscientious - High) }
                                })
                            }
                            break;
                        case "Extraversion":
                            if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (user.Questions[index].pointExtraversion - High) }
                                })
                            }
                            break;
                        case "Agreeable":
                            if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                                console.log("Yesss");
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (user.Questions[index].pointAgreeable - High) }
                                })
                            }
                            break;
                        default:
                            if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (user.Questions[index].pointAgreeable - High) }
                                })
                            }
                            break;
                    }
                }
                else {
                    switch (checkQuestion.check(questionFind)) {
                        case "Openness":
                            if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (High - user.Questions[index].pointOpenness) }
                                })
                            }
                            break;
                        case "Conscientious":
                            if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (High - user.Questions[index].pointConscientious) }
                                })
                            }
                            break;
                        case "Extraversion":
                            if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (High - user.Questions[index].pointExtraversion) }
                                })
                            }
                            break;
                        case "Agreeable":
                            if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (High - user.Questions[index].pointAgreeable) }
                                })
                            }
                            break;
                        default:
                            if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (High - user.Questions[index].pointNeuroticism) }
                                })
                            }
                            break;
                    }

                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointHigh": High }
            })
        }
        async function doUpdatePointHighWithPointHigh(High, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithPointHighLoop(High, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithoutPointHighLoop(High, question, objectQuestionFind, index, id, phoneNumber) {
            if (High == 20) {
                return;
            }
            const questionFind = await db.collection("questions").findOne({ "question": question });
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (objectQuestionFind.point.answer == null || objectQuestionFind.point.answer == "") {

            }
            else {
                if (20 > High) {
                    switch (checkQuestion.check(questionFind)) {
                        case "Openness":
                            if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (20 - High) }
                                })
                            }
                            break;
                        case "Conscientious":
                            if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (20 - High) }
                                })
                            }
                            break;
                        case "Extraversion":
                            if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (20 - High) }
                                })
                            }
                            break;
                        case "Agreeable":
                            if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (20 - High) }
                                })
                            }
                            break;
                        default:
                            if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (20 - High) }
                                })
                            }
                            break;
                    }

                }
                else {
                    switch (checkQuestion.check(questionFind)) {
                        case "Openness":
                            if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (High - 20) }
                                })
                            }
                            break;
                        case "Conscientious":
                            if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (High - 20) }
                                })
                            }
                            break;
                        case "Extraversion":
                            if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (High - 20) }
                                })
                            }
                            break;
                        case "Agreeable":
                            if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (High - 20) }
                                })
                            }
                            break;
                        default:
                            if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                    "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (High - 20) }
                                })
                            }
                            break;
                    }
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointHigh": High }
            })
        }
        async function doUpdatePointHighWithoutPointHigh(High, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithoutPointHighLoop(High, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionPointHigh(High, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });

            if (questionFind.point.pointHigh != null) {
                if (questionFind.point.pointHigh == High) {
                    return;
                }
                else {
                    doUpdatePointHighWithPointHigh(High, question);
                    console.log(2);
                }
            }
            else {
                if (questionFind.point.pointHigh == High) {
                    return;
                }
                else {
                    doUpdatePointHighWithoutPointHigh(High, question);
                }
            }
        }

        //Update Point Medium
        async function doUpdateQuestionWithPointMediumLoop(Medium, question, objectQuestionFind, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (objectQuestionFind.point.answer == null || objectQuestionFind.point.answer == "") {

            }
            else {
                if (questionFind.point.pointMedium > Medium) {
                    if (questionFind.point.personality[0].Openness == "Medium") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointMedium - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Medium") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointMedium - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Medium") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointMedium - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Medium") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointMedium - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Medium") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointMedium - Medium) }
                            })
                        }
                    }
                }
                else {
                    if (questionFind.point.personality[0].Openness == "Medium") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (Medium - questionFind.point.pointMedium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Medium") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (Medium - questionFind.point.pointMedium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Medium") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (Medium - questionFind.point.pointMedium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Medium") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (Medium - questionFind.point.pointMedium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Medium") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (Medium - questionFind.point.pointMedium) }
                            })
                        }
                    }
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointMedium": Medium }
            })
        }
        async function doUpdateQuestionPointMediumWithPointMedium(Medium, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithPointMediumLoop(Medium, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithoutPointMediumLoop(Medium, question, objectQuestionFind, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (Medium == 15) {
                return;
            }
            if (objectQuestionFind.point.answer == null || objectQuestionFind.point.answer == "") {

            }
            else {
                if (15 > Medium) {
                    if (questionFind.point.personality[0].Openness == "Medium") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (15 - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Medium") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (15 - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Medium") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (15 - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Medium") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (15 - Medium) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Medium") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (15 - Medium) }
                            })
                        }
                    }
                }
                else {
                    if (questionFind.point.personality[0].Openness == "Medium") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (Medium - 15) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Medium") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (Medium - 15) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Medium") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (Medium - 15) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Medium") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (Medium - 15) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Medium") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (Medium - 15) }
                            })
                        }
                    }
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointMedium": Medium }
            })
        }
        async function doUpdateQuestionPointMediumWithoutPointMedium(Medium, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithoutPointMediumLoop(Medium, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionPointMedium(Medium, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.pointMedium != null) {
                if (questionFind.point.pointMedium == Medium) {
                    return;
                }
                else {
                    doUpdateQuestionPointMediumWithPointMedium(Medium, question);
                }
            }
            else {
                if (questionFind.point.pointMedium == Medium) {
                    return;
                }
                else {
                    console.log("Zo")
                    doUpdateQuestionPointMediumWithoutPointMedium(Medium, question);
                }
            }
        }
        //Update Point Low
        async function doUpdateQuestionWithPointLowLoop(Low, question, objectQuestionFind, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (objectQuestionFind.point.answer == null || objectQuestionFind.point.answer == "") {

            }
            else {
                if (questionFind.point.pointLow > Low) {
                    if (questionFind.point.personality[0].Openness == "Low") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointLow - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Low") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointLow - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Low") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointLow - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Low") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointLow - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Low") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointLow - Low) }
                            })
                        }
                    }
                }
                else {
                    if (questionFind.point.personality[0].Openness == "Low") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (Low - questionFind.point.pointLow) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Low") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (Low - questionFind.point.pointLow) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Low") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (Low - questionFind.point.pointLow) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Low") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (Low - questionFind.point.pointLow) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Low") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (Low - questionFind.point.pointLow) }
                            })
                        }
                    }
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointLow": Low }
            })
        }
        async function doUpdateQuestionPointLowWithPointLow(Low, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithPointLowLoop(Low, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionPointLowWithoutPointLow(Low, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithoutPointLowLoop(Low, question, listUser[i].Questions[j].questionsDate[k], j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithoutPointLowLoop(Low, question, objectQuestionFind, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (Low == 10) {
                return;
            }
            if (objectQuestionFind.point.answer == null || objectQuestionFind.point.answer == "") {

            }
            else {
                if (10 > Low) {
                    if (questionFind.point.personality[0].Openness == "Low") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (10 - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Low") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (10 - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Low") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (10 - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Low") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (10 - Low) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Low") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (10 - Low) }
                            })
                        }
                    }
                }
                else {
                    if (questionFind.point.personality[0].Openness == "Low") {
                        if (user.Questions[index].pointOpenness != null || user.Questions[index].pointOpenness != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (Low - 10) }
                            })
                        }
                    }
                    if (questionFind.point.personality[1].Conscientious == "Low") {
                        if (user.Questions[index].pointConscientious != null || user.Questions[index].pointConscientious != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (Low - 10) }
                            })
                        }
                    }
                    if (questionFind.point.personality[2].Extraversion == "Low") {
                        if (user.Questions[index].pointExtraversion != null || user.Questions[index].pointExtraversion != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (Low - 10) }
                            })
                        }
                    }
                    if (questionFind.point.personality[3].Agreeable == "Low") {
                        if (user.Questions[index].pointAgreeable != null || user.Questions[index].pointAgreeable != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (Low - 10) }
                            })
                        }
                    }
                    if (questionFind.point.personality[4].Neuroticism == "Low") {
                        if (user.Questions[index].pointNeuroticism != null || user.Questions[index].pointNeuroticism != "") {
                            await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                                "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (Low - 10) }
                            })
                        }
                    }
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointLow": Low }
            })
        }
        async function updateQuestionPointLow(Low, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.pointLow != null) {
                if (questionFind.point.pointLow == Low) {
                    return;
                }
                else {
                    doUpdateQuestionPointLowWithPointLow(Low, question);
                }
            }
            else {
                if (questionFind.point.pointLoww == Low) {
                    return;
                }
                else {
                    doUpdateQuestionPointLowWithoutPointLow(Low, question);
                }
            }
        }
        updateQuestionPointHigh(High, question);
        updateQuestionPointMedium(Medium, question);
        updateQuestionPointLow(Low, question);
    },
    updateQuestionPersonalityPoint: async (question, OpennessPoint, ConscientiousPoint, ExtraversionPoint, AgreeablePoint, NeuroticismPoint) => {
        const db = client.db("green_big_5");
        async function doUpdateQuestionWithNoOpennessPointInDB(OpennessPoint, question, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (checkQuestion.check(questionFind) == "Openness") {
                if (OpennessPoint > 20) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (OpennessPoint - 20) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (20 - OpennessPoint) } }
                    )
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Openness") {
                if (OpennessPoint > 15) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (OpennessPoint - 15) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (15 - OpennessPoint) } }
                    )
                }
            }
            else {
                if (OpennessPoint > 10) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (OpennessPoint - 10) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (10 - OpennessPoint) } }
                    )
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointOpenness": OpennessPoint }
            })
        }
        async function updateQuestionWithNoOpennessPointInQuestion(OpennessPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithNoOpennessPointInDB(OpennessPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithOpennessPointInQuestion(OpennessPoint, question, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.OpennessPoint > OpennessPoint) {
                console.log(questionFind.point.OpennessPoint - OpennessPoint)
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness - (questionFind.point.pointOpenness - OpennessPoint) }
                })
            }
            else {
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointOpenness": user.Questions[index].pointOpenness + (OpennessPoint - questionFind.point.pointOpenness) }
                })
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointOpenness": OpennessPoint }
            })
        }
        async function updateQuestionWithOpennessPointInQuestion(OpennessPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithOpennessPointInQuestion(OpennessPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionPointOpenness(OpennessPoint, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.pointOpenness == null || questionFind.point.pointOpenness == "") {
                updateQuestionWithNoOpennessPointInQuestion(OpennessPoint, question)
            }
            else {
                if (OpennessPoint == questionFind.point.pointOpenness) {
                    return;
                }
                else {
                    updateQuestionWithOpennessPointInQuestion(OpennessPoint, question)
                }
            }
        }

        // Update Conscientious Point

        async function doUpdateQuestionWithNoConscientiousPointInDB(ConscientiousPoint, question, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (checkQuestion.check(questionFind) == "Conscientious") {
                if (ConscientiousPoint > 20) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (ConscientiousPoint - 20) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (20 - ConscientiousPoint) } }
                    )
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Conscientious") {
                if (ConscientiousPoint > 15) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (ConscientiousPoint - 15) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (15 - ConscientiousPoint) } }
                    )
                }
            }
            else {
                if (ConscientiousPoint > 10) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (ConscientiousPoint - 10) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (10 - ConscientiousPoint) } }
                    )
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointConscientious": ConscientiousPoint }
            })
        }
        async function updateQuestionWithNoConscientiousPointInQuestion(ConscientiousPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithNoConscientiousPointInDB(ConscientiousPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithConscientiousPointInQuestion(ConscientiousPoint, question, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            const questionFind = await db.collection("questions").findOne({ "question": question });
            console.log(questionFind.point.pointConscientious - ConscientiousPoint);
            if (questionFind.point.ConscientiousPoint > ConscientiousPoint) {
                console.log(questionFind.point.ConscientiousPoint - ConscientiousPoint)
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious - (questionFind.point.pointConscientious - ConscientiousPoint) }
                })
            }
            else {
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointConscientious": user.Questions[index].pointConscientious + (ConscientiousPoint - questionFind.point.pointConscientious) }
                })
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointConscientious": ConscientiousPoint }
            })
        }
        async function updateQuestionWithConscientiousPointInQuestion(ConscientiousPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithConscientiousPointInQuestion(ConscientiousPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionPointConscientious(ConscientiousPoint, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.pointConscientious == null || questionFind.point.pointConscientious == "") {
                updateQuestionWithNoConscientiousPointInQuestion(ConscientiousPoint, question)
            }
            else {
                if (ConscientiousPoint == questionFind.point.pointConscientious) {
                    return;
                }
                else {
                    updateQuestionWithConscientiousPointInQuestion(ConscientiousPoint, question)
                }
            }
        }

        // Update Extraversion Point
        async function doUpdateQuestionWithNoExtraversionPointInDB(ExtraversionPoint, question, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (checkQuestion.check(questionFind) == "Extraversion") {
                if (ExtraversionPoint > 20) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (ExtraversionPoint - 20) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (20 - ExtraversionPoint) } }
                    )
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Extraversion") {
                if (ExtraversionPoint > 15) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (ExtraversionPoint - 15) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (15 - ExtraversionPoint) } }
                    )
                }
            }
            else {
                if (ExtraversionPoint > 10) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (ExtraversionPoint - 10) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (10 - ExtraversionPoint) } }
                    )
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointExtraversion": ExtraversionPoint }
            })
        }
        async function updateQuestionWithNoExtraversionPointInQuestion(ExtraversionPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithNoExtraversionPointInDB(ExtraversionPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithExtraversionPointInQuestion(ExtraversionPoint, question, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            const questionFind = await db.collection("questions").findOne({ "question": question });
            console.log(questionFind.point.pointExtraversion - ExtraversionPoint);
            if (questionFind.point.ExtraversionPoint > ExtraversionPoint) {
                console.log(questionFind.point.ExtraversionPoint - ExtraversionPoint)
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion - (questionFind.point.pointExtraversion - ExtraversionPoint) }
                })
            }
            else {
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointExtraversion": user.Questions[index].pointExtraversion + (ExtraversionPoint - questionFind.point.pointExtraversion) }
                })
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointExtraversion": ExtraversionPoint }
            })
        }
        async function updateQuestionWithExtraversionPointInQuestion(ExtraversionPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithExtraversionPointInQuestion(ExtraversionPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionPointExtraversion(ExtraversionPoint, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.pointExtraversion == null || questionFind.point.pointExtraversion == "") {
                updateQuestionWithNoExtraversionPointInQuestion(ExtraversionPoint, question)
            }
            else {
                if (ExtraversionPoint == questionFind.point.pointExtraversion) {
                    return;
                }
                else {
                    updateQuestionWithExtraversionPointInQuestion(ExtraversionPoint, question)
                }
            }
        }

        // Update Agreeable Point
        async function doUpdateQuestionWithNoAgreeablePointInDB(AgreeablePoint, question, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (checkQuestion.check(questionFind) == "Agreeable") {
                if (AgreeablePoint > 20) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (AgreeablePoint - 20) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (20 - AgreeablePoint) } }
                    )
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Agreeable") {
                if (AgreeablePoint > 15) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (AgreeablePoint - 15) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (15 - AgreeablePoint) } }
                    )
                }
            }
            else {
                if (AgreeablePoint > 10) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (AgreeablePoint - 10) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (10 - AgreeablePoint) } }
                    )
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointAgreeable": AgreeablePoint }
            })
        }
        async function updateQuestionWithNoAgreeablePointInQuestion(AgreeablePoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithNoAgreeablePointInDB(AgreeablePoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithAgreeablePointInQuestion(AgreeablePoint, question, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            const questionFind = await db.collection("questions").findOne({ "question": question });
            console.log(questionFind.point.pointAgreeable - AgreeablePoint);
            if (questionFind.point.AgreeablePoint > AgreeablePoint) {
                console.log(questionFind.point.AgreeablePoint - AgreeablePoint)
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable - (questionFind.point.pointAgreeable - AgreeablePoint) }
                })
            }
            else {
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointAgreeable": user.Questions[index].pointAgreeable + (AgreeablePoint - questionFind.point.pointAgreeable) }
                })
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointAgreeable": AgreeablePoint }
            })
        }
        async function updateQuestionWithAgreeablePointInQuestion(AgreeablePoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithAgreeablePointInQuestion(AgreeablePoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionPointAgreeable(AgreeablePoint, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.pointAgreeable == null || questionFind.point.pointAgreeable == "") {
                updateQuestionWithNoAgreeablePointInQuestion(AgreeablePoint, question)
            }
            else {
                if (AgreeablePoint == questionFind.point.pointAgreeable) {
                    return;
                }
                else {
                    updateQuestionWithAgreeablePointInQuestion(AgreeablePoint, question)
                }
            }
        }

        // Update Neuroticism Point
        async function doUpdateQuestionWithNoNeuroticismPointInDB(NeuroticismPoint, question, index, id, phoneNumber) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            var user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            if (checkQuestion.check(questionFind) == "Neuroticism") {
                if (NeuroticismPoint > 20) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (NeuroticismPoint - 20) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (20 - NeuroticismPoint) } }
                    )
                }
            }
            else if (checkQuestion.checkMedium(questionFind) == "Neuroticism") {
                if (NeuroticismPoint > 15) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (NeuroticismPoint - 15) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (15 - NeuroticismPoint) } }
                    )
                }
            }
            else {
                if (NeuroticismPoint > 10) {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (NeuroticismPoint - 10) } }
                    )
                }
                else {
                    await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id },
                        { "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (10 - NeuroticismPoint) } }
                    )
                }
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointNeuroticism": NeuroticismPoint }
            })
        }
        async function updateQuestionWithNoNeuroticismPointInQuestion(NeuroticismPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithNoNeuroticismPointInDB(NeuroticismPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function doUpdateQuestionWithNeuroticismPointInQuestion(NeuroticismPoint, question, index, id, phoneNumber) {
            const user = await db.collection("users").findOne({ "phoneNumber": phoneNumber });
            const questionFind = await db.collection("questions").findOne({ "question": question });
            console.log(questionFind.point.pointNeuroticism - NeuroticismPoint);
            if (questionFind.point.NeuroticismPoint > NeuroticismPoint) {
                console.log(questionFind.point.NeuroticismPoint - NeuroticismPoint)
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism - (questionFind.point.pointNeuroticism - NeuroticismPoint) }
                })
            }
            else {
                await db.collection("users").updateOne({ "phoneNumber": phoneNumber, "Questions.id": id }, {
                    "$set": { "Questions.$.pointNeuroticism": user.Questions[index].pointNeuroticism + (NeuroticismPoint - questionFind.point.pointNeuroticism) }
                })
            }
            await db.collection("questions").updateOne({ "question": question }, {
                "$set": { "pointNeuroticism": NeuroticismPoint }
            })
        }
        async function updateQuestionWithNeuroticismPointInQuestion(NeuroticismPoint, question) {
            var listUser = await db.collection("users").find({}).toArray();
            for (var i = 0; i < listUser.length; i++) {
                for (var j = 0; j < listUser[i].Questions.length; j++) {
                    for (var k = 0; k < listUser[i].Questions[j].questionsDate.length; k++) {
                        if (question == listUser[i].Questions[j].questionsDate[k].question) {
                            doUpdateQuestionWithNeuroticismPointInQuestion(NeuroticismPoint, question, j, listUser[i].Questions[j].id, listUser[i].phoneNumber);
                        }
                    }
                }
            }
        }
        async function updateQuestionPointNeuroticism(NeuroticismPoint, question) {
            const questionFind = await db.collection("questions").findOne({ "question": question });
            if (questionFind.point.pointNeuroticism == null || questionFind.point.pointNeuroticism == "") {
                updateQuestionWithNoNeuroticismPointInQuestion(NeuroticismPoint, question)
            }
            else {
                if (NeuroticismPoint == questionFind.point.pointNeuroticism) {
                    return;
                }
                else {
                    updateQuestionWithNeuroticismPointInQuestion(NeuroticismPoint, question)
                }
            }
        }
        updateQuestionPointOpenness(OpennessPoint, question);
        updateQuestionPointConscientious(ConscientiousPoint, question);
        updateQuestionPointExtraversion(ExtraversionPoint, question);
        updateQuestionPointAgreeable(AgreeablePoint, question);
        updateQuestionPointNeuroticism(NeuroticismPoint, question);
    }
}
