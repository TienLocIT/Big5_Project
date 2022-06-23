const client = require("../../mongoDB");
module.exports = {
    changeDisplayName: async (phoneNumber, displayName) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        db.collection("users").updateOne({ "phoneNumber": phoneNumber }, {
            "$set": {
                "displayName": displayName
            }
        })
    },
    changePhoneNumber: async (phoneNumber, phoneNumberChange) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        db.collection("users").updateOne({ "phoneNumber": phoneNumber }, {
            "$set": {
                "phoneNumber": phoneNumberChange
            }
        })
    },
}
