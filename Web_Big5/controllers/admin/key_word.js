const client = require("../../mongoDB");
module.exports = {
    getAll: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = await dbs.db("green_big_5");
        try {
            const key_words = await db.collection("key_words").find({}).toArray();
            return res.status(200).json(key_words);
        }
        catch (err) {
            return res.status(200).json(err);
        }

    }
}