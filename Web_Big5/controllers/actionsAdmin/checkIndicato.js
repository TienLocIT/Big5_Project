const client = require("../../mongoDB");
module.exports = {
    checkIndicator: async (req, res) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        })
        const db = dbs.db("green_big_5");
        const indicator = req.query.indicator;
        const indicatorNumber = await db.collection("questions").find({ "bi5EnvIndicator": indicator }).length;
        if (indicatorNumber > 0) {
            return res.status(200).json({ "Notification": "Đã có indicator này" });
        }
        else {
            return res.status(200).json({ "Notification": "Success" });
        }
    }
}