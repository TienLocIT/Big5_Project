const client = require("../mongoDB");
const bcrypt = require("bcrypt");
async function checkAdminFunc(string, adminId) {
    const checkAdmin = new Promise((resolve, reject) => {
        bcrypt.compare(string, adminId, function (err, result) {
            if (err) reject(err)
            resolve(result)
        });
    })
    return checkAdmin;
}
module.exports = {
    requireAuth: async (req, res, next) => {
        const dbs = await client.connect().catch(err => {
            console.log(err);
        });
        const db = dbs.db("green_big_5");
        if (!req.signedCookies.admin_id) {
            res.redirect("/login");
            return;
        }
        const admin = await db.collection("admin").findOne({ "id": req.signedCookies.admin_id });
        if (!admin) {
            res.redirect("/login");
            return;
        }
        next();
    },
}
