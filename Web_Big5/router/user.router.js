const controller = require("../controllers/user.controller")
const express = require("express")
const router = express.Router()
router.post("/register", controller.register);
router.post('/login', controller.login);
router.post('/postQuestions', controller.postQuestions);
router.get("/getUser", controller.getUser);
router.post("/postAnswer", controller.postAnswerQuestion);
router.post("/changeDisplayName", controller.changeDisplayName)
router.post("/changePhoneNumber", controller.changePhoneNumber);
router.post("/address/update", controller.updateAddress);


// router.post("/updateQuestionPersonalityLevel",controller.updateQuestionPersonalityLevel);
// router.post("/updateQuestionPersonalityPoint",controller.updateQuestionPersonalityPoint);
// router.post("/updateQuestionLevelPoint",controller.updateQuestionLevelPoint);
module.exports = router;