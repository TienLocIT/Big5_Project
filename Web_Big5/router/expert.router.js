const express = require("express");
const router = express.Router();
const expertController = require("../controllers/expert.controller");
router.post("/sendQuestionToAll",expertController.expertSendQuestionToAll);
router.get("/getQuestion",expertController.expertGetQuestion);
router.post("/testAPI",expertController.testAPI);
router.post("/getPointUser",expertController.getPointForMachineLearning);
router.post("/cloneDocument",expertController.cloneDocument);
module.exports = router;
