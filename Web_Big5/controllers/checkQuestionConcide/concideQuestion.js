module.exports = {
    concideQuestion: (question, questionArray) => {
        for (var i = 0; i < questionArray.length; i++) {
            if (question == questionArray[i].question) {
                return true;
            }
        }
        return false;
    },
    coincideQuestionDB: (question, questionArray) => {
        for (var i = 0; i < questionArray.length; i++) {
            var questionChildArray = questionArray[i].questionsDate;
            for (var j = 0; j < questionChildArray.length; j++) {
                if (question == questionChildArray[j].question) {
                    return true;
                }
            }
        }
        return false;
    }
}