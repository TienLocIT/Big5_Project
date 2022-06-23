module.exports = {
    check: (question) => {
        if (question.personality[0].Openness == "High") {
            return "Openness"
        }
        else if (question.personality[1].Conscientious == "High") {
            return "Conscientious"
        }
        else if (question.personality[2].Extraversion == "High") {
            return "Extraversion"
        }
        else if (question.personality[3].Agreeable == "High") {
            return "Agreeable"
        }
        else if (question.personality[4].Neuroticism == "High") {
            return "Neuroticism"
        }
    },
    checkMedium: (question) => {
        var arrayMedium = [];
        if (question.personality[0].Openness == "Medium") {
            arrayMedium.push("Openness");
        }
        if (question.personality[1].Conscientious == "Medium") {
            arrayMedium.push("Conscientious");
        }
        if (question.personality[2].Extraversion == "Medium") {
            arrayMedium.push("Extraversion")
        }
        if (question.personality[3].Agreeable == "Medium") {
            arrayMedium.push("Agreeable")
        }
        if (question.personality[4].Neuroticism == "Medium") {
            arrayMedium.push("Neuroticism")
        }
        return arrayMedium;
    }
}