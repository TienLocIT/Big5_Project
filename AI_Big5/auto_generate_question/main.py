from auto_generate_question import aqgFunction


# Main Function
def generate_questions(x=""):
    # Create AQG object
    aqg = aqgFunction.AutomaticQuestionGenerator()

    # inputTextPath = "x.txt"
    # readFile = open(inputTextPath, 'r', encoding="utf8")
    # #readFile = open(inputTextPath, 'r+', encoding="utf8", errors = 'ignore')

    # inputText = readFile.read()
    inputText = '''{}'''

    questionList = aqg.aqgParse(inputText.format(x))
    return aqg.display(questionList)

    # aqg.DisNormal(questionList)

    return 0


# Call Main Function
if __name__ == "__main__":
    generate_questions()
