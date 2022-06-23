from pprint import pprint
from generate import gen_text_by_key_k2t
from textgenrnn1 import textgen_rn
from Questgen import main
from auto_generate_question import main as gen_questions2
from synonyms import find_synonyms

qe = main.BoolQGen()


def gen_questions(text):
    array = gen_questions2.generate_questions(text)
    result = []
    for question in array:
        payload = {"input_text": question}
        output = qe.predict_boolq(payload)
        result += output['Boolean Questions']
    return result

# def gen_questions(text):
#     newText = (' ').join(gen_questions2.generate_questions(text))
#     payload = {"input_text": newText}
#     output = qe.predict_boolq(payload)
#     result = output['Boolean Questions']
#     return result


def analyze_data(dataArr):
    dataArr = [x.title() for x in dataArr]
    # synonym_nouns = find_synonyms(dataArr, False)
    text = (' ').join(gen_text_by_key_k2t(dataArr))
    payload = {"input_text": text}
    texts = textgen_rn(dataArr[0] + ' ' + dataArr[1])
    # texts = textgen_rn(text)
    texts_join = ('. ').join(texts)
    payload = {"input_text": text + texts_join}
    bool_question1 = gen_questions(payload['input_text'])
    bool_questions = bool_question1

    boolean_questions_not_distinc = []

    results = []
    for question in bool_questions:
        if question not in results and type(question) != None:
            boolean_questions_not_distinc.append({
                "question": question,
                "big5EnvIndicator": (' ').join(dataArr),
                # "keyword": dataArr[1],
                # "facet": dataArr[0],
                # 'tags': find_synonyms(dataArr),
            })
            results.append(question)
    return boolean_questions_not_distinc


# res = analyze_data(['Rice Field', 'Ideas'])


# print("\n")
# print("------X------")
# print("Start  output:\n")
# pprint(res)
# print("")
# print("End OutPut")
# print("-----X-----\n\n")
