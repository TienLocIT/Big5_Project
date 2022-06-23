from Questgen import main
from generate import gen_text_by_text_k2t
from textgenrnn1 import textgen_rb_gen_text_by_text
from auto_generate_question import main as gen_questions2
import nltk
from synonyms import find_synonyms
from pprint import pprint
from generate_boolean_questions_using_T5_transformer import t5_inference

generate_boolean_questions = t5_inference.generate_boolean_questions
qe = main.BoolQGen()


def gen_questions(text):
    array = gen_questions2.generate_questions(text)
    result = []
    for question in array:
        payload = {"input_text": question}
        output = qe.predict_boolq(payload)
        result += output['Boolean Questions']
    return result


def gen_ques_for_add_to_db(dataArr, txt, medium=True, high=False, combine_text=False):
    dataArr = [x.title() for x in dataArr]
    text = (' ').join(gen_text_by_text_k2t(txt))
    payload = {"input_text": txt + ' ' + text}
    if medium == True and combine_text == True:
        texts = textgen_rb_gen_text_by_text(txt)
        texts_join = ('. ').join(texts)
        payload = {"input_text": txt + ' ' +
                   ' ' + texts_join + ' ' + text + ' '}
    output = qe.predict_boolq(payload)
    bool_questions = output['Boolean Questions']
    # bool_question1 = gen_questions(payload['input_text'])
    # book_questions2 = generate_boolean_questions(payload['input_text'])
    # bool_questions += bool_question1 + book_questions2

    # if medium == True and combine_text == False:
    #     texts = textgen_rb_gen_text_by_text(txt)
    #     texts_join = ('. ').join(texts)
    #     bool_questions += qe.predict_boolq({"input_text": texts_join})[
    #         'Boolean Questions'] + gen_questions(texts_join)

    boolean_questions_not_distinc = []

    results = []
    for question in bool_questions:
        if question not in results and type(question) != None:
            tags = []
            for (word, pos) in nltk.pos_tag(
                    nltk.word_tokenize(question)):
                if pos[0] == 'N' and word.upper().strip() not in tags:
                    tags.append(word.upper().strip())
            boolean_questions_not_distinc.append({
                "question": question,
                "big5EnvIndicator": (' ').join(dataArr),
                'tags': find_synonyms(dataArr),
            })
            results.append(question)

    return boolean_questions_not_distinc


res = gen_ques_for_add_to_db(['Smoke', 'Vulnerability'], combine_text=True,
                             txt='Identifying communities vulnerable to adverse health effects from exposure to wildfire smoke may help prepare responses, increase the resilience to smoke and improve public health outcomes during smoke days. ')
# print(generate_boolean_questions('Identifying communities vulnerable to adverse health effects from exposure to wildfire smoke may help prepare responses, increase the resilience to smoke and improve public health outcomes during smoke days.'))

print("\n")
print("------X------")
print("Start  output:\n")
pprint(res)
print("")
print("End OutPut")
print("-----X-----\n\n")
