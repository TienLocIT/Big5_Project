from auto_generate_question import main as gen_questions2
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# from random import choice
# import random
from pprint import pprint
from Questgen import main
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
import en_core_web_sm
from generate import gen_text_by_key_k2t, gen_text_by_text_k2t
from textgenrnn1 import textgen_rn, textgen_rb_gen_text_by_text
from synonyms import find_synonyms
# from generator_3 import gen_3_from_text

# nlp = spacy.load("en_core_web_lg")
# nlp = en_core_web_sm.load()


def get_database():
    from pymongo import MongoClient
    CONNECTION_STRING = "mongodb+srv://GreenBig5:1QgvDJucHUpHxEbn@greenbig5.uszbe.mongodb.net/green_big_5?retryWrites=true&w=majority"

    client = MongoClient(CONNECTION_STRING)

    return client['green_big_5']


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
dbname = get_database()
qe = main.BoolQGen()

payload = {
    "input_text": """Smoke Compliances Animation Of Lights To His Story Man Who Is So Have To Get Hitting Person That Work"""
}


# big_five_predict = dbname['questions_test']
# big_five_predict_collections = big_five_predict.find()
# question_objects = []
# for item in big_five_predict_collections:
#     question_objects.append(item)
# print(len(question_objects))

suggest_questions_collections = dbname['suggest_questions']


# def classify_trait(arr):
#     for question in arr:
#         points = []
#         max = {"result": None}
#         question_analyse = nlp(question['question'])
#         for i in range(len(question_objects)):
#             question_object_analyse = nlp(question_objects[i]['question'])
#             points.append(
#                 {'index': i, "result": question_analyse.similarity(question_object_analyse),
#                  'compare_with': question_objects[i]})
#         for point in points:
#             if max['result'] is None or max['result'] < point['result']:
#                 max = point
#         question['personality'] = max['compare_with']['personality']


def filter_distinc_question(arr): pass


def gen_questions(text):
    array = gen_questions2.generate_questions(text)
    result = []
    for question in array:
        payload = {"input_text": question}
        output = qe.predict_boolq(payload)
        result += output['Boolean Questions']
    return result


def analyze_data(dataArr, medium=True, high=False, combine_text=False):
    dataArr = [x.title() for x in dataArr]
    text = (' ').join(gen_text_by_key_k2t(dataArr))
    payload = {"input_text": text}
    if medium == True and combine_text == True:
        texts = textgen_rn(dataArr[0] + ' ' + dataArr[1])
        texts_join = ('. ').join(texts)
        payload = {"input_text": text + texts_join}
    output = qe.predict_boolq(payload)
    bool_questions = output['Boolean Questions']
    bool_question1 = gen_questions(payload['input_text'])
    # bool_question2 = output2['Boolean Questions']
    bool_questions += bool_question1

    # if (high == True):
    #     text_gen3 = gen_3_from_text(text)
    #     output2 = qe.predict_boolq({"input_text": (' ').join(text_gen3)})
    #     bool_questions += output2['Boolean Questions']

    if medium == True and combine_text == False:
        texts = textgen_rn(dataArr[0] + ' ' + dataArr[1])
        texts_join = ('. ').join(texts)
        bool_questions += qe.predict_boolq({"input_text": texts_join})[
            'Boolean Questions'] + gen_questions(texts_join)

    boolean_questions_not_distinc = []
    words = []

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


# def gen_ques_for_add_to_db(dataArr, txt, medium=True, high=False, combine_text=False):
#     text = (' ').join(gen_text_by_text_k2t(txt))
#     payload = {"input_text": text}
#     if medium == True and combine_text == True:
#         texts = textgen_rb_gen_text_by_text(txt)
#         texts_join = ('. ').join(texts)
#         payload = {"input_text": text + ' ' + texts_join}
#     output = qe.predict_boolq(payload)
#     bool_questions = output['Boolean Questions']
#     bool_question1 = gen_questions(payload['input_text'])
#     # bool_question2 = output2['Boolean Questions']
#     bool_questions += bool_question1

#     # if (high == True):
#     #     text_gen3 = gen_3_from_text(text)
#     #     output2 = qe.predict_boolq({"input_text": (' ').join(text_gen3)})
#     #     bool_questions += output2['Boolean Questions']

#     if medium == True and combine_text == False:
#         texts = textgen_rn(dataArr[0] + ' ' + dataArr[1])
#         texts_join = ('. ').join(texts)
#         bool_questions += qe.predict_boolq({"input_text": texts_join})[
#             'Boolean Questions'] + gen_questions(texts_join)

#     boolean_questions_not_distinc = []
#     words = []

#     results = []
#     for question in bool_questions:
#         if question not in results and type(question) != None:
#             tags = []
#             for (word, pos) in nltk.pos_tag(
#                     nltk.word_tokenize(question)):
#                 if pos[0] == 'N' and word.upper().strip() not in tags:
#                     tags.append(word.upper().strip())
#             boolean_questions_not_distinc.append({
#                 "question": question,
#                 "big5EnvIndicator": dataArr[0] + " " + dataArr[1],
#                 'tags': tags,
#             })
#             results.append(question)
#     # for question in boolean_questions_not_distinc:
#     #     # words = word_tokenize(question)
#     #     # starting_words = []
#     #     # starting_words.append(words[0].upper()) if words[0].upper(
#     #     # ) not in starting_words else False
#     #     # # filtered_list = word_tokenize(question)
#     #     # structure_word = []
#     #     # structure_word_for_loop = nltk.pos_tag(words)
#     #     # structure_word.extend(structure_word_for_loop)
#     #     # noun = []
#     #     # related_verb = []
#     #     # for pair in structure_word_for_loop:
#     #     #     if "NN" in pair[1]:
#     #     #         n0 = lemmatizer.lemmatize(pair[0], pos='n')
#     #     #         if n0 == pair[0] and n0.upper() not in noun:
#     #     #             noun.append(n0.upper())
#     #     #         elif n0 != pair[0]:
#     #     #             if n0.upper() not in noun:
#     #     #                 noun.append(n0.upper())
#     #     #             if pair[0].upper() not in noun:
#     #     #                 noun.append(pair[0].upper())

#     #     #     if "VB" in pair[1]:
#     #     #         v0 = lemmatizer.lemmatize(pair[0], pos="v")
#     #     #         if v0 == pair[0] and v0.upper() not in related_verb:
#     #     #             related_verb.append(v0.upper())
#     #     #         elif v0 != pair[0]:
#     #     #             if v0.upper() not in related_verb:
#     #     #                 related_verb.append(v0.upper())

#     #     #             if pair[0].upper() not in related_verb:
#     #     #                 related_verb.append(pair[0].upper())

#     #     results.append({
#     #         # "tags": noun,
#     #         # "capabilities": related_verb,
#     #         "question": question,
#     #         # "starting_word": starting_words,
#     #         # "structure_word": structure_word,
#     #         # "personality": None
#     #     })
#     # classify_trait(results)
#     return boolean_questions_not_distinc


res = analyze_data(['Rice Field', 'Ideas'], combine_text=True)


# suggest_questions_collections.insert_many(res)

print("\n")
print("------X------")
print("Start  output:\n")
pprint(res)
print("")
print("End OutPut")
print("-----X-----\n\n")
