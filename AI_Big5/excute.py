from nltk.stem import WordNetLemmatizer
from words import question_word
from words import question
import random
from random import choice
from nltk.corpus import stopwords
from flask import Flask, request
from pprint import pprint
from Questgen import main
from pandas import DataFrame
from nltk.tokenize import word_tokenize
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('stopwords')
# convert the dictionary objects to dataframe
# items_df = DataFrame(item_details)
# print(items_df)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def get_database():
    from pymongo import MongoClient
    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    CONNECTION_STRING = "mongodb+srv://GreenBig5:1QgvDJucHUpHxEbn@greenbig5.uszbe.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient

    client = MongoClient(CONNECTION_STRING)

    # Create the database for our example (we will use the same database throughout the tutorial
    return client['green_big_5']


dbname = get_database()
collection_name = dbname["questions"]
pattern_questions = dbname["pattern_questions"]
pattern_questions.insert_one(question)
item_details = collection_name.find()
for item in item_details:
    # This does not give a very readable output
    print(item)

qe = main.BoolQGen()
payload = {
    "input_text": """
... If you wish to make some apples from scratch,
... you must first invent the animals. It's very delicious"""
}

words = word_tokenize(payload['input_text'])
filtered_list = [x for x in words if x.casefold() not in stop_words]
structure_word = nltk.pos_tag(filtered_list)
print(structure_word)


def genQ():
    words_from_user = word_tokenize(payload['input_text'])
    noun_singular_in_words = []
    noun_plural_in_words = []
    verb_origin_in_words = []
    adjective_in_words = []
    passive_questions = []
    unpassive_questions = []
    for_user_questions = []

    ques = ''
    for pair in structure_word:
        if pair[1] == "NN" or pair[1] == 'NNP':
            noun_singular_in_words.append(lemmatizer.lemmatize(pair[0]))
        if pair[1] == "NNS" or pair[1] == 'NNPS':
            noun_plural_in_words.append(lemmatizer.lemmatize(pair[0]))
        if pair[1] == "VB":
            verb_origin = lemmatizer.lemmatize(
                pair[0], pos='v')
            verb_origin_in_words.append(verb_origin)
        if pair[1].casefold() == "JJ":
            adjective_in_words.append(pair[0])
    if len(noun_singular_in_words) > 0:
        ques_for_noun_singular = '{}' + noun_singular_in_words[0] + '?'
        starting_questions = []
        while(len(starting_questions) == 2):
            x = random.choice(question_word['singular'])
            starting_questions.append(
                x) if x not in starting_questions else False
        
        for starting_question in starting_questions:
            
        
        if question_word['singular'][number_question[0]] == 'Does':
            # rand = random.randrange(0, len(question['user']))
            rand = 0
            for rand_ques in question['user'][rand]['favorite']:
                if (rand == 0):
                    full_ques = ques.format(
                        ques_word + ' you ' + rand_ques + ' ')
                    for_user_questions.append(full_ques)
            return for_user_questions


print(genQ())
# output = qe.predict_boolq(payload)
# print(output)
