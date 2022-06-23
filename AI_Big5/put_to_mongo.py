from words import question_word
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('stopwords')
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
questions_from_db = dbname['questions']
big_five_predict = dbname['big_five_predict']
item_details = questions_from_db.find()

big_five_items = [{'words': [], 'model': "Openness"},
                  {'words': [], 'model': "Conscientious"},
                  {'words': [], 'model': "Extraversion"},
                  {'words': [], 'model': "Agreeable"},
                  {'words': [], 'model': "Neuroticism"}]


def update_mongo():
    for item in item_details:
        words = word_tokenize(item['question'])
        pair_words = nltk.pos_tag(words)
        _words = []
        for pair in pair_words:
            if 'VB' in pair[1] and pair[0].upper() not in _words and pair[0].capitalize() not in question_word['singular'] and pair[0].capitalize() not in question_word['plural']:
                _words.append(pair[0].upper())

        personalities = item['personality']
        for personality in personalities:
            if personality.get('Openness') == 'High':
                for check_word in _words:
                    if check_word not in big_five_items[0]['words']:
                        big_five_items[0]['words'].append(check_word)
            elif personality.get('Conscientious') == 'High':
                for check_word in _words:
                    if check_word not in big_five_items[1]['words']:
                        big_five_items[1]['words'].append(check_word)
            elif personality.get('Extraversion') == 'High':
                for check_word in _words:
                    if check_word not in big_five_items[2]['words']:
                        big_five_items[2]['words'].append(check_word)
            elif personality.get('Agreeable') == 'High':
                for check_word in _words:
                    if check_word not in big_five_items[3]['words']:
                        big_five_items[3]['words'].append(check_word)
            elif personality.get('Neuroticism') == 'High':
                for check_word in _words:
                    if check_word not in big_five_items[4]['words']:
                        big_five_items[4]['words'].append(check_word)


update_mongo()
print(big_five_items)
for item in big_five_items:
    big_five_predict.insert_one(item)
