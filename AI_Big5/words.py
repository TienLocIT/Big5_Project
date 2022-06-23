question_word = {
    "singular": ["Does", "Is", 'Has', 'Doesn\'t', "Isn\'t"],
    "plural": ["Do", "Are", 'Have', "Aren\'t"],
    "other": ["Would", 'Would you like', 'Looks like', 'Can']
}
question_verb_word = []
x = ["love", "like", "understand", "hate", "want",
     "need", "mind", "know", "believe", "trust"],
structures = [
    {
        "starting_words": ["DOES", "DO", "CAN", "CAN'T", "DON'T", "Doesn't"],
        "structure": [
            {
                "combine_with_n": False,
                "starting_verb": [
                    {
                        "tag": "know how to",
                        "verb": ["run", "collapse", "love", "like", "start up", "sit", "eat", "drink", "understand", "play", "hate", "mind", "believe", "trust", "cook", "stand", "fly", "go", "go out", "swim", "jump"],
                    },
                    {
                        "tag": "want to",
                        "verb": ["eat", "start up", "run", "drink", "fly", "go", "go out", "swim", "cook", "jump", "sit", "stand"],
                    },
                ],

            },
            {
                "combine_with_n": True,
                "starting_verb": [
                    {
                        'combine_with_preposition': True,
                        "tag": "know how to",
                        "verb": ["swim", "eat", "run", "jump", "collapse", "fly", "start up"],
                        "preposition": ["with"]
                    }
                ],

            },

        ],
        "verb_about": []
    },
    {
        "starting_words": [""]
    }
]

verb_words = ["love", "like", "understand", "hate",
              "want", "prefer", "need", "mind", "care", "know", "think", "believe", "guess", "mean"]
feeling_verb_words = ['like', 'dislike', 'love',
                      'hate', 'prefer', 'want', 'need', 'mind', 'care']
thought_verb_words = ['know', "know", "think",
                      "understand", "believe", "guess", "mean", 'suppose', 'doubt', 'realize', 'remember', 'forget', 'agree']
sense_verb_words = ['feel', 'hear', 'see',
                    'smell', 'sound', 'taste', 'touch', 'look']
v3 = ["seen", 'had', 'bought', 'dropped',
              'thrown', 'hold', 'broken', 'collected', 'killed']
v1 = ['have', 'eat', 'buy', 'see', 'drop', 'throw',
      'hold', 'get', 'break', 'collect', 'talk', 'kill']

question = {
    "user": [
        {
            "favorite": [
                "like anything than", "like a lot of", "love anything than", "love a lot of",
                "prefer", "hate a lot of", "hate anything than", "like to see"],
        },
        {
            "want_to": ["want to"]
        },
        {"have_ever": ["Have you ever"]},
        {
            "would_like_to": ["Would you like to"]
        },
    ],
    "capabilities": {
        'passive': {
            "can_be": ["can be"]
        },
        'unpassive': {
            "can": ["can"]
        }
    }
}

x = 1


# def get_database():
#     from pymongo import MongoClient
#     CONNECTION_STRING = "mongodb+srv://GreenBig5:1QgvDJucHUpHxEbn@greenbig5.uszbe.mongodb.net/green_big_5?retryWrites=true&w=majority"

#     client = MongoClient(CONNECTION_STRING)

#     return client['green_big_5']


# dbname = get_database()
# suggest_questions_collections = dbname['suggest_questions']
# for x in suggest_questions_collections.find():
#     print(x)
