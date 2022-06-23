# from auto_generate_question import main
# print(main.generate_questions("""My best friend and I have been studying in the same school since kindergarten. We have been classmates each year at
# school. We share a very close bond and have a special friendship that we cherish and treasure. My friend is my
# partner, sitting beside me in class. She is kindly and helpful, and if I have any difficulties in understanding any
# topic in my studies, or in completing my homework or school project, she helps me. She is brilliant in mathematics
# and the sciences, while I am good at English. So we both help each other in whatever way possible. She helps me
# without ever belittling me. I greatly appreciate the quality in her. She does not make me feel obliged.""")
#       )

# x = ['There is a restaurant that has a price range of more than £30 called the restaurant called The Rams.', 'A restaurant that has a price range of more than £30 is called the restaurant.', 'A restaurant that has a price range of more than £30 is called the restaurant called the Rams.',
#      'There is a restaurant that serves Japanese food called the Food restaurant called The Ram Rebellion.', 'There is a restaurant with a price range of £20-25.', 'There is a restaurant that serves sushi.', 'There is a restaurant that serves moderately priced food.', 'There is a restaurant called The Food is not rated highly.']

# print((' ').join(x))

# import spacy
import nltk
# from textblob import TextBlob
# txt = """There is a restaurant that has a price range of more than £30 called the restaurant called The Rams.', 'A restaurant that has a price range of more than £30 is called the restaurant.', 'A restaurant that has a price range of more than £30 is called the restaurant called the Rams."""
# blob = TextBlob(txt)
# print(blob.noun_phrases)


txt = """'There is a restaurant that has a price range of more than £30 called the restaurant called The Rams.', 'A restaurant that has a price range of more than £30 is called the restaurant.', 'A restaurant that has a price range of more than £30 is called the restaurant called the Rams."""
print([word for (word, pos) in nltk.pos_tag(
    nltk.word_tokenize(txt)) if pos[0] == 'N'])
