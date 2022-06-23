from question_generator import questiongenerator
qg = questiongenerator.QuestionGenerator()

text_samples = 'There is a restaurant that has a price range of more than £30 called the restaurant called The Rams. A restaurant that has a price range of more than £30 is called the restaurant. A restaurant that has a price range of more than £30 is called the restaurant called the Rams. There is a restaurant that serves Japanese food called the Food restaurant called The Ram Rebellion. There is a restaurant with a price range of £20-25. There is a restaurant that serves sushi. There is a restaurant that serves moderately priced food. There is a restaurant called The Food is not rated highly.'

x = ('. ').join(['Smoke Compliances the Force A Therapy And State Was Started To Be The Best Trailer', 'Smoke Compliance of Star Wars Problem', 'Smoke Compliance in the San Friend [OC] [1000x1000]', 'Smoke Compliances - The Star Wars', "Smoke Compliance is a big thing about the state of the story of the streets of the state of the state of the state of the marijuana on the time that you can see what they do to stop the state of the state of the same state was the best streamer and I don't know what to do it?", 'Smoke Compliance of Internet Points In Stone Revealed',
                 'Smoke Compliance has to see if any design is the best guitar at the best friend and is my mother in the wall of the movie in a right day in the pirated by the players between a bad angle with a content of the hell as they probably have a single thing the new place and then being managed to say to i', 'Smoke Compliances Bear Bell!', 'Smoke Compliances Competition - One Protesting Facebook Pro Project (Everyone)', 'Smoke Compliances but it was a space for 10 minutes in the Star Mario or How to change the sales and have saved by a big community in my life that we all come out of a company.'])


def gen_3_from_text(text):
    filtered_questions = []
    questions = qg.generate(text, use_evaluator=False, num_questions=10)
    for question in questions:
        filtered_questions.append(question['question'])
    return filtered_questions


# print(gen_3_from_text(x))
