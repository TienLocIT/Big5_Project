import spacy
import en_core_web_sm

# nlp = spacy.load("en_core_web_lg")
nlp = en_core_web_sm.load()


doc1 = nlp('the person wear red T-shirt')
doc2 = nlp('this person is walking')
doc3 = nlp('the boy wear red T-shirt')


print("A", doc1.similarity(doc2))
print("A", doc1.similarity(doc3))
print("A", doc2.similarity(doc3))
