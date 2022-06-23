# from textgenrnn import textgenrnn

# textgen = textgenrnn()
# textgen.generate()


# from textgenrnn1 import textgenrnn
from keytotext import pipeline


nlp = pipeline("k2t-base")
nlp1 = pipeline('k2t')
# x = nlp('Smoke Compliance')
# print(type(nlp), nlp(['restaurant', 'Food']),
#       nlp(['restaurant Food']), nlp1(
#           ['restaurant', 'Food']), nlp1(['restaurant Food']))


def gen_text_by_key_k2t(keys):
    text = []
    text = analyze(nlp, keys) + analyze(nlp1, keys)
    return text
    # print(nlp(['restaurant', 'Food']),
    #       nlp(['restaurant Food']), nlp1(
    #     ['restaurant', 'Food']), nlp1(['Food restaurant']))
    # print(nlp(['Food', 'restaurant']),
    #       nlp(['Food restaurant']), nlp1(
    #     ['Food', 'restaurant']), nlp1(['Food restaurant']))


def analyze(a, keys):
    return [a(keys), a([keys[0] + " " + keys[1]]), a([keys[1], keys[0]]), a([keys[1] + " " + keys[0]])]


def gen_text_by_text_k2t(text):
    return [nlp(text)]


# print(gen_text_by_text_k2t('Identifying communities vulnerable to adverse health effects from exposure to wildfire smoke may help prepare responses, increase the resilience to smoke and improve public health outcomes during smoke days. '))
# synonym_nouns = find_synonyms(['Rice Field', 'Ideas'], False)
# print(synonym_nouns)
# print(gen_text_by_key_k2t(['Rice Field', 'Ideas']))

# print(gen_text_by_key_k2t(['Restaurant', 'food']))
