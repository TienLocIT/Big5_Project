import re
from nltk.corpus import wordnet


def replaceWord(w):
    return re.sub("[_-]", " ", w)


# for ss in wordnet.synsets('Smoke'):
#     # x = re.sub("[_-]", " ", txt)
#     for x in ss.lemma_names():
#         print(replaceWord(x))


def find_synonyms(arr, upper=True):
    tag0 = []
    tag1 = []
    for ss in wordnet.synsets(arr[0]):
        for x in ss.lemma_names():
            w = replaceWord(x)
            if w.upper().strip() not in tag0:
                if upper == True:
                    tag0.append(w.upper().strip())
                else:
                    tag0.append(w.title().strip())

    for ss in wordnet.synsets(arr[1]):
        for x in ss.lemma_names():
            w = replaceWord(x)
            if w.upper().strip() not in tag1:
                if upper == True:
                    tag1.append(w.upper().strip())
                else:
                    tag1.append(w.title().strip())
    combine = tag0 + tag1
    results = []
    if upper == True:
        results = [x.upper()
                   for x in arr if x.upper() not in combine] + combine
    else:
        results = [x.title()
                   for x in arr if x.upper() not in combine] + combine
    return results


# print(find_synonyms(['Smoke', 'Ideas']))
# dataArr = ['ANH YEU', 'em nhieu']
# print([x.title() for x in dataArr])
