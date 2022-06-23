from textgenrnn import textgenrnn
import random

textgen = textgenrnn.textgenrnn()
# generated_texts = textgen.generate(
#     n=5, prefix="Smoke Compliance", temperature=0.2, return_as_list=True)
# print(generated_texts, "----------------------------")
# print(textgen.generate(n=5, prefix="Smoke Compliance",
#       temperature=0.5, return_as_list=True))


def textgen_rn(key):
    random_temperature = float("{:.1f}".format(random.randrange(1, 11)*0.1))
    texts1 = textgen.generate(
        n=5, prefix=key, temperature=random_temperature, return_as_list=True)
    # texts2 = textgen.generate(n=5, prefix=key,
    #                           temperature=0.5, return_as_list=True)
    # return texts1 + texts2
    return texts1


def textgen_rb_gen_text_by_text(txt):
    texts1 = textgen.generate(
        n=5, prefix=[txt], temperature=0.2, return_as_list=True)
    return texts1


# print(textgen_rb_gen_text_by_text('Identifying communities vulnerable to adverse health effects from exposure to wildfire smoke may help prepare responses, increase the resilience to smoke and improve public health outcomes during smoke days. '))
# print(textgen_rn('Rice Field Ideas'))
