import time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

model = T5ForConditionalGeneration.from_pretrained(
    'ramsrigouthamg/t5_boolean_questions')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)


def greedy_decoding(inp_ids, attn_mask):
    greedy_output = model.generate(
        input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
    Question = tokenizer.decode(
        greedy_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return Question.strip().capitalize()


def beam_search_decoding(inp_ids, attn_mask):
    beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                                 num_beams=10,
                                 num_return_sequences=7,
                                 no_repeat_ngram_size=2,
                                 early_stopping=True
                                 )
    Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
                 beam_output]
    return [Question.strip().capitalize() for Question in Questions]


def topkp_decoding(inp_ids, attn_mask):
    topkp_output = model.generate(input_ids=inp_ids,
                                  attention_mask=attn_mask,
                                  max_length=256,
                                  do_sample=True,
                                  top_k=40,
                                  top_p=0.80,
                                  num_return_sequences=3,
                                  no_repeat_ngram_size=2,
                                  early_stopping=True
                                  )
    Questions = [tokenizer.decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in topkp_output]
    return [Question.strip().capitalize() for Question in Questions]


# passage = "Starlink, of SpaceX, is a satellite constellation project being developed by Elon Musk and team to give satellite Internet go-to access for people in any part of the world. The plan is to comprise thousands of mass-delivered little satellites in low Earth circle, orbit, working in mix with ground handheld devices, for instance, our iPhones. Elon Musk speaks about it as a grand Idea that could change the way we view and access the world around us."
# truefalse ="yes"
# passage ="About 400 years ago, a battle was unfolding about the nature of the Universe. For millennia, astronomers had accurately described the orbits of the planets using a geocentric model, where the Earth was stationary and all the other objects orbited around it."
# truefalse ="yes"
passage = '''London Wildlife Trust, founded in 1981, is the local nature conservation charity for Greater London. It is one of 46 members of the Royal Society of Wildlife Trusts (known as The Wildlife Trusts), each of which is a local nature-conservation charity for its area. The trust aims to protect London\'s wildlife and wild spaces, and it manages over 40 nature reserves in Greater London. The trust\'s oldest reserves include Sydenham Hill Wood (pictured), which was managed by Southwark Wildlife Group before 1982 and was thus already a trust reserve at that date. The campaign to save Gunnersbury Triangle began that same year, succeeding in 1983 when a public inquiry ruled that the site could not be developed because of its value for nature. The trust has some 50 members of staff and 500 volunteers who work together on activities such as water management, chalk grassland restoration, helping people with special needs, and giving children an opportunity to go pond-dipping.'''
truefalse = "yes"

# passage ="The US has passed the peak on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month. The US has over 637000 confirmed Covid-19 cases and over 30826 deaths, the highest for any country in the world."
# truefalse = "yes"


text = "truefalse: %s passage: %s </s>" % (passage, truefalse)


max_len = 256


# print("Context: ", passage)
# print ("\nGenerated Question: ",truefalse)

# output = greedy_decoding(input_ids,attention_masks)
# print ("\nGreedy decoding:: ",output)

# output = beam_search_decoding(input_ids, attention_masks)
# print("\nBeam decoding [Most accurate questions] ::\n")
# for out in output:
#     print(out)


def generate_boolean_questions(text):
    passage = text
    text = "truefalse: %s passage: %s </s>" % (passage, truefalse)
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(
        device), encoding["attention_mask"].to(device)
    q1 = beam_search_decoding(input_ids, attention_masks)
    q2 = topkp_decoding(input_ids, attention_masks)
    return q1 + q2


# print(generate_boolean_questions('London Wildlife Trust, founded in 1981, is the local nature conservation charity for Greater London. It is one of 46 members of the Royal Society of Wildlife Trusts (known as The Wildlife Trusts), each of which is a local nature-conservation charity for its area. The trust aims to protect London\'s wildlife and wild spaces, and it manages over 40 nature reserves in Greater London. The trust\'s oldest reserves include Sydenham Hill Wood (pictured), which was managed by Southwark Wildlife Group before 1982 and was thus already a trust reserve at that date. The campaign to save Gunnersbury Triangle began that same year, succeeding in 1983 when a public inquiry ruled that the site could not be developed because of its value for nature. The trust has some 50 members of staff and 500 volunteers who work together on activities such as water management, chalk grassland restoration, helping people with special needs, and giving children an opportunity to go pond-dipping.'))


# output = topkp_decoding(input_ids, attention_masks)
# print(
#     "\nTopKP decoding [Not very accurate but more variety in questions] ::\n")
# for out in output:
#     print(out)


# print("\nTime elapsed ", end-start)
# print("\n")
