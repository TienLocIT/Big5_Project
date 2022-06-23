from textgenrnn1 import textgenrnn

textgen = textgenrnn()
generated_texts = textgen.generate(
    n=5, prefix="Smoke Compliance", temperature=0.2, return_as_list=True)
print(generated_texts)
print(textgen.generate_samples(prefix="Smoke Compliance"))
