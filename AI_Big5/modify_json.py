import json
from read_json import read_data
data = "A"

with open('words.json', "w") as f:
    x = read_data(None)
    json.dump(x, f, ensure_ascii=False, indent=4)
print("A")
