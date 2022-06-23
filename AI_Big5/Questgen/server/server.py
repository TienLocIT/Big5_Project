from flask import Flask
from pprint import pprint
from Questgen import main
qe = main.BoolQGen()
payload = {
    "input_text": "Ronaldo was born on February 5, 1985, in Funchal, Madeira, Portugal, a small island off the western coast of the country. Ronaldo is the youngest of four children born to Maria Dolores dos Santos and José Dinis Aveiro. He was named after Ronald Reagan, one of his father's favorite actors."
}
output = qe.predict_boolq(payload)
print("A")
pprint(output)

app = Flask(__name__)


@app.route("/members")
def members():
    return output


if __name__ == "__main__":
    app.run(debug=True)
