from flask import Flask, request
from pprint import pprint
from Questgen import main
qe = main.BoolQGen()
payload = {
    "input_text": "The aims to protect London's wildlife and wild spaces, and it manages over 40 nature reserves in Greater London."
}
output = qe.predict_boolq(payload)
print(output)
# app = Flask(__name__)


# @app.route("/members", methods=["POST", "GET"])
# def members():
#     return output


# if __name__ == "__main__":
#     app.run(debug=False)
