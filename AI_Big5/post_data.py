# import string
# from tokenize import String
# from unittest import result
# from fastapi import FastAPI, Form, Request, Body
# # from starlette.responses import HTMLResponse
# from main import run_model, X_train, y_train
# # from pydantic import BaseModel
# import requests
from gen_text_by_key5 import analyze_data
from pprint import pprint

# #

# app = FastAPI()


# # @app.get('/') #basic get view
# # def basic_view():
# #     return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}

# # @app.get('/http://greenbig5.herokuapp.com/expert/getPointUser')
# # def get_data():
# #     return "hello"

# # @app.get('/predict', response_class=HTMLResponse) #data input by forms
# # def take_inp():
# #     return '''<form method="post">
# #     <input type="submit"/>
# #     </form>'''

# def convert_trait(argument):
#     switcher2 = {
#         0: "No",
#         1: "Yes",
#     }

#     return switcher2.get(argument, "nothing")


# @app.post('/predict')
# def show(body: dict = Body(...)):
#     print(body)
#     API_ENDPOINT = "http://greenbig5.herokuapp.com/expert/getPointUser"

#     # your API key here
#     API_KEY = "MBe2IdLb-9dhasuidad02316156156552352sdggsdgsdaga92345iosfiashfkja"

#     data = body

#     # sending post request and saving response as response object
#     r = requests.post(url=API_ENDPOINT, data=data)

#     # extracting response text
#     pastebin_url = r.json()
#     [float(i) for i in pastebin_url]
#     input_model = [pastebin_url]
#     result = run_model(X_train, y_train, input_model)
#     output_model = convert_trait(int(result))

#     # res = analyze_data(['Rice Field', 'Ideas'])
#     print('result', result)
#     print('input_model', input_model)

#     return {"OUTPUT_MODEL": output_model}
#     # return {"OUTPUT_MODEL": res}

from fastapi import FastAPI, Form, Request, Body
from starlette.responses import HTMLResponse
from main import run_model, X_train, y_train
from pydantic import BaseModel
import requests


app = FastAPI()


# @app.get('/') #basic get view
# def basic_view():
#     return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}

# @app.get('/http://greenbig5.herokuapp.com/expert/getPointUser')
# def get_data():
#     return "hello"

# @app.get('/predict', response_class=HTMLResponse) #data input by forms
# def take_inp():
#     return '''<form method="post">
#     <input type="submit"/>
#     </form>'''

def convert_trait(argument):
    switcher2 = {
        0: "No",
        1: "Yes",
    }

    return switcher2.get(argument, "nothing")


@app.post('/predict')
def show(body: dict = Body(...)):
    API_ENDPOINT = "http://greenbig5.herokuapp.com/expert/getPointUser"

    # your API key here
    API_KEY = "MBe2IdLb-9dhasuidad02316156156552352sdggsdgsdaga92345iosfiashfkja"
    body['key'] = API_KEY
    data = body

    # sending post request and saving response as response object
    r = requests.post(url=API_ENDPOINT, data=data)

    # extracting response text
    pastebin_url = r.json()
    [float(i) for i in pastebin_url]
    input_model = [pastebin_url]
    result = run_model(X_train, y_train, input_model)
    output_model = convert_trait(int(result))
    print(input_model)
    print(data)

    return {"OUTPUT_MODEL": output_model}


@app.post('/generate-questions')
def handleGenerate(body=Body(...)):
    print(body)
    res = analyze_data(body)
    pprint(res)

    return {"data": res}
