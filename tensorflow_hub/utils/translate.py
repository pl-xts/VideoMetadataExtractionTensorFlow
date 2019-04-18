import requests

URL = "https://translate.yandex.net/api/v1.5/tr.json/translate"

file = open("./tensorflow_hub/utils/api-key.txt","r")
key = file.read()
file.close()
headers = {'content-type': 'application/x-www-form-urlencoded'}
params = {'key': key,
          'format': 'html'}

def translate(lang, requestBody):
    #requestBody example: "text=[Hello world],[Hello world]"
    params["lang"] = lang
    r = requests.post(url=URL, params=params, data=requestBody, headers=headers)
    response = r.json()
    # print(response)
    if (r.status_code == 200):
        return response
    else:
        return requestBody