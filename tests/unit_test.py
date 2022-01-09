import requests

url="http://localhost:5000/predict"
image= open('', 'rb')
resp = requests.post(url, file={'file': image})

print(resp.text)