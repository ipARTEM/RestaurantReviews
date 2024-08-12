import requests

# URL вашего API
url = "http://127.0.0.1:8000/chat"

# Пример сообщения для чатбота
message = {"message": "Hello, how are you?"}

# Отправка POST-запроса и вывод ответа
response = requests.post(url, json=message)
print("Response from OpenAI:", response.json())
