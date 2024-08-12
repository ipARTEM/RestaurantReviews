import openai

import openaikey

# Вывод текущей версии библиотеки OpenAI
print(f"OpenAI Library Version: {openai.__version__}")

# Установка ключа API OpenAI
openai.api_key =openaikey.KEY 

# Попытка использовать метод ChatCompletion
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=150,
        temperature=0.9,
    )
    print("ChatCompletion method works.")
    print(response.choices[0].message["content"].strip())
except Exception as e:
    print(f"ChatCompletion method error: {e}")

# Попытка использовать метод Completion
try:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Hello, how are you?",
        max_tokens=150,
        temperature=0.9,
    )
    print("Completion method works.")
    print(response.choices[0].text.strip())
except Exception as e:
    print(f"Completion method error: {e}")
