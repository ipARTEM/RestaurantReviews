from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import openai
import openaikey  # Импорт вашего модуля с API-ключом

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники (можете указать конкретный, если нужно)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, PUT, DELETE и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

# Установите ваш API-ключ OpenAI
openai.api_key = openaikey.KEY

@app.post("/chat")
async def chat(prompt: str):
    response = openai.Completion.create(
        engine="ada",
        prompt=prompt,
        max_tokens=100
    )
    return {"response": response.choices[0].text.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
