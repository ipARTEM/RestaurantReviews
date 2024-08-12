from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

# Создание приложения FastAPI
app = FastAPI()

# Установка ключа API OpenAI
openai.api_key = 'your_openai_api_key_here'

# Модель данных для входного запроса
class UserMessage(BaseModel):
    message: str

# Маршрут для обработки сообщений пользователя и получения ответа от OpenAI
@app.post("/chat")
def chat_with_openai(user_message: UserMessage):
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=f"User: {user_message.message}\nAssistant:",
            max_tokens=150,
            temperature=0.9,
        )
        answer = response.choices[0].text.strip()
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск приложения с помощью Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# .\venvrestreviews\Scripts\activate  
# uvicorn chatbot:app --reload --port 8000