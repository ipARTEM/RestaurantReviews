import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Создание приложения FastAPI
app = FastAPI()

# Загрузка предобученного токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('./model10/tokenizer10')
model = TFBertForSequenceClassification.from_pretrained('./model10/sentiment_model10')

# Модель данных для входного запроса
class Review(BaseModel):
    review: str

# Функция для токенизации данных
def tokenize_review(review, tokenizer, max_length=128):
    encoded_data = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True
    )
    return tf.convert_to_tensor([encoded_data['input_ids']]), tf.convert_to_tensor([encoded_data['attention_mask']])

# Маршрут для классификации новых отзывов
@app.post("/classify")
def classify_review(review: Review):
    try:
        # Токенизация нового отзыва
        input_ids, attention_mask = tokenize_review(review.review, tokenizer)

        # Предсказание
        logits = model(input_ids, attention_mask=attention_mask).logits
        prediction = tf.nn.softmax(logits, axis=-1)
        label = tf.argmax(prediction, axis=1).numpy()[0]

        # Возврат результата классификации
        return {"label": int(label)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Маршрут для обработки GET-запросов 
@app.get("/")
def read_root():
    return {"message": "Welcome to the sentiment analysis API. Use POST /classify to classify reviews."}

# Запуск приложения с помощью Uvicorn           # uvicorn main:app --reload --port 8888
if __name__ == "__main__":                      # uvicorn main:app --reload --port=8888
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)

# Пример запроса с использованием curl:
# curl -X POST http://127.0.0.1:8888/classify -H "Content-Type: application/json" -d '{"review": "The food was amazing and the service was excellent"}'

