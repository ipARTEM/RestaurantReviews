import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from collections import Counter
import re

# Создание приложения FastAPI
app = FastAPI()

# Загрузка предобученного токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('./model10/tokenizer10')
model = TFBertForSequenceClassification.from_pretrained('./model10/sentiment_model10')

# Настройка SQLAlchemy
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Модель базы данных для хранения результатов классификации
class ReviewResult(Base):
    __tablename__ = "review_results"
    id = Column(Integer, primary_key=True, index=True)
    review = Column(Text, index=True)
    label = Column(Integer)

# Создание таблиц в базе данных
Base.metadata.create_all(bind=engine)

# Зависимость для создания новой сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
def classify_review(review: Review, db: Session = Depends(get_db)):
    try:
        # Токенизация нового отзыва
        input_ids, attention_mask = tokenize_review(review.review, tokenizer)

        # Предсказание
        logits = model(input_ids, attention_mask=attention_mask).logits
        prediction = tf.nn.softmax(logits, axis=-1)
        label = tf.argmax(prediction, axis=1).numpy()[0]

        # Сохранение результата в базу данных
        review_result = ReviewResult(review=review.review, label=int(label))
        db.add(review_result)
        db.commit()
        db.refresh(review_result)

        # Возврат результата классификации
        return {"label": int(label)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Маршрут для извлечения всех результатов классификации
@app.get("/results")
def get_results(db: Session = Depends(get_db)):
    results = db.query(ReviewResult).all()
    return results

# Маршрут для подсчета общего количества отзывов
@app.get("/count")
def get_count(db: Session = Depends(get_db)):
    count = db.query(ReviewResult).count()
    return {"count": count}

# Маршрут для подсчета количества позитивных и негативных отзывов
@app.get("/count_by_label")
def get_count_by_label(db: Session = Depends(get_db)):
    positive_count = db.query(ReviewResult).filter(ReviewResult.label == 1).count()
    negative_count = db.query(ReviewResult).filter(ReviewResult.label == 0).count()
    return {"positive_count": positive_count, "negative_count": negative_count}

# Маршрут для извлечения отзывов по метке
@app.get("/reviews_by_label")
def get_reviews_by_label(label: int, db: Session = Depends(get_db)):
    reviews = db.query(ReviewResult).filter(ReviewResult.label == label).all()
    return reviews

# Функция для очистки текста от нежелательных символов и токенизации
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.lower().split()
    return tokens

# Маршрут для анализа частоты слов
@app.get("/word_frequency")
def get_word_frequency(label: int, db: Session = Depends(get_db)):
    reviews = db.query(ReviewResult).filter(ReviewResult.label == label).all()
    all_words = []
    for review in reviews:
        tokens = preprocess_text(review.review)
        all_words.extend(tokens)
    word_freq = Counter(all_words)
    return word_freq.most_common(10)

    
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

