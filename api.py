import jwt
import mlflow.pyfunc
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import settings
import time
import logging
from sqlalchemy.orm import Session


engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    sentiment = Column(String(50))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)


class TweetRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    inference_time: float


try:
    mlflow_model = mlflow.pyfunc.load_model(settings.MODEL_URI)
    print(f"✅ MLflow Model Loaded from: {settings.MODEL_URI}")
except Exception as e:
    mlflow_model = None
    print(f"❌ Model Load Error: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_smart_identifier)
app = FastAPI(title="Airline Sentiment Enterprise API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # هيقرأ القائمة اللي حددناها في الـ env
    allow_credentials=True,                 # مهم عشان الـ Tokens اللي عندك
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


logging.basicConfig(
    filename="sentiment_activity.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict(
        payload: TweetRequest,
        request: Request,
        background_tasks: BackgroundTasks,
):
    if mlflow_model is None:
        logger.error("Model access attempted but mlflow_model is None")
        raise HTTPException(status_code=503, detail="Model not loaded on server.")

    try:
        start_time = time.time()
        df = pd.DataFrame({"text": [payload.text]})
        result_df = mlflow_model.predict(df)

        sentiment = str(result_df["label"].iloc[0])
        confidence = float(result_df["confidence"].iloc[0])

        inference_time = round(time.time() - start_time, 4)

        background_tasks.add_task(db_log_prediction, payload.text, sentiment, confidence)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "inference_time": inference_time
        }
    except Exception as e:
        logger.error(f"Prediction Error Details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


def db_log_prediction(text: str, sentiment: str, confidence: float):
    new_db = SessionLocal()
    try:
        log_entry = PredictionLog(text=text, sentiment=sentiment, confidence=confidence)
        new_db.add(log_entry)
        new_db.commit()
        logger.info(f"Log saved for sentiment: {sentiment}")
    except Exception as e:
        logger.error(f"Failed to save background log: {str(e)}")
    finally:
        new_db.close()

# --- جلب سجل التوقعات (الـ History) ---

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    """
    هذا الـ Endpoint يقوم بجلب آخر 50 عملية تحليل من قاعدة البيانات
    ليتم عرضها في الـ Dashboard الخاص بـ Streamlit.
    """
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(50).all()
        return logs
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch history from database")