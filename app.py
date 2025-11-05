import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
# --- 1. REMOVED ALL EXTRA IMPORTS ---
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils.logger import logger
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="An API to predict customer subscription probability using our 0.8158 roc auc CatBoost model.",
    version="1.0.0"
)

class BankCustomer(BaseModel):
    # data validation and field are required
    age: int = Field(..., example=45)
    job: str = Field(..., example="blue-collar")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="basic.9y")
    default: str = Field(..., example="no")
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    month: str = Field(..., example="may")
    day_of_week: str = Field(..., example="mon")
    campaign: int = Field(..., example=2)
    pdays: int = Field(..., example=999)
    previous: int = Field(..., example=0)
    poutcome: str = Field(..., example="nonexistent")
    # alias as in traing used as . seperation
    emp_var_rate: float = Field(..., alias="emp.var.rate", example=1.1)
    cons_price_idx: float = Field(..., alias="cons.price.idx", example=93.994)
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", example=-36.4)
    euribor3m: float = Field(..., example=4.857)
    nr_employed: float = Field(..., alias="nr.employed", example=5191.0)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 45, "job": "blue-collar", "marital": "married", "education": "basic.9y",
                "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
                "month": "may", "day_of_week": "mon", "campaign": 2, "pdays": 999,
                "previous": 0, "poutcome": "nonexistent",
                "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
                "euribor3m": 4.857, "nr.employed": 5191.0
            }
        }

try:
    pipeline = PredictPipeline()
    logger.info("API is up: Prediction pipeline loaded successfully.")
except Exception as e:
    logger.error(f"API startup error: Could not load pipeline. {e}")
    pipeline = None

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Welcome to the Bank Marketing Prediction API. Visit /docs for more."}

@app.post("/predict", tags=["Prediction"])
async def predict(customer: BankCustomer):
    """
    Predicts the likelihood of a bank customer subscribing to a term deposit.
    Returns a JSON object with:
     'prediction': 0 (No) or 1 (Yes) based on our optimal 0.71 threshold.
     'probability_of_yes': The raw probability score (0.0 to 1.0).
    """
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please check server logs.")
        
    try:
        data_dict = customer.model_dump(by_alias=True)
        
        for key, value in data_dict.items():
            if isinstance(value, str):
                data_dict[key] = value.lower().strip()

        result = pipeline.predict(data_dict)
        return result

    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app)