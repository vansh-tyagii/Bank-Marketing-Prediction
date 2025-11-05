import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils.logger import logger

# chatgpt some test case to check 
def run_prediction_test():
    try:
        pipeline = PredictPipeline()
        
        data_1 = {
            "age": 45, "job": "blue-collar", "marital": "married", "education": "basic.9y",
            "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
            "month": "may", "day_of_week": "mon", "campaign": 2, "pdays": 999,
            "previous": 0, "poutcome": "nonexistent", "emp.var.rate": 1.1,
            "cons.price.idx": 93.994, "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0
        }
        
        data_2 = {
            "age": 22, "job": "student", "marital": "single", "education": "high.school",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "mar", "day_of_week": "tue", "campaign": 1, "pdays": 10, # Contacted 10 days ago
            "previous": 1, "poutcome": "success", "emp.var.rate": -1.8,
            "cons.price.idx": 92.843, "cons.conf.idx": -50.0, "euribor3m": 1.687, "nr.employed": 5008.7
        }

        data_3 = {
            "age": 35, "job": "admin.", "marital": "single", "education": "university.degree",
            "default": "no", "housing": "no", "loan": "no", "contact": "cellular",
            "month": "sep", "day_of_week": "fri", "campaign": 1, "pdays": 999,
            "previous": 0, "poutcome": "nonexistent", "emp.var.rate": -3.4,
            "cons.price.idx": 92.379, "cons.conf.idx": -29.8, "euribor3m": 0.77, "nr.employed": 5017.5
        }
        
        logger.info("--- TEST 1: CONFIDENT NO ---")
        result_1 = pipeline.predict(data_1)
        print(f"Input: {data_1['job']}, {data_1['poutcome']}, {data_1['month']}")
        print(f"Output: {result_1}\n")

        logger.info("--- TEST 2: SURE BET ---")
        result_2 = pipeline.predict(data_2)
        print(f"Input: {data_2['job']}, {data_2['poutcome']}, {data_2['month']}")
        print(f"Output: {result_2}\n")
        
        logger.info("--- TEST 3: ON THE FENCE ---")
        result_3 = pipeline.predict(data_3)
        print(f"Input: {data_3['job']}, {data_3['poutcome']}, {data_3['month']}")
        print(f"Output: {result_3}\n")

    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    run_prediction_test()