import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, Field

# Load ML Pipeline
try:
  model = joblib.load('niper_model.joblib')
except Exception:
  model = None

app = FastAPI(
    title='NIPER Scheme Predictor API',
    description='REST API for predicting financial utilization and risk flags for NIPER grants.',
    version='1.0.0',
)

# Enable CORS for web requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class SchemeProposal(BaseModel):
  campus_name: str = Field(..., example='NIPER SAS Nagar (Mohali)')
  grant_category: str = Field(..., example='Campus Infrastructure & Civil')
  quarter_released: str = Field(..., example='Q1')
  budget_allocated_cr: float = Field(..., gt=0, example=25.0)
  release_delay_days: int = Field(..., ge=0, example=15)
  procurement_completion_pct: float = Field(..., ge=0, le=100, example=80.0)
  prior_year_utilization_pct: float = Field(..., ge=0, le=100, example=90.0)
  active_projects_count: int = Field(..., ge=0, example=12)


@app.get('/')
def health_check():
  return {
      'status': 'online',
      'system': 'NIPER Scheme Predictor API',
      'docs_url': 'http://127.0.0.1:8000/docs',
  }


@app.post('/predict')
def predict_utilization(proposal: SchemeProposal):
  if model is None:
    raise HTTPException(
        status_code=500, detail='Model file niper_model.joblib not found.'
    )

  input_df = pd.DataFrame([proposal.model_dump()])
  predicted_val = float(model.predict(input_df)[0])
  predicted_utilization = round(max(0.0, min(100.0, predicted_val)), 2)

  if predicted_utilization >= 85.0:
    risk_level = 'LOW_RISK'
    recommendation = (
        'Optimal execution predicted. Approved for full allocation.'
    )
  elif predicted_utilization >= 70.0:
    risk_level = 'MEDIUM_RISK'
    recommendation = (
        'Moderate utilization. Monitor procurement milestones closely.'
    )
  else:
    risk_level = 'HIGH_RISK'
    recommendation = (
        'High under-utilization risk. Re-evaluate release timing or procurement'
        ' bandwidth.'
    )

  estimated_expenditure = round(
      (predicted_utilization / 100.0) * proposal.budget_allocated_cr, 2
  )

  return {
      'input_proposal': proposal.model_dump(),
      'predicted_utilization_pct': predicted_utilization,
      'estimated_expenditure_cr': estimated_expenditure,
      'risk_level': risk_level,
      'recommendation': recommendation,
  }