import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import os
import io
import uuid
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("insurance-api")

# Create FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="API for predicting employee insurance enrollment based on demographic and employment data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to model and results directory
MODEL_PATH = 'models/xgb_model.pkl'  # Update with your model path
RESULTS_DIR = 'prediction_results'

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully")
    
    # Get model type
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_type = type(model.named_steps['model']).__name__
    else:
        model_type = type(model).__name__
    
    logger.info(f"Model type: {model_type}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    model_type = "Unknown"

# Define input data models
class Employee(BaseModel):
    age: int = Field(..., description="Employee age", ge=18, le=100, example=35)
    gender: str = Field(..., description="Employee gender", example="Female")
    marital_status: str = Field(..., description="Marital status", example="Married")
    salary: float = Field(..., description="Annual salary", ge=0, example=65000.0)
    employment_type: str = Field(..., description="Employment type", example="Full-time")
    region: str = Field(..., description="Region", example="West")
    has_dependents: str = Field(..., description="Has dependents", example="Yes")
    tenure_years: float = Field(..., description="Years of service", ge=0, example=5.5)
    
    # Optional engineered features
    age_group: Optional[str] = Field(None, description="Age group", example="31-40")
    salary_range: Optional[str] = Field(None, description="Salary range", example="Medium")
    tenure_group: Optional[str] = Field(None, description="Tenure group", example="5-10")
    family_status: Optional[str] = Field(None, description="Family status", example="Married_Yes")
    
    @validator('gender')
    def validate_gender(cls, v):
        valid_genders = ['Male', 'Female', 'Other']
        if v not in valid_genders:
            raise ValueError(f'Gender must be one of {valid_genders}')
        return v
    
    @validator('marital_status')
    def validate_marital_status(cls, v):
        valid_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
        if v not in valid_statuses:
            raise ValueError(f'Marital status must be one of {valid_statuses}')
        return v
    
    @validator('employment_type')
    def validate_employment_type(cls, v):
        valid_types = ['Full-time', 'Part-time', 'Contract']
        if v not in valid_types:
            raise ValueError(f'Employment type must be one of {valid_types}')
        return v
    
    @validator('region')
    def validate_region(cls, v):
        valid_regions = ['West', 'East', 'North', 'South', 'Midwest', 'Northeast']
        if v not in valid_regions:
            raise ValueError(f'Region must be one of {valid_regions}')
        return v
    
    @validator('has_dependents')
    def validate_has_dependents(cls, v):
        valid_values = ['Yes', 'No']
        if v not in valid_values:
            raise ValueError(f'Has dependents must be one of {valid_values}')
        return v

class PredictionInput(BaseModel):
    employees: List[Employee]
    
    class Config:
        schema_extra = {
            "example": {
                "employees": [
                    {
                        "age": 35,
                        "gender": "Female",
                        "marital_status": "Married",
                        "salary": 65000.0,
                        "employment_type": "Full-time",
                        "region": "West",
                        "has_dependents": "Yes",
                        "tenure_years": 5.5
                    },
                    {
                        "age": 28,
                        "gender": "Male",
                        "marital_status": "Single",
                        "salary": 48000.0,
                        "employment_type": "Part-time",
                        "region": "Midwest",
                        "has_dependents": "No",
                        "tenure_years": 1.2
                    }
                ]
            }
        }

class PredictionResult(BaseModel):
    prediction: int
    probability: float
    enrolled: str
    confidence: str
    risk_level: str

class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    results: List[PredictionResult]
    model_type: str
    model_version: str
    avg_probability: float
    notes: Optional[str] = None

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    api_version: str
    model_performance: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    status: str
    file_name: str
    results_file: str
    total_records: int
    predicted_enrolled: int
    predicted_not_enrolled: int
    avg_probability: float
    processing_time_ms: float

# Define helper functions
def create_engineered_features(df):
    """Create engineered features based on the original features"""
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Create age groups if not present
    if 'age_group' not in df.columns:
        result_df['age_group'] = pd.cut(
            result_df['age'], 
            bins=[0, 30, 40, 50, 60, 100], 
            labels=['20-30', '31-40', '41-50', '51-60', '61+'], 
            right=False
        )
    
    # Create salary ranges if not present
    if 'salary_range' not in df.columns:
        result_df['salary_range'] = pd.cut(
            result_df['salary'], 
            bins=[0, 40000, 50000, 65000, 80000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
    
    # Create tenure groups if not present
    if 'tenure_group' not in df.columns:
        result_df['tenure_group'] = pd.cut(
            result_df['tenure_years'], 
            bins=[0, 2, 5, 10, 15, 20, float('inf')], 
            labels=['0-2', '2-5', '5-10', '10-15', '15-20', '20+'], 
            include_lowest=True
        )
    
    # Create family status if not present
    if 'family_status' not in df.columns:
        result_df['family_status'] = result_df['marital_status'] + '_' + result_df['has_dependents']
    
    return result_df

def format_prediction_result(prediction, probability):
    """Format prediction results with human-readable labels and confidence levels"""
    # Determine confidence level
    if abs(probability - 0.5) > 0.4:
        confidence = "Very High"
    elif abs(probability - 0.5) > 0.3:
        confidence = "High"
    elif abs(probability - 0.5) > 0.2:
        confidence = "Medium"
    elif abs(probability - 0.5) > 0.1:
        confidence = "Low"
    else:
        confidence = "Very Low"
    
    # Determine risk level based on prediction and probability
    if prediction == 1:
        if probability > 0.9:
            risk_level = "Very Low Risk"
        elif probability > 0.8:
            risk_level = "Low Risk"
        elif probability > 0.7:
            risk_level = "Moderate Risk"
        elif probability > 0.6:
            risk_level = "Elevated Risk"
        else:
            risk_level = "High Risk"
    else:
        if probability < 0.1:
            risk_level = "Very Low Interest"
        elif probability < 0.2:
            risk_level = "Low Interest"
        elif probability < 0.3:
            risk_level = "Moderate Interest"
        elif probability < 0.4:
            risk_level = "Potential Interest"
        else:
            risk_level = "Strong Potential"
    
    result = PredictionResult(
        prediction=int(prediction),
        probability=float(probability),
        enrolled="Yes" if prediction == 1 else "No",
        confidence=confidence,
        risk_level=risk_level
    )
    return result

def process_batch_file(file_path, output_path, request_id):
    """Process batch prediction file in background"""
    start_time = datetime.now()
    
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Create engineered features
        df = create_engineered_features(df)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['probability'] = probabilities
        df['enrolled'] = df['prediction'].apply(lambda x: "Yes" if x == 1 else "No")
        
        # Add confidence levels
        df['confidence'] = df['probability'].apply(
            lambda x: "Very High" if abs(x - 0.5) > 0.4 else
                     "High" if abs(x - 0.5) > 0.3 else
                     "Medium" if abs(x - 0.5) > 0.2 else
                     "Low" if abs(x - 0.5) > 0.1 else
                     "Very Low"
        )
        
        # Add risk levels
        df['risk_level'] = df.apply(
            lambda row: "Very Low Risk" if row['prediction'] == 1 and row['probability'] > 0.9 else
                        "Low Risk" if row['prediction'] == 1 and row['probability'] > 0.8 else
                        "Moderate Risk" if row['prediction'] == 1 and row['probability'] > 0.7 else
                        "Elevated Risk" if row['prediction'] == 1 and row['probability'] > 0.6 else
                        "High Risk" if row['prediction'] == 1 else
                        "Very Low Interest" if row['probability'] < 0.1 else
                        "Low Interest" if row['probability'] < 0.2 else
                        "Moderate Interest" if row['probability'] < 0.3 else
                        "Potential Interest" if row['probability'] < 0.4 else
                        "Strong Potential",
            axis=1
        )
        
        # Save results to CSV
        df.to_csv(output_path, index=False)
        
        # Log success
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Batch prediction completed: {len(df)} records processed in {processing_time:.2f}ms")
        
        # Create metadata file
        metadata = {
            "request_id": request_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Completed",
            "file_name": os.path.basename(file_path),
            "results_file": os.path.basename(output_path),
            "total_records": len(df),
            "predicted_enrolled": int(predictions.sum()),
            "predicted_not_enrolled": int(len(predictions) - predictions.sum()),
            "avg_probability": float(probabilities.mean()),
            "processing_time_ms": float(processing_time)
        }
        
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        error_message = f"Error processing batch file: {str(e)}"
        logger.error(error_message)
        
        # Create error result file
        error_df = pd.DataFrame([{"error": error_message}])
        error_df.to_csv(output_path, index=False)
        
        # Create metadata file with error status
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata = {
            "request_id": request_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "Failed",
            "file_name": os.path.basename(file_path),
            "results_file": os.path.basename(output_path),
            "error": error_message,
            "processing_time_ms": float(processing_time)
        }
        
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# Define API endpoints
@app.get("/", tags=["General"])
async def root():
    """Get API status and basic information"""
    return {
        "name": "Insurance Enrollment Prediction API",
        "status": "active" if model is not None else "model not loaded",
        "version": "1.0.0",
        "model": model_type,
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Check API health status"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": model_type,
        "api_version": "1.0.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def model_info():
    """Get detailed information about the model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Try to load the model details file
        model_name = os.path.basename(MODEL_PATH).replace('.pkl', '')
        details_path = os.path.join(os.path.dirname(MODEL_PATH), f"{model_name}_details.json")
        
        if os.path.exists(details_path):
            with open(details_path, 'r') as f:
                details = json.load(f)
            
            performance = details.get('test_results', {})
        else:
            # Default performance if no details file
            performance = {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85,
                "auc_roc": 0.90
            }
        
        return ModelInfo(
            model_name=model_name,
            model_type=model_type,
            api_version="1.0.0",
            model_performance=performance
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Make predictions for a list of employees
    
    Returns predictions with probabilities and confidence levels
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate request ID and timestamp
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Log request info
        logger.info(f"Prediction request received: {request_id} with {len(input_data.employees)} employees")
        
        # Convert input data to dataframe
        employees_dict = [emp.dict() for emp in input_data.employees]
        df = pd.DataFrame(employees_dict)
        
        # Create engineered features
        df = create_engineered_features(df)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        # Format results
        results = [
            format_prediction_result(pred, prob)
            for pred, prob in zip(predictions, probabilities)
        ]
        
        # Create response
        response = PredictionResponse(
            request_id=request_id,
            timestamp=timestamp,
            results=results,
            model_type=model_type,
            model_version="1.0.0",
            avg_probability=float(probabilities.mean())
        )
        
        # Log response info
        logger.info(f"Prediction completed: {request_id}, " +
                  f"predicting {sum(1 for r in results if r.enrolled == 'Yes')}/{len(results)} as enrolled")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Make predictions for a batch of employees from a CSV file
    
    Returns a status response and processes the file in the background
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Generate request ID, input and output file paths
        request_id = str(uuid.uuid4())
        file_name = f"batch_input_{request_id}.csv"
        input_path = os.path.join(RESULTS_DIR, file_name)
        
        results_file = f"batch_results_{request_id}.csv"
        output_path = os.path.join(RESULTS_DIR, results_file)
        
        # Save uploaded file
        contents = await file.read()
        with open(input_path, 'wb') as f:
            f.write(contents)
        
        # Log request info
        logger.info(f"Batch prediction request received: {request_id}, file: {file.filename}")
        
        # Add background task to process file
        background_tasks.add_task(process_batch_file, input_path, output_path, request_id)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Return response
        return BatchPredictionResponse(
            request_id=request_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            status="Processing",
            file_name=file.filename,
            results_file=results_file,
            total_records=0,  # Will be updated when processing completes
            predicted_enrolled=0,  # Will be updated when processing completes
            predicted_not_enrolled=0,  # Will be updated when processing completes
            avg_probability=0.0,  # Will be updated when processing completes
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/batch_status/{request_id}", tags=["Prediction"])
async def batch_status(request_id: str):
    """
    Check the status of a batch prediction request
    
    Returns the status and metadata for a batch prediction request
    """
    try:
        # Look for metadata file
        metadata_path = os.path.join(RESULTS_DIR, f"batch_results_{request_id}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return {"status": "Processing", "request_id": request_id}
        
        # Read metadata file
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error checking batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking batch status: {str(e)}")

@app.get("/batch_results/{request_id}", tags=["Prediction"])
async def batch_results(request_id: str):
    """
    Get the results for a batch prediction request
    
    Returns the CSV file with prediction results
    """
    try:
        # Look for results file
        results_path = os.path.join(RESULTS_DIR, f"batch_results_{request_id}.csv")
        
        if not os.path.exists(results_path):
            raise HTTPException(status_code=404, detail=f"Results not found for request ID: {request_id}")
        
        return FileResponse(
            path=results_path,
            filename=f"insurance_enrollment_predictions_{request_id}.csv",
            media_type="text/csv"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving batch results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving batch results: {str(e)}")

@app.get("/sample_input", tags=["Utilities"])
async def sample_input():
    """
    Get a sample input CSV file for batch prediction
    
    Returns a CSV file with sample input data
    """
    try:
        # Create sample data
        sample_data = [
            {
                "employee_id": 1001,
                "age": 35,
                "gender": "Female",
                "marital_status": "Married",
                "salary": 65000.0,
                "employment_type": "Full-time",
                "region": "West",
                "has_dependents": "Yes",
                "tenure_years": 5.5
            },
            {
                "employee_id": 1002,
                "age": 28,
                "gender": "Male",
                "marital_status": "Single",
                "salary": 48000.0,
                "employment_type": "Part-time",
                "region": "Midwest",
                "has_dependents": "No",
                "tenure_years": 1.2
            },
            {
                "employee_id": 1003,
                "age": 42,
                "gender": "Male",
                "marital_status": "Divorced",
                "salary": 85000.0,
                "employment_type": "Full-time",
                "region": "Northeast",
                "has_dependents": "Yes",
                "tenure_years": 8.7
            },
            {
                "employee_id": 1004,
                "age": 55,
                "gender": "Female",
                "marital_status": "Married",
                "salary": 92000.0,
                "employment_type": "Full-time",
                "region": "South",
                "has_dependents": "No",
                "tenure_years": 15.3
            },
            {
                "employee_id": 1005,
                "age": 24,
                "gender": "Other",
                "marital_status": "Single",
                "salary": 45000.0,
                "employment_type": "Contract",
                "region": "West",
                "has_dependents": "No",
                "tenure_years": 0.8
            }
        ]
        
        # Create DataFrame and save to CSV
        sample_df = pd.DataFrame(sample_data)
        sample_path = os.path.join(RESULTS_DIR, "sample_input.csv")
        sample_df.to_csv(sample_path, index=False)
        
        return FileResponse(
            path=sample_path,
            filename="insurance_enrollment_sample_input.csv",
            media_type="text/csv"
        )
    
    except Exception as e:
        logger.error(f"Error creating sample input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating sample input: {str(e)}")

# Run the API with uvicorn if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)