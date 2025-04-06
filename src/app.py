from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

app = FastAPI(title="Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("uploaded/company_data", exist_ok=True)
os.makedirs("uploaded/rfp", exist_ok=True)

@app.post("/upload")
async def upload(
    company_data: UploadFile = File(...),
    rfp: UploadFile = File(...)
):
    # Save company data file
    company_data_path = os.path.join("uploaded/company_data", company_data.filename)
    with open(company_data_path, "wb") as buffer:
        shutil.copyfileobj(company_data.file, buffer)
    
    # Save RFP file
    rfp_path = os.path.join("uploaded/rfp", rfp.filename)
    with open(rfp_path, "wb") as buffer:
        shutil.copyfileobj(rfp.file, buffer)

    

    return {
        "output" : {
            "company_data_path": company_data_path,
            "rfp_path": rfp_path,
            "status": "Files uploaded successfully"
        }
    }