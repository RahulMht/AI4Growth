from fastapi import FastAPI, UploadFile,File
from fastapi.responses import JSONResponse
import pandas as pd
from roboflow import Roboflow
import io
from PIL import Image
import tempfile

app = FastAPI()

rf = Roboflow(api_key="oVWDq3G9R53AAGqZLHuR")
project = rf.workspace().project("ai4growth-coding-challange")
model = project.version(7).model
url_map = pd.read_csv(r'url_map.csv')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    
    # Save the image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(temp_file, 'JPEG')
    
    data = model.predict(temp_file.name, confidence=70, overlap=30).json()

    if data['predictions']:
        for prediction in data['predictions']:
            # Check if Class ID exists in the CSV file
            matched_rows = url_map[url_map['Class ID'] == prediction['class_id']]
            if not matched_rows.empty:
                return JSONResponse(content={"URL": matched_rows['URL'].values[0], "Location": matched_rows['Locations'].values[0]})
                
        return JSONResponse(content={"error": "No matching URL found."})
    else:
        return JSONResponse(content={"error": "No predictions available."})
                
        