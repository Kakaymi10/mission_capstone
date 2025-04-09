from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from typing import List, Optional
import uvicorn
import numpy as np
import cv2
import os
import uuid
from ultralytics import FastSAM
from PIL import Image
import io

app = FastAPI(title="FastSAM Image Segmentation API")

# Initialize FastSAM model
model = FastSAM("FastSAM-s.pt")  # You can use "FastSAM-x.pt" for better performance

# Create a temporary directory to store processed images
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/segment/")
async def segment_image(
    image: UploadFile = File(...),
    bboxes: str = Form(...),  # Format: "x1,y1,x2,y2;x1,y1,x2,y2" for multiple boxes
):
    try:
        # Parse bounding boxes
        bbox_list = []
        for bbox_str in bboxes.split(';'):
            if bbox_str:
                coords = [float(coord) for coord in bbox_str.split(',')]
                if len(coords) == 4:
                    bbox_list.append(coords)
        
        if not bbox_list:
            raise HTTPException(status_code=400, detail="No valid bounding boxes provided")
        
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (FastSAM expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.IMREAD_COLOR)
        
        # Create a PIL Image for FastSAM
        pil_img = Image.fromarray(img_rgb)
        
        # Process with FastSAM
        results = model(pil_img, bboxes=bbox_list, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # Generate output filename
        output_filename = f"{TEMP_DIR}/{uuid.uuid4()}.png"
        
        # Save visualization with masks
        result_img = results[0].plot()
        cv2.imwrite(output_filename, result_img)
        
        return FileResponse(output_filename, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "FastSAM-s"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)