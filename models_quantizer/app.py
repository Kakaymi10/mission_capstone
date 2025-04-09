import json
import re
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import numpy as np
import cv2
import os
import uuid
import io
from datetime import datetime
import onnxruntime
from PIL import Image
import base64
import ollama
from ultralytics import FastSAM

app = FastAPI(title="Computer Vision API", 
              description="API for image segmentation using FastSAM, super resolution using RealESRGAN, and image description using Ollama LLaVA")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing results
TEMP_DIR = "temp"
RESULTS_DIR = "upscaled_results"
IMAGE_DIR = "images"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Initialize models
# FastSAM for segmentation
fastsam_model = FastSAM("FastSAM-s.pt")

# Initialize ONNX runtime for RealESRGAN
try:
    onnx_session = onnxruntime.InferenceSession('realesrgan_x4_quantized.onnx')
except Exception as e:
    print(f"Error loading RealESRGAN model: {e}")
    onnx_session = None

# LLaVA model configuration
LLAVA_MODEL = "llava:7b"

# RealESRGAN helper functions
def preprocess_image(image: Image.Image, input_size: int = 256) -> np.ndarray:
    """Preprocess image for RealESRGAN model input"""
    # Resize to a larger size for better quality
    image = image.resize((input_size, input_size))
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    # Transpose to CHW format
    img_array = img_array.transpose(2, 0, 1)
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    return img_array

def postprocess_output(output: np.ndarray) -> Image.Image:
    """Convert model output to image"""
    # Remove batch dimension and transpose back to HWC
    output = output[0].transpose(1, 2, 0)
    # Clip values and convert to uint8
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(output)

# API Endpoints

@app.post("/segment/")
async def segment_image(
    image: UploadFile = File(...),
    bboxes: str = Form(...),  # Format: "x1,y1,x2,y2;x1,y1,x2,y2" for multiple boxes
):
    """
    Segment an image using FastSAM with bounding box inputs.
    
    Args:
        image: Input image file
        bboxes: Comma-separated bounding box coordinates in format "x1,y1,x2,y2;x1,y1,x2,y2" for multiple boxes
        
    Returns:
        The segmented image
    """
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
        results = fastsam_model(pil_img, bboxes=bbox_list, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # Generate output filename
        output_filename = f"{TEMP_DIR}/{uuid.uuid4()}.png"
        
        # Save visualization with masks
        result_img = results[0].plot()
        cv2.imwrite(output_filename, result_img)
        
        return FileResponse(output_filename, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    input_size: Optional[int] = Form(256)
):
    """
    Upscale an image using the quantized RealESRGAN model.
    
    Args:
        file: The input image file
        input_size: The size to resize the input image to (default: 256)
    
    Returns:
        The upscaled image
    """
    if onnx_session is None:
        raise HTTPException(status_code=500, detail="RealESRGAN model not initialized")
        
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess the image
        input_data = preprocess_image(image, input_size)
        
        # Run inference
        output = onnx_session.run(['output'], {'input': input_data})[0]
        
        # Postprocess the output
        sr_image = postprocess_output(output)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upscaled_{timestamp}.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Save the image
        sr_image.save(filepath, format='PNG')
        
        # Convert to bytes for response
        img_byte_arr = io.BytesIO()
        sr_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return the image data directly
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error upscaling image: {str(e)}")

@app.post("/describe/")
async def describe_image(
    image: UploadFile = File(...),
    annotations: Optional[str] = Form(None)  # Text-based annotations describing the image content
):
    """
    Describe an image using Ollama LLaVA model, with optional text annotations.
    
    Args:
        image: Input image file
        annotations: Optional text describing annotations, objects, or regions in the image
        
    Returns:
        A JSON response with the description generated by LLaVA
    """
    try:
        # Read image
        contents = await image.read()
        
        # Generate unique filename for the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"image_{timestamp}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        # Save the original image without modifications
        with open(image_path, "wb") as img_file:
            img_file.write(contents)
        
        # Create a professional prompt focused on microscopic imagery with confident tone
        if annotations:
            prompt = f"""This is a microscopic image with the following annotations: {annotations}.

Provide a clear, confident analysis of this microscopic sample using precise scientific terminology.
Identify the key cellular or tissue structures visible, their characteristics, and their significance.
State definitively what type of sample this is and describe any notable features or abnormalities.
Include specific contextual information about the biological significance.
Use direct, affirmative statements throughout your explanation without hedging language like "appears to be" or "seems like"."""
        else:
            prompt = """This is a microscopic image.

Provide a clear, confident analysis of this microscopic sample using precise scientific terminology.
Identify the key cellular or tissue structures visible, their characteristics, and their significance.
State definitively what type of sample this is and describe any notable features or abnormalities.
Include specific contextual information about the biological significance.
Use direct, affirmative statements throughout your explanation without hedging language like "appears to be" or "seems like"."""
        
        # Send image to Ollama LLaVA
        res = ollama.chat(
            model=LLAVA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }
            ]
        )
        
        # Extract response
        description = res['message']['content']
        
        return JSONResponse(
            content={
                "description": description
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error describing image: {str(e)}")

# Add this new endpoint to your existing FastAPI application
@app.post("/chat_region/")
async def chat_about_image_region(
    image: UploadFile = File(...),
    question: str = Form(...),
    bbox: str = Form(...),  # Required bounding box coordinates (x1,y1,x2,y2)
    label: Optional[str] = Form(None)  # Optional label for the region
):
    """
    Chat with the LLaVA model about a specific region in the image defined by a bounding box.
    
    Args:
        image: Input image file (typically a microscopic image)
        question: User's question about the specific region
        bbox: Bounding box coordinates in format "x1,y1,x2,y2"
        label: Optional label for the region
        
    Returns:
        A JSON response with focused answer about the specific region
    """
    try:
        # Read image
        contents = await image.read()
        
        # Generate unique filename for the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"region_chat_{timestamp}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        # Parse bbox coordinates
        try:
            coords = [int(float(coord)) for coord in bbox.split(',')]
            if len(coords) != 4:
                raise ValueError("Bounding box must have 4 coordinates (x1,y1,x2,y2)")
            
            # Read the image for drawing
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Draw rectangle
            x1, y1, x2, y2 = coords
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label if provided
            if label:
                display_label = label[:20] + "..." if len(label) > 20 else label
                cv2.putText(img, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save the modified image
            cv2.imwrite(image_path, img)
        except Exception as e:
            print(f"Error drawing bounding box: {e}")
            # Save the original image without modifications if error in drawing
            with open(image_path, "wb") as img_file:
                img_file.write(contents)
        
        # Create a focused prompt that guides the model to address only the specified region
        system_guidance = """You are analyzing a specific highlighted region in a microscopic image (marked with a green box). Provide a scientifically accurate, educational response about this specific region.

Your response should:
- Focus ONLY on the specific region marked by the green box
- Be declarative and confident in tone
- Use precise scientific terminology to identify and explain the structure
- Avoid hedging language like "appears to be" or "seems like"
- Provide educational context about this specific structure
- NOT describe other parts of the image outside the highlighted region
- Not use However with limited knowledge, it shold give a direct answer
"""

        # Create the prompt with region focus
        prompt = f"""{system_guidance}

QUESTION ABOUT THE HIGHLIGHTED REGION: {question}

Remember to only discuss the specific region highlighted by the green box."""
        
        # Send image and prompt to Ollama LLaVA
        res = ollama.chat(
            model=LLAVA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }
            ]
        )
        
        # Extract response
        answer = res['message']['content']
        
        return JSONResponse(
            content={
                "answer": answer,
                "bbox": bbox
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing region chat: {str(e)}")
    

@app.post("/segment_everything/")
async def segment_everything(
    image: UploadFile = File(...),
    conf: float = Form(0.4),
    iou: float = Form(0.9),
    output_format: str = Form("image")  # Options: "image", "json"
):
    """
    Segment everything in an image using FastSAM.
    
    Args:
        image: Input image file
        conf: Confidence threshold (default: 0.4)
        iou: IoU threshold (default: 0.9)
        output_format: Response format - "image" for visualization or "json" for mask data
        
    Returns:
        The segmented image visualization or mask data in JSON format
    """
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (FastSAM expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.IMREAD_COLOR)
        
        # Create a PIL Image for FastSAM
        pil_img = Image.fromarray(img_rgb)
        
        # Process with FastSAM in everything mode
        results = fastsam_model(pil_img, device="cpu", retina_masks=True, imgsz=1024, conf=conf, iou=iou)
        
        # Generate output filename
        output_filename = f"{TEMP_DIR}/{uuid.uuid4()}.png"
        
        if output_format.lower() == "json":
            # Return JSON with mask data
            masks_data = []
            for i, mask in enumerate(results[0].masks.data):
                mask_np = mask.cpu().numpy()
                # Convert binary mask to base64 string for transmission
                mask_bytes = io.BytesIO()
                np.save(mask_bytes, mask_np)
                mask_bytes.seek(0)
                mask_base64 = base64.b64encode(mask_bytes.read()).decode('utf-8')
                
                # Add mask data to response
                masks_data.append({
                    "id": i,
                    "mask": mask_base64,
                    "box": results[0].boxes.data[i].cpu().numpy().tolist() if i < len(results[0].boxes) else None
                })
            
            return JSONResponse(content={"masks": masks_data})
        else:
            # Save visualization with masks
            result_img = results[0].plot()
            cv2.imwrite(output_filename, result_img)
            
            return FileResponse(output_filename, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/segment_click/")
async def segment_click(
    image: UploadFile = File(...),
    point_x: float = Form(...),  # x-coordinate of click
    point_y: float = Form(...),  # y-coordinate of click
    point_type: int = Form(1),   # 1 for foreground, 0 for background
    conf: float = Form(0.4),
    iou: float = Form(0.9)
):
    """
    Segment an image based on click point using FastSAM.
    
    Args:
        image: Input image file
        point_x: X-coordinate of the clicked point
        point_y: Y-coordinate of the clicked point
        point_type: 1 for foreground point, 0 for background point
        conf: Confidence threshold (default: 0.4)
        iou: IoU threshold (default: 0.9)
        
    Returns:
        The segmented image with mask around the clicked point
    """
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (FastSAM expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.IMREAD_COLOR)
        
        # Create a PIL Image for FastSAM
        pil_img = Image.fromarray(img_rgb)
        
        # First get all masks
        results = fastsam_model(pil_img, device="cpu", retina_masks=True, imgsz=1024, conf=conf, iou=iou)
        
        # Format the point for FastSAM point prompt
        # FastSAM expects points as [[x, y, point_type]]
        # where point_type is 1 for foreground, 0 for background
        point_prompt = [[point_x, point_y, point_type]]
        
        # For click-based segmentation, we run everything mode first, 
        # then use point prompts on the results
        # Get the original masks and filter them with point prompt
        everything_masks = results[0].masks.data
        
        if everything_masks is None or len(everything_masks) == 0:
            raise HTTPException(status_code=400, detail="No masks found in the image")
        
        # Use FastSAM's point prompt functionality
        ann = results[0].masks.data.cpu().numpy()
        
        # Find the mask that contains the clicked point
        selected_mask = None
        selected_index = -1
        
        # Convert point to integer coordinates for mask indexing
        point_x_int, point_y_int = int(point_x), int(point_y)
        
        # Check if the point is within image boundaries
        h, w = ann[0].shape if len(ann) > 0 else (0, 0)
        if not (0 <= point_x_int < w and 0 <= point_y_int < h):
            raise HTTPException(status_code=400, detail="Click point is outside image boundaries")
        
        for i, mask in enumerate(ann):
            if mask[point_y_int, point_x_int] > 0:  # If point is inside this mask
                if point_type == 1:  # Foreground point
                    selected_mask = mask
                    selected_index = i
                    break
                # For background points, we'd typically use a different logic
        
        if selected_mask is None and point_type == 1:
            # If no mask contains the point but it's a foreground point,
            # find the closest mask or use proximity-based selection
            
            # Simple approach: find the nearest mask (distance to mask boundary)
            min_distance = float('inf')
            for i, mask in enumerate(ann):
                # Find distance transform of mask
                # This gives distance to the nearest non-zero pixel
                dist = cv2.distanceTransform((~mask.astype(np.uint8)) * 255, cv2.DIST_L2, 3)
                distance_at_point = dist[point_y_int, point_x_int]
                
                if distance_at_point < min_distance:
                    min_distance = distance_at_point
                    selected_mask = mask
                    selected_index = i
        
        if selected_mask is None:
            raise HTTPException(status_code=400, detail="No suitable mask found for the clicked point")
        
        # Create output visualization
        output_img = img.copy()
        
        # Apply the selected mask
        mask_vis = np.zeros_like(output_img)
        mask_vis[:,:,1] = selected_mask.astype(np.uint8) * 255  # Green channel
        
        # Blend with original image
        alpha = 0.5
        output_img = cv2.addWeighted(output_img, 1, mask_vis, alpha, 0)
        
        # Draw the click point
        cv2.drawMarker(output_img, 
                      (int(point_x), int(point_y)), 
                      (0, 0, 255) if point_type == 1 else (255, 0, 0),  # Red for foreground, Blue for background
                      markerType=cv2.MARKER_CROSS, 
                      markerSize=20, 
                      thickness=2)
        
        # Generate output filename
        output_filename = f"{TEMP_DIR}/{uuid.uuid4()}.png"
        cv2.imwrite(output_filename, output_img)
        
        # Also prepare mask data for response
        mask_bytes = io.BytesIO()
        np.save(mask_bytes, selected_mask)
        mask_bytes.seek(0)
        mask_base64 = base64.b64encode(mask_bytes.read()).decode('utf-8')
        
        # Create JSON response with both image path and mask data
        result = {
            "image_path": output_filename,
            "mask_data": {
                "mask": mask_base64,
                "index": selected_index
            }
        }
        
        # Return the image file
        return FileResponse(output_filename, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing click-based segmentation: {str(e)}")
    
@app.post("/generate_arcane_quiz/")
async def generate_arcane_quiz(
    image: UploadFile = File(...),
    specimen_name: Optional[str] = Form(None),
    difficulty: Optional[str] = Form("medium"),  # easy, medium, hard
    quiz_count: Optional[int] = Form(5),  # Number of questions to generate
    segmentation_data: Optional[str] = Form(None)  # Optional JSON string containing segmentation masks
):
    """
    Generate an interactive quiz based on a specimen image with optional segmentation data
    
    Args:
        image: Input specimen image
        specimen_name: Name or label of the specimen (optional)
        difficulty: Quiz difficulty level (easy, medium, hard)
        quiz_count: Number of questions to generate
        segmentation_data: Optional JSON string with segmentation masks information
        
    Returns:
        A JSON response with the generated quiz questions and answers
    """
    try:
        # Read image
        contents = await image.read()
        
        # Generate unique filename for the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"quiz_specimen_{timestamp}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        # Save the image
        with open(image_path, "wb") as img_file:
            img_file.write(contents)
        
        # Parse segmentation data if provided
        segmentation_regions = []
        if segmentation_data:
            try:
                seg_data = json.loads(segmentation_data)
                segmentation_regions = seg_data.get("regions", [])
            except json.JSONDecodeError:
                pass
        
        # Create different prompt templates based on difficulty
        base_prompt = f"""Generate an educational and engaging quiz about this microscopic specimen{' of ' + specimen_name if specimen_name else ''}.

The quiz should:
- Include {quiz_count} multiple-choice questions with 4 options each
- Have scientifically accurate questions and answers
- Be formatted as a structured JSON with questions, options, correct answers, and explanations
- Include questions about visual features visible in the image
"""

        # Add difficulty-specific instructions
        if difficulty == "easy":
            base_prompt += """
- Focus on basic identification and fundamental concepts
- Use simpler terminology and more obvious visual features
- Provide more descriptive explanations for correct answers
"""
        elif difficulty == "hard":
            base_prompt += """
- Include advanced technical terminology and concepts
- Ask about subtle visual details and relationships
- Require higher-level understanding of the specimen's characteristics
- Include comparative questions about similar structures
"""
        
        # Add segmentation-specific questions if segmentation data is provided
        if segmentation_regions:
            base_prompt += """
- Include at least 2 questions that refer to specific regions visible in the image
- For region-specific questions, specify the coordinates or identifiers so they can be highlighted
"""

        # Add specific output format requirements
        base_prompt += """
The response must be valid JSON in the following format:
{
  "quiz_title": "Title related to the specimen",
  "questions": [
    {
      "id": 1,
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_index": 0, // Zero-based index of correct answer
      "explanation": "Explanation of the correct answer",
      "region_id": null, // Include region identifier if question relates to a specific region
      "type": "basic" // basic, visual, advanced
    },
    ...
  ]
}
"""
        
        # Send image to Ollama LLaVA for quiz generation
        res = ollama.chat(
            model=LLAVA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': base_prompt,
                    'images': [image_path]
                }
            ]
        )
        
        # Extract response and parse JSON
        try:
            quiz_text = res['message']['content']
            
            # Extract JSON content from the response
            # First, look for JSON between triple backticks
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", quiz_text)
            if json_match:
                quiz_json = json_match.group(1)
            else:
                # Look for JSON between single backticks
                json_match = re.search(r"`([\s\S]*?)`", quiz_text)
                if json_match:
                    quiz_json = json_match.group(1)
                else:
                    # Try to extract the entire text as JSON
                    quiz_json = quiz_text
            
            # Clean up any narrative text before or after the JSON
            quiz_json = re.sub(r"^.*?(\{)", r"\1", quiz_json, flags=re.DOTALL)
            quiz_json = re.sub(r"(\}).*?$", r"\1", quiz_json, flags=re.DOTALL)
            
            # Parse the JSON
            quiz_data = json.loads(quiz_json)
            
            # Add metadata
            quiz_data["metadata"] = {
                "specimen_name": specimen_name,
                "difficulty": difficulty,
                "generated_at": datetime.now().isoformat(),
                "image_path": image_filename
            }
            
            return JSONResponse(content=quiz_data)
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return the raw text and error
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Failed to parse quiz JSON",
                    "message": str(e),
                    "raw_response": quiz_text
                }
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")


@app.get("/quizzes/{quiz_id}")
async def get_quiz(quiz_id: str):
    """
    Retrieve a previously generated quiz by ID
    
    Args:
        quiz_id: The ID of the quiz to retrieve
        
    Returns:
        The quiz data if found
    """
    quiz_path = os.path.join("quizzes", f"{quiz_id}.json")
    
    if not os.path.exists(quiz_path):
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    try:
        with open(quiz_path, "r") as f:
            quiz_data = json.load(f)
        return quiz_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving quiz: {str(e)}")


@app.post("/save_quiz_result/")
async def save_quiz_result(
    quiz_id: str = Form(...),
    user_id: Optional[str] = Form(None),
    score: int = Form(...),
    answers: str = Form(...)  # JSON string of user answers
):
    """
    Save a user's quiz results
    
    Args:
        quiz_id: ID of the completed quiz
        user_id: Optional user identifier
        score: User's score
        answers: JSON string of user's answers
        
    Returns:
        Confirmation of saved results
    """
    try:
        # Parse user answers
        user_answers = json.loads(answers)
        
        # Create result data
        result_data = {
            "quiz_id": quiz_id,
            "user_id": user_id,
            "score": score,
            "answers": user_answers,
            "completed_at": datetime.now().isoformat()
        }
        
        # Generate result ID
        result_id = f"{quiz_id}_{uuid.uuid4().hex[:8]}"
        
        # Ensure directory exists
        os.makedirs("quiz_results", exist_ok=True)
        
        # Save result
        result_path = os.path.join("quiz_results", f"{result_id}.json")
        with open(result_path, "w") as f:
            json.dump(result_data, f)
        
        return {"result_id": result_id, "status": "saved"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving quiz result: {str(e)}")
    
    
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models": {
            "fastsam": "FastSAM-s",
            "realesrgan": "RealESRGAN-x4" if onnx_session is not None else "Not loaded",
            "llava": LLAVA_MODEL
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Computer Vision API",
        "endpoints": {
            "/segment/": "POST - Segment an image using FastSAM with bounding boxes",
            "/upscale/": "POST - Upscale an image using RealESRGAN",
            "/describe/": "POST - Describe an image using LLaVA with optional annotations",
            "/chat_region/": "POST - Chat about a specific region or structure in a microscopic image",
            "/health/": "GET - Check API health status"
        },
        "results_directories": {
            "segmentation": TEMP_DIR,
            "upscaled": RESULTS_DIR,
            "images": IMAGE_DIR
        }
    }
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)