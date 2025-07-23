from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import io
import logging
import requests
from datetime import datetime, timedelta
import random
from typing import Optional
import base64
from RAG.pipeline import load_documents, chunk_documents, get_vectorstore, get_llm, retrieve_answer
from RAG.utils import get_prompt_template

app = FastAPI()

# Allow frontend to connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NASA API configuration
NASA_API_KEY = "DEMO_KEY"  # Replace with your NASA API key
NASA_BASE_URL = "https://api.nasa.gov/planetary/earth/imagery"

# Load models and RAG components once
try:
    logger.info("Loading BLIP-2 model and processor...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model loaded successfully on {device}.")
    
    # Load RAG components
    logger.info("Loading RAG components...")
    documents = load_documents()
    chunks = chunk_documents(documents)
    vector_store = get_vectorstore(chunks)
    llm = get_llm()
    prompt_template = get_prompt_template()
    logger.info("RAG components loaded successfully.")
    
except Exception as e:
    logger.error(f"Failed to load components: {e}")
    raise e

def fetch_nasa_image(lat: float = None, lon: float = None, date: str = None) -> Optional[bytes]:
    """Fetch satellite image from NASA API"""
    try:
        # Default coastal coordinates if not provided
        if lat is None or lon is None:
            # Some interesting coastal locations for phytoplankton
            coastal_locations = [
                (37.7749, -122.4194),  # San Francisco Bay
                (25.7617, -80.1918),   # Miami coast
                (40.7128, -74.0060),   # New York coast
                (34.0522, -118.2437),  # Los Angeles coast
                (47.6062, -122.3321),  # Seattle coast
            ]
            lat, lon = random.choice(coastal_locations)
        
        if date is None:
            # Get image from last 30 days
            date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
        
        params = {
            "lon": lon,
            "lat": lat,
            "date": date,
            "dim": 0.15,  # Image dimension
            "api_key": NASA_API_KEY
        }
        
        response = requests.get(NASA_BASE_URL, params=params, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"Successfully fetched NASA image for lat={lat}, lon={lon}, date={date}")
            return response.content
        else:
            logger.warning(f"NASA API returned status {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch NASA image: {e}")
        return None

def analyze_image_with_blip(image_bytes: bytes, question: str) -> str:
    """Analyze image using BLIP-2 model"""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = processor(pil_image, question, return_tensors="pt").to(
            device, torch.float16 if device == "cuda" else torch.float32
        )
        output = model.generate(**inputs, max_length=100)
        answer = processor.decode(output[0], skip_special_tokens=True)
        
        return answer
    except Exception as e:
        logger.error(f"BLIP-2 analysis failed: {e}")
        return "Unable to analyze the image."

def get_rag_answer(question: str) -> tuple[str, str]:
    """Get answer from RAG system"""
    try:
        answer, context = retrieve_answer(vector_store, llm, prompt_template, question)
        return answer, context
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return "Unable to retrieve information from knowledge base.", ""

@app.post("/analyze")
async def analyze_image(
    image: Optional[UploadFile] = File(None), 
    question: str = Form(...),
    use_nasa: bool = Form(False),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    date: Optional[str] = Form(None)
):
    logger.info(f"Received question: '{question}' with image: {image.filename if image else 'None'}")
    
    image_bytes = None
    image_analysis = ""
    nasa_used = False
    
    # Handle image input
    if image:
        try:
            image_bytes = await image.read()
            # Validate image
            Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info("User uploaded image validated successfully")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    elif use_nasa or "satellite" in question.lower() or "nasa" in question.lower():
        # Automatically fetch NASA image
        logger.info("Fetching NASA satellite image...")
        image_bytes = fetch_nasa_image(lat, lon, date)
        nasa_used = True
        
        if not image_bytes:
            return {
                "answer": "Could not fetch satellite image. Please try uploading your own image or try again later.",
                "rag_answer": "",
                "image_analysis": "",
                "nasa_used": False,
                "error": "NASA image fetch failed"
            }
    
    # Analyze image if available
    if image_bytes:
        # Create image-specific question for BLIP-2
        image_question = f"What can you see in this coastal satellite image related to: {question}"
        image_analysis = analyze_image_with_blip(image_bytes, image_question)
        logger.info(f"Image analysis: {image_analysis}")
    
    # Get RAG-based answer
    rag_answer, context = get_rag_answer(question)
    
    # Combine answers intelligently
    if image_analysis and rag_answer:
        combined_question = f"""
        Based on the image analysis: "{image_analysis}"
        And the research knowledge: "{rag_answer}"
        
        Provide a comprehensive answer to: {question}
        """
        final_answer, _ = get_rag_answer(combined_question)
    elif image_analysis:
        final_answer = f"Based on the image analysis: {image_analysis}"
    elif rag_answer:
        final_answer = rag_answer
    else:
        final_answer = "I couldn't analyze the image or find relevant information. Please try rephrasing your question."
    
    return {
        "answer": final_answer,
        "rag_answer": rag_answer,
        "image_analysis": image_analysis,
        "nasa_used": nasa_used,
        "context": context[:500] + "..." if len(context) > 500 else context
    }

@app.get("/fetch-nasa-image")
async def fetch_nasa_image_endpoint(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    date: Optional[str] = None
):
    """Endpoint to just fetch a NASA image"""
    image_bytes = fetch_nasa_image(lat, lon, date)
    
    if image_bytes:
        # Convert to base64 for frontend display
        image_b64 = base64.b64encode(image_bytes).decode()
        return {
            "success": True,
            "image": f"data:image/jpeg;base64,{image_b64}",
            "message": "NASA image fetched successfully"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch NASA image")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)