from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import base64
import io
import os
from typing import List, Tuple

app = FastAPI(title="Image Text Translator API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Nuxt dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR and translator
#ocr_reader = easyocr.Reader(['en', 'ch_sim', 'ch_tra'])
ocr_reader = easyocr.Reader(['en','ch_sim'])
translator = Translator()

class ImageTextProcessor:
    def __init__(self):
        self.ocr_reader = ocr_reader
        self.translator = translator
    
    def extract_text_with_coordinates(self, image_path: str) -> List[Tuple]:
        """Extract text with bounding box coordinates"""
        results = self.ocr_reader.readtext(image_path)
        return results
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using Google Translate"""
        try:
            # Map language codes
            lang_map = {'en': 'en', 'zh': 'zh-cn'}
            src = lang_map.get(source_lang, 'en')
            dest = lang_map.get(target_lang, 'zh-cn')
            print(f"Translating from {src} to {dest}")
            if src == dest:
                return text
                
            result = self.translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def calculate_font_size(self, original_text: str, translated_text: str, bbox_width: int, bbox_height: int,scale_factor: float) -> int:
        """Calculate appropriate font size based on original text characteristics and bounding box"""
        # Estimate original font size based on bounding box height
        # Typically, font height is about 70-80% of the bounding box height
        estimated_original_font_size = int(bbox_height * 0.75)
        
        # Consider text length ratio for width fitting
        original_len = len(original_text) if original_text else 1
        translated_len = len(translated_text) if translated_text else 1
        
        # Adjust font size based on text length difference
        # If translated text is longer, reduce font size proportionally
        length_ratio = original_len / translated_len
        width_based_size = int((bbox_width / translated_len) * 1.2) if translated_len > 0 else estimated_original_font_size
        
        # Use the smaller of height-based and width-based estimates
        #estimated_font_size = min(estimated_original_font_size, width_based_size)
        estimated_font_size =  estimated_original_font_size
        
        # Apply length ratio adjustment
        if length_ratio < 1:  # Translated text is longer
            estimated_font_size = int(estimated_font_size * max(0.6, length_ratio))
        
        # Ensure reasonable bounds
        return max(8, min(estimated_font_size, 72))
    
    def get_dominant_text_color(self, image_array, bbox):
        """Extract the dominant text color from the bounding box region"""
        try:
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            # Extract the text region
            text_region = image_array[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # Convert to grayscale to find text pixels
            gray_region = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu's thresholding to separate text from background
            _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find text pixels (assuming text is darker than background)
            text_mask = binary < 128
            
            if np.any(text_mask):
                # Get the average color of text pixels
                text_pixels = text_region[text_mask]
                avg_color = np.mean(text_pixels, axis=0)
                return tuple(map(int, avg_color))
            else:
                # Fallback to black if no text pixels found
                return (0, 0, 0)
                
        except Exception as e:
            print(f"Error extracting text color: {e}")
            return (0, 0, 0)  # Default to black

    def replace_text_in_image(self, image_path: str, source_lang: str, target_lang: str, font_path: str = None, scale_factor: float = 1.0) -> str:
        """Replace text in image with translated text"""
        try:
            print(f"Loading image: {image_path}")
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            print("Extracting text with OCR...")
            # Extract text with coordinates
            results = self.extract_text_with_coordinates(image_path)
            print(f"Found {len(results)} text regions")
            
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"Processing text {i+1}: '{text}' (confidence: {confidence:.2f})")
                
                if confidence > 0.5:  # Only process high-confidence detections
                    # Get bounding box coordinates
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    print(f"size: 'top left {top_left}, bottom_right {bottom_right}'")
                    # Calculate dimensions
                    width = bottom_right[0] - top_left[0]
                    height = bottom_right[1] - top_left[1]
                    print(f"size: 'width {width}, height {height}'")
                    
                    # Extract original text color
                    original_color = self.get_dominant_text_color(image_rgb, bbox)
                    print(f"Detected original text color: {original_color}")
                    
                    print(f"Translating text: '{text}'")
                    # Translate text
                    translated_text = self.translate_text(text, source_lang, target_lang)
                    print(f"Translated to: '{translated_text}'")
                    
                    # Calculate font size and apply scale factor
                    base_font_size = self.calculate_font_size(text, translated_text, width, height,scale_factor)
                    font_size = int(base_font_size )
                    print(f"Base font size: {base_font_size}, Scale factor: {scale_factor}, Final font size: {font_size}")
                    try:
                        font = None
                        
                        # If font_path is provided and target language is Chinese, use it
                        if font_path and target_lang == 'zh-cn' and os.path.exists(font_path):
                            font = ImageFont.truetype(font_path, font_size)
                            print(f"Using specified Chinese font: {font_path}")
                        else:
                            # Try to use system fonts that support multiple languages
                            font_paths = [
                                "/usr/share/fonts/truetype/arphic/ukai.ttc",
                                "/usr/share/fonts/truetype/arphic/uming.ttc",
                                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Fallback Chinese font
                                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                            ]
                            
                            for fallback_font_path in font_paths:
                                if os.path.exists(fallback_font_path):
                                    font = ImageFont.truetype(fallback_font_path, font_size)
                                    print(f"Using fallback font: {fallback_font_path}")
                                    break
                        
                        if font is None:
                            font = ImageFont.load_default()
                            print("Using default font")
                            
                    except Exception as font_error:
                        print(f"Font loading error: {font_error}")
                        font = ImageFont.load_default()
                    
                    # REMOVED: Create a white rectangle to cover original text
                    # draw.rectangle([top_left, bottom_right], fill='white', outline='white')
                    
                    # Draw translated text with original color (transparent background)
                    # Center the text in the bounding box
                    text_bbox = draw.textbbox((0, 0), translated_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = top_left[0] + (width - text_width) // 2
                    text_y = top_left[1] + (height - text_height) // 2
                    
                    # Use the detected original text color instead of black
                    draw.text((text_x, text_y), translated_text, fill=original_color, font=font)
                    print(f"Drew text at ({text_x}, {text_y}) with color {original_color}")
            
            print("Converting image to base64...")
            # Convert back to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            print(f"Image processing completed, base64 length: {len(img_str)}")
            return img_str
            
        except Exception as e:
            print(f"Error in replace_text_in_image: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            raise

processor = ImageTextProcessor()

@app.post("/api/translate-image")
async def translate_image(
    image: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    font_path: str = Form(None),  # Optional font path parameter
    scale_factor: float = Form(1.0)  # Add scale factor parameter
):
    temp_path = None
    try:
        print(f"Received request: source_lang={source_lang}, target_lang={target_lang}, font_path={font_path}, filename={image.filename}")
        print(f"Received request: font_path={font_path}")
        # Save uploaded image temporarily
        temp_path = f"temp_{image.filename}"
        print(f"Saving to temp path: {temp_path}")
        
        with open(temp_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        print(f"File saved, size: {len(content)} bytes")
        
        # Process image with font parameter
        print("Starting image processing...")
        translated_image_b64 = processor.replace_text_in_image(
            temp_path, source_lang, target_lang, font_path
        )
        
        print("Image processing completed")
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Cleaned up temp file: {temp_path}")
        
        return JSONResponse({
            "success": True,
            "translated_image": translated_image_b64
        })
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        # Clean up temp file in case of error
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up temp file after error: {temp_path}")
            except Exception as cleanup_error:
                print(f"Failed to cleanup temp file: {cleanup_error}")
        
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )

@app.get("/")
async def root():
    return {"message": "Image Text Translator API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)