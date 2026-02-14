"""
Bhai, ye OCR ka module hai. Image se text nikalega.
Tesseract aur EasyOCR dono try karega, jo better ho use karega.
"""

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract not installed. Install: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ easyocr not installed. Install: pip install easyocr")

from PIL import Image
import numpy as np
import traceback

class OCRHandler:
    def __init__(self):
        # Initialize EasyOCR if available
        self.easy_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easy_reader = easyocr.Reader(['en'])
                print("✅ EasyOCR initialized")
            except Exception as e:
                print(f"❌ EasyOCR init error: {e}")
        
    def extract_with_tesseract(self, image):
        """Tesseract se extract karo"""
        try:
            if not TESSERACT_AVAILABLE:
                return "", 0.0
            
            # Windows ke liye tesseract path set karo
            # try:
            #     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            # except:
            #     pass  # Linux/Mac pe default use hoga
            
            # Convert PIL Image if needed
            if isinstance(image, Image.Image):
                # Preprocess image for better OCR
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to numpy for preprocessing
                img_array = np.array(image)
                
                # Simple preprocessing
                if len(img_array.shape) == 3:
                    # Convert to grayscale
                    import cv2
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    
                    # Apply thresholding
                    _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                    
                    # Convert back to PIL
                    processed_image = Image.fromarray(processed)
                else:
                    processed_image = image
            else:
                processed_image = Image.open(image) if isinstance(image, str) else image
            
            # Extract text
            text = pytesseract.image_to_string(processed_image, lang='eng')
            
            # Get confidence
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            
            if confidences:
                confidence = sum(confidences) / len(confidences) / 100  # Normalize to 0-1
            else:
                confidence = 0.5 if text.strip() else 0.0
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"❌ Tesseract error: {str(e)[:100]}")
            return "", 0.0
    
    def extract_with_easyocr(self, image):
        """EasyOCR se extract karo"""
        try:
            if not EASYOCR_AVAILABLE or not self.easy_reader:
                return "", 0.0
            
            # Convert to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                # Convert RGB to BGR for EasyOCR
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = img_array[:, :, ::-1]  # RGB to BGR
            elif isinstance(image, str):
                # Load from file path
                img_array = np.array(Image.open(image))
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = img_array[:, :, ::-1]
            else:
                img_array = image
            
            # Perform OCR
            results = self.easy_reader.readtext(img_array, paragraph=True)
            
            if not results:
                return "", 0.0
            
            # Combine all text
            text = ' '.join([result[1] for result in results])
            
            # Calculate average confidence
            confidence = sum([result[2] for result in results]) / len(results)
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"❌ EasyOCR error: {str(e)[:100]}")
            return "", 0.0
    
    def extract_text(self, image):
        """
        Dono methods try karo, jo better result de use karo.
        Image: PIL Image object ya file path
        Returns: (text, confidence)
        """
        try:
            # Validate input
            if image is None:
                return "No image provided", 0.0
            
            # Try both methods
            text1, conf1 = "", 0.0
            text2, conf2 = "", 0.0
            
            if TESSERACT_AVAILABLE:
                text1, conf1 = self.extract_with_tesseract(image)
                print(f"📝 Tesseract: {conf1:.2%} confidence")
            
            if EASYOCR_AVAILABLE and self.easy_reader:
                text2, conf2 = self.extract_with_easyocr(image)
                print(f"📝 EasyOCR: {conf2:.2%} confidence")
            
            # Choose the best result
            if conf1 >= conf2 and text1.strip():
                return text1, conf1
            elif text2.strip():
                return text2, conf2
            elif text1.strip():
                return text1, conf1
            else:
                return "No text could be extracted from the image", 0.0
                
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            traceback.print_exc()
            return f"OCR Error: {str(e)[:100]}", 0.0

# Global instance
ocr_handler = OCRHandler()

def extract_text_from_image(image):
    """Simple function for Streamlit app"""
    return ocr_handler.extract_text(image)
