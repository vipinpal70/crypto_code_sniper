import os
import re
import sys
import traceback

# Ensure output is flushed immediately
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

print("Starting UUID extraction script...")

try:
    import easyocr
    import cv2
    import numpy as np
    print("Successfully imported required libraries")
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please install required libraries with: pip install easyocr opencv-python numpy")
    sys.exit(1)

def extract_text_from_image(image_path):
    """Extract all text from an image using EasyOCR"""
    try:
        # Initialize reader
        reader = easyocr.Reader(['en'], verbose=False)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return ""
            
        # Extract text from image
        results = reader.readtext(image)
        
        # Combine all text
        full_text = ' '.join([result[1] for result in results])
        
        print(f"Extracted text length: {len(full_text)} characters")
        print(f"Raw text: {full_text}")
        
        return full_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def extract_uuids(text):
    """Extract all possible UUIDs from text"""
    # Common UUID patterns
    patterns = [
        r'UID[:\s]*([A-Z]{2,3}\d{7,9})',     # UID: followed by code
        r'\b([A-Z]{2}\d{8})\b',              # Two letters + 8 digits
        r'\b([A-Z]{3}\d{7})\b',              # Three letters + 7 digits  
        r'\b([A-Z]{2}\d{7})\b',              # Two letters + 7 digits
        r'\b([A-Z]{1,3}\d{5,9})\b',          # General pattern
        r'\b([A-Z]{1,3}[-\s]?\d{5,9})\b',    # Letters followed by digits with optional separator
        r'\b(AY\s?\d{8})\b',                 # AY followed by 8 digits
        r'\b(ARS?\d{7})\b',                  # AR or ARS followed by digits
        r'\b(RA\d{8})\b',                    # RA followed by 8 digits
        r'\b(NI\d{4,8})\b',                  # NI followed by 4-8 digits
        r'\b(N\d{9})\b',                     # N followed by 9 digits
        r'\b(D\s?\d{5,8})\b',                # D followed by 5-8 digits
        r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b'  # Standard UUID format
    ]
    
    found_uuids = []
    
    # Process with original text
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            uuid = match.upper().strip()
            # Basic validation
            if len(uuid) >= 5 and uuid not in found_uuids:
                found_uuids.append(uuid)
    
    # Process with spaces removed
    text_no_spaces = text.replace(' ', '')
    for pattern in patterns:
        matches = re.findall(pattern, text_no_spaces, re.IGNORECASE)
        for match in matches:
            uuid = match.upper().strip()
            if len(uuid) >= 5 and uuid not in found_uuids:
                found_uuids.append(uuid)
    
    # Process with common OCR error corrections
    text_corrected = text.replace('O', '0').replace('I', '1').replace('l', '1').replace('S', '5')
    for pattern in patterns:
        matches = re.findall(pattern, text_corrected, re.IGNORECASE)
        for match in matches:
            uuid = match.upper().strip()
            if len(uuid) >= 5 and uuid not in found_uuids:
                found_uuids.append(uuid)
    
    # Manual pattern search for specific formats
    special_patterns = [
        r'AY\s?\d{8}',          # AY followed by 8 digits
        r'ARS?\d{7}',           # AR or ARS followed by digits
        r'RA\d{8}',             # RA followed by 8 digits
        r'NI\d{4,8}',           # NI followed by 4-8 digits
        r'N\d{9}',              # N followed by 9 digits
        r'D\s?\d{5,8}'          # D followed by 5-8 digits
    ]
    
    for pattern in special_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            uuid = match.upper().strip().replace(' ', '')
            if uuid not in found_uuids:
                found_uuids.append(uuid)
    
    # Clean up results
    cleaned_uuids = []
    for uuid in found_uuids:
        # Remove any trailing incomplete parts
        if uuid.endswith('<') or uuid.endswith('\n') or uuid.endswith(','):
            uuid = uuid.rstrip('<\n,')
        # Add to cleaned list if valid
        if len(uuid) >= 5 and uuid not in cleaned_uuids:
            cleaned_uuids.append(uuid)
    
    return cleaned_uuids

def process_image(image_path):
    """Process a single image to extract UUIDs"""
    print(f"Processing image: {image_path}")
    
    # Extract text from image
    text = extract_text_from_image(image_path)
    
    # Extract UUIDs from text
    uuids = extract_uuids(text)
    
    print(f"Found UUIDs ({len(uuids)}):")
    for i, uuid in enumerate(uuids, 1):
        print(f"{i}. {uuid}")
    
    return uuids

if __name__ == "__main__":
    try:
        # Process the image
        image_path = "coindcx2.jpg"
        print(f"\nChecking if image exists: {os.path.exists(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found!")
            print(f"Current directory: {os.getcwd()}")
            print("Available files:")
            for file in os.listdir('.'):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"  - {file}")
            sys.exit(1)
        
        # Process the image
        uuids = process_image(image_path)
        
        # Save results to file
        output_file = "extracted_uuids.txt"
        with open(output_file, "w") as f:
            f.write(f"UUIDs extracted from {image_path}:\n")
            f.write("-" * 40 + "\n")
            for i, uuid in enumerate(uuids, 1):
                f.write(f"{i}. {uuid}\n")
        
        print(f"\nExtracted {len(uuids)} UUIDs and saved to {output_file}")
        
        # Print results again for clarity
        print("\nExtracted UUIDs:")
        for uuid in uuids:
            print(f"  {uuid}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        sys.exit(1)
