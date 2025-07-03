import os
import re
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import pymongo


MONGO_URL = "mongodb+srv://upadhyaymanisha13:Manisha%401306@cluster0.opfmq9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

mongo_client = pymongo.MongoClient(MONGO_URL)
db = mongo_client["CryptoSniper"]
users = db["users"]



def extract_uids_with_easyocr(image_path):
    """Extract UIDs using EasyOCR (recommended - no system dependencies)"""
    try:
        import easyocr
        
        # Initialize reader (downloads model on first use)
        reader = easyocr.Reader(['en'], verbose=False)
        
        # Extract text from image
        results = reader.readtext(str(image_path))
        
        # Combine all text
        full_text = ' '.join([result[1] for result in results])
        
        print(f"  Extracted text length: {len(full_text)} characters")
        
        return full_text
        
    except ImportError:
        print("  EasyOCR not installed. Install with: pip install easyocr")
        return ""
    except Exception as e:
        print(f"  EasyOCR error: {e}")
        return ""

def extract_uids_from_text(text):
    """Extract UIDs from text using regex patterns"""
    
    # Common UID patterns - enhanced for better detection
    patterns = [
        r'UID[:\s]*([A-Z]{2,3}\d{7,9})',     # UID: followed by code
        r'\b([A-Z]{2}\d{8})\b',              # Two letters + 8 digits
        r'\b([A-Z]{3}\d{7})\b',              # Three letters + 7 digits  
        r'\b([A-Z]{2}\d{7})\b',              # Two letters + 7 digits
        r'([A-Z]{2,3}\d{7,9})',              # General pattern
        r'([A-Z]{1,3}[-\s]?\d{5,9})',        # Letters followed by digits with optional separator
        r'([A-Z]{1,3}\d{5,9}[A-Z]{0,2})',    # Letters-digits-optional letters format
        r'ID[:\s]*([A-Za-z0-9]{8,12})',      # ID: followed by alphanumeric
        r'UUID[:\s]*([A-Za-z0-9-]{8,36})',   # UUID format
        r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b',  # Standard UUID format
        r'\b([A-Z]{2}\d{7,9})\b',            # Two letters followed by 7-9 digits
        r'\b([A-Z]\d{5,10})\b',              # One letter followed by 5-10 digits
        r'\b([A-Z]\s\d{5,8})\b',            # One letter space digits format
        r'\b(\d{5,8})\b',                    # Just digits (5-8 digits)
        r'\b(N\d{8,9})\b',                   # N followed by 8-9 digits
        r'\b(D\s\d{5,8})\b'                  # D space digits format
    ]
    
    found_uids = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            uid = match.upper().strip()
            # Basic validation
            if len(uid) >= 5 and uid not in found_uids:
                found_uids.append(uid)
    
    return found_uids

def find_image_files(directory):
    """Find all image files in directory"""
    
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_files = []
    
    directory = Path(directory)
    
    for ext in extensions:
        # Find files with both lowercase and uppercase extensions
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(list(set(image_files)))  # Remove duplicates and sort



def process_images_in_directory(directory_path, output_folder="uid_extraction_results"):
    """ Process all images in directory and extract UIDs """
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Find image files
    image_files = find_image_files(directory_path)
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    print("-" * 50)
    
    all_results = []
    all_uids = []
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            # Extract text from image
            extracted_text = extract_uids_with_easyocr(image_path)
            
            if not extracted_text:
                print("  No text extracted")
                continue
            
            # Find UIDs in the text
            uids = extract_uids_from_text(extracted_text)
            
            # Store result
            result = {
                'file_name': image_path.name,
                'file_path': str(image_path),
                'uids_count': len(uids),
                'uids': uids,
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            all_results.append(result)
            all_uids.extend(uids)
            
            if uids:
                print(f"  Found {len(uids)} UIDs: {', '.join(uids)}")
            else:
                print("  No UIDs found")
                
        except Exception as e:
            print(f"  Error: {e}")
            result = {
                'file_name': image_path.name,
                'file_path': str(image_path),
                'error': str(e),
                'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            all_results.append(result)
    
    # Remove duplicate UIDs
    unique_uids = list(dict.fromkeys(all_uids))
    
    # Save results
    save_results(all_results, unique_uids, output_folder)
    
    return all_results, unique_uids

def save_results(all_results, unique_uids, output_folder):
    """Save extraction results in multiple formats"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save all UIDs to text file
    uids_file = Path(output_folder) / f"extracted_uids_{timestamp}.txt"
    with open(uids_file, 'w') as f:
        f.write(f"UID Extraction Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total unique UIDs found: {len(unique_uids)}\n")
        f.write(f"Total files processed: {len(all_results)}\n\n")
        f.write("All UIDs:\n")
        f.write("-" * 20 + "\n")
        
        for i, uid in enumerate(unique_uids, 1):
            f.write(f"{i:2d}. {uid}\n")
        
        f.write(f"\nUIDs for copy-paste:\n")
        f.write("-" * 20 + "\n")
        for uid in unique_uids:
            f.write(f"{uid}\n")
    
    # 2. Save detailed results as JSON
    json_file = Path(output_folder) / f"detailed_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'summary': {
                'total_files': len(all_results),
                'total_unique_uids': len(unique_uids),
                'extraction_date': datetime.now().isoformat()
            },
            'unique_uids': unique_uids,
            'file_results': all_results
        }, f, indent=2)
    
    # 3. Save as CSV
    csv_data = []
    for result in all_results:
        if 'uids' in result:
            for uid in result['uids']:
                csv_data.append({
                    'file_name': result['file_name'],
                    'uid': uid,
                    'processed_at': result['processed_at']
                })
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_file = Path(output_folder) / f"uids_by_file_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Create summary CSV
        summary_data = []
        for result in all_results:
            if 'uids' in result:
                summary_data.append({
                    'file_name': result['file_name'],
                    'uid_count': result['uids_count'],
                    'uids': ', '.join(result['uids']) if result['uids'] else '',
                    'processed_at': result['processed_at']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = Path(output_folder) / f"summary_{timestamp}.csv"
        summary_df.to_csv(summary_csv, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETED!")
    print("=" * 60)
    print(f"Files processed: {len(all_results)}")
    print(f"Total unique UIDs: {len(unique_uids)}")
    print(f"Results saved to: {output_folder}/")
    
    if unique_uids:
        print(f"\nExtracted UIDs:")
        for i, uid in enumerate(unique_uids, 1):
            print(f"  {i:2d}. {uid}")
    
    print(f"\nOutput files:")
    print(f"  - {uids_file.name} (UIDs list)")
    print(f"  - {json_file.name} (detailed results)")

    if csv_data:
        print(f"  - {csv_file.name} (CSV format)")
        print(f"  - {summary_csv.name} (summary)")

def main():
    """Main function - run the UID extraction"""
    
    print("Multi-File UID Extractor")
    print("=" * 40)
    print("This tool extracts UIDs from all images in a directory")
    print()
    
    # Get directory path
    while True:
        directory = input("Enter the directory path containing images: ").strip()
        
        if not directory:
            directory = "."
        
        directory = directory.strip('"')  # Remove quotes if present
        
        if os.path.exists(directory):
            break
        else:
            print(f"Directory '{directory}' does not exist. Please try again.")
    
    print(f"Processing images in: {os.path.abspath(directory)}")
    
    # Process images
    try:
        results, uids = process_images_in_directory(directory)
        
        if uids:
            print(f"\n Successfully extracted {len(uids)} unique UIDs!")
        else:
            print("\n No UIDs were found in any images.")
            print("Make sure:")
            print("  1. Images contain clear, readable text")
            print("  2. UIDs are in standard format (e.g., AB12345678)")
            print("  3. EasyOCR is installed: pip install easyocr")
        
        return results, uids
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Utility functions
def extract_from_single_image(image_path):
    """Extract UIDs from a single image file with enhanced detection"""
    print(f"Processing single image: {image_path}")
    
    # Extract text using EasyOCR
    text = extract_uids_with_easyocr(image_path)
    # print(f"Raw extracted text: {text}")
    
    # Try different preprocessing techniques to improve detection
    # 1. Original text
    uids = extract_uids_from_text(text)
    
    # 2. Remove spaces to handle cases where OCR splits characters
    text_no_spaces = text.replace(' ', '')
    uids_no_spaces = extract_uids_from_text(text_no_spaces)
    
    # 3. Try with common OCR error corrections
    text_corrected = text.replace('O', '0').replace('I', '1').replace('l', '1').replace('S', '5')
    uids_corrected = extract_uids_from_text(text_corrected)
    
    # 4. Try with specific patterns that might be in this image
    special_patterns = [
        r'AY\s?\d{8}',          # AY followed by 8 digits
        r'ARS?\d{7}',           # AR or ARS followed by digits
        r'RA\d{8}',             # RA followed by 8 digits
        r'NI\d{4,8}',           # NI followed by 4-8 digits
        r'N\d{9}',              # N followed by 9 digits
        r'D\s?\d{5,8}'          # D followed by 5-8 digits
    ]
    
    special_uids = []
    for pattern in special_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            uid = match.upper().strip().replace(' ', '')
            if uid not in special_uids:
                special_uids.append(uid)
    
    # Combine all results
    all_uids = uids + [uid for uid in uids_no_spaces if uid not in uids]
    all_uids = all_uids + [uid for uid in uids_corrected if uid not in all_uids]
    all_uids = all_uids + [uid for uid in special_uids if uid not in all_uids]
    
    # Clean up the results
    cleaned_uids = []
    for uid in all_uids:
        # Remove any trailing incomplete parts
        if uid.endswith('<') or uid.endswith('\n') or uid.endswith(','): 
            uid = uid.rstrip('<\n,')
        # Add to cleaned list if valid
        if len(uid) >= 5 and uid not in cleaned_uids:
            cleaned_uids.append(uid)
    
    # print(f"Found UIDs: {cleaned_uids}")
    return cleaned_uids

def batch_extract(directory_path):
    """Simple batch extraction function"""
    return process_images_in_directory(directory_path)

if __name__ == "__main__":
    # Check if EasyOCR is installed
    try:
        import easyocr
        print("✓ EasyOCR is available")
    except ImportError:
        print("❌ EasyOCR not found!")
        print("Install it with: pip install easyocr opencv-python pandas")
        print("Then run this script again.")
        exit(1)
    
    # Run main function  
    # main()


    uids = extract_from_single_image("coindcx4.jpg")


    # print(uids)
    for x in uids:
        ss = x.replace(" ","")
        last_five = ss[-5:]
        query = {"status":"Pending","broker_connection.broker_id": {"$regex": f"{last_five}$", "$options": "i"}}
        rs =users.find_one(query)
        if rs:
            print(rs)
            users.update_one(query, {"$set": {"status": "Approved"}})
            users.update_one(query, {"$set": {"approved_at": datetime.now()}})




# Example usage in script:
# For single image: uids = extract_from_single_image("path/to/your/image.jpg")
# For directory: results, all_uids = batch_extract("path/to/your/directory")
# For your specific case: results, all_uids = batch_extract(r"D:/CryptoSniper/Backtest")