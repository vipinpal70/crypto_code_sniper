from pymongo import MongoClient
import re

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string if needed
db = client['crypto_sniper']  # Update with your collection name
users = db['users']  # Update with your collection name

# List of broker IDs to search
uuids = [
    '0A33202797I0',
    'AR54852510',
    'D 540549741',
    'N104556042',
    'D 23775',
    'D54054974',
    'D23775'
]

print("Searching for broker IDs in MongoDB...")

# Method 1: Simple search with space removal and last 6 digits
print("\n=== Method 1: Original approach with space removal and last 6 digits ===")
for x in uuids:
    # Try exact match with space removal
    ss = x.replace(" ", "")
    rs = users.find_one({"broker_connection.broker_id": ss})
    if rs:
        print(f"Found match for '{x}' (as '{ss}'): {rs.get('username', 'Unknown')}")
    
    # Try matching last 6 digits
    digits_only = re.sub(r'\D', '', ss)  # Remove all non-digits
    if len(digits_only) >= 6:
        last_six = digits_only[-6:]
        rs = users.find_one({"broker_connection.broker_id": {"$regex": f"{last_six}$", "$options": "i"}})
        if rs:
            print(f"Found match for '{x}' by last 6 digits '{last_six}': {rs.get('username', 'Unknown')}")

# Method 2: Using regex pattern matching with last 6 digits
print("\n=== Method 2: Using regex pattern matching with last 6 digits ===")
for x in uuids:
    # Extract digits only
    no_spaces = x.replace(" ", "")
    digits_only = re.sub(r'\D', '', no_spaces)  # Remove all non-digits
    
    if len(digits_only) >= 6:
        last_six = digits_only[-6:]  # Get last 6 digits
        # Search for any broker ID ending with these 6 digits
        rs = users.find_one({"broker_connection.broker_id": {"$regex": f"{last_six}$", "$options": "i"}})
        if rs:
            print(f"Found match for '{x}' (last 6 digits: {last_six}): {rs.get('username', 'Unknown')}")
    
    # Also try the full pattern as a fallback
    pattern = ""
    for char in x:
        if char == " ":
            pattern += r"\s*"  # Make spaces optional
        else:
            pattern += re.escape(char)  # Escape special characters
    
    # Search with MongoDB's regex operator
    rs = users.find_one({"broker_connection.broker_id": {"$regex": pattern, "$options": "i"}})
    if rs:
        print(f"Found match for '{x}' using full pattern: {rs.get('username', 'Unknown')}")

# Method 3: Batch search with multiple patterns using $or
print("\n=== Method 3: Batch search with multiple patterns using $or ===")

# Create a list of conditions for all broker IDs
or_conditions = []
for x in uuids:
    # Add the original ID as exact match
    or_conditions.append({"broker_connection.broker_id": x})
    
    # Add ID without spaces as exact match
    no_spaces = x.replace(" ", "")
    or_conditions.append({"broker_connection.broker_id": no_spaces})
    
    # Extract last 6 digits if available
    digits_only = re.sub(r'\D', '', no_spaces)  # Remove all non-digits
    if len(digits_only) >= 6:
        last_six = digits_only[-6:]  # Get last 6 digits
        # Match any broker ID ending with these 6 digits
        or_conditions.append({"broker_connection.broker_id": {"$regex": f"{last_six}$", "$options": "i"}})
    
    # Add regex pattern for flexible matching with original ID
    pattern = ""
    for char in x:
        if char == " ":
            pattern += r"\s*"  # Make spaces optional
        else:
            pattern += re.escape(char)  # Escape special characters
    
    or_conditions.append({"broker_connection.broker_id": {"$regex": pattern, "$options": "i"}})

# Search using $or operator
results = users.find({"$or": or_conditions})

found_count = 0
for result in results:
    found_count += 1
    broker_id = result.get('broker_connection', {}).get('broker_id', 'Unknown')
    username = result.get('username', 'Unknown')
    print(f"Found user '{username}' with broker ID: {broker_id}")

if found_count == 0:
    print("No matches found with any pattern.")
else:
    print(f"Found {found_count} total matches.")

print("\nSearch completed.")

# Method 4: Individual regex searches with last 6 digits
print("\n=== Method 4: Individual regex searches with last 6 digits ===")
print("This method searches for each broker ID focusing on the last 6 digits")

for x in uuids:
    print(f"\nSearching for: {x}")
    
    # Extract digits only
    no_spaces = x.replace(" ", "")
    digits_only = re.sub(r'\D', '', no_spaces)  # Remove all non-digits
    
    if len(digits_only) >= 6:
        last_six = digits_only[-6:]  # Get last 6 digits
        print(f"  Looking for last 6 digits: {last_six}")
        
        # Search for any broker ID ending with these 6 digits
        results = users.find({"broker_connection.broker_id": {"$regex": f"{last_six}$", "$options": "i"}})
    else:
        print(f"  Not enough digits to extract last 6 (found {len(digits_only)})")
        # Fall back to regular pattern matching
        pattern = ""
        for char in x:
            if char == " ":
                pattern += r"\s*"  # Make spaces optional
            else:
                pattern += re.escape(char)  # Escape special characters
        
        results = users.find({"broker_connection.broker_id": {"$regex": pattern, "$options": "i"}})
    
    # Count results
    count = 0
    for result in results:
        count += 1
        broker_id = result.get('broker_connection', {}).get('broker_id', 'Unknown')
        username = result.get('username', 'Unknown')
        print(f"  Match: '{username}' with broker ID: {broker_id}")
    
    if count == 0:
        print(f"  No matches found for '{x}'")
    else:
        print(f"  Found {count} matches for '{x}'")

print("\nAll searches completed.")
