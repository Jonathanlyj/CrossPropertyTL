import json
import requests
from pathlib import Path
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Download files from a Figshare article.')
parser.add_argument('--item_id', type=int, required=True, help='Figshare article item ID')
parser.add_argument('--dir_path', required=True, help='Full directory path to save the files')

args = parser.parse_args()
item_id = args.item_id
dir_path = Path(args.dir_path)

# --- Figshare API ---
BASE_URL = 'https://api.figshare.com/v2'

# Collect file metadata
r = requests.get(f'{BASE_URL}/articles/{item_id}/files')
file_metadata = json.loads(r.text)

# Create directory if it does not exist
dir_path.mkdir(parents=True, exist_ok=True)

# Download files
for file_entry in file_metadata:
    response = requests.get(f'{BASE_URL}/file/download/{file_entry["id"]}')
    file_path = dir_path / file_entry['name']
    with open(file_path, 'wb') as f:
        f.write(response.content)

print(f'All files saved to: {dir_path}')