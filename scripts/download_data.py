import os
import requests

ACCESS_TOKEN = "RE0bb4S4V9GaSh8j8LmG9qghFNgM8WAKKm5ZSwQ3k9Z0nDShoTjZPBQoiXUm"
record_id = "14223624"  # LUNA25 record id

# Files you want to download
target_files = {"luna25_nodule_blocks.zip.001", "luna25_nodule_blocks.zip.002"}

# Output folder
output_folder = "/vol/csedu-nobackup/course/IMC037_aimi/group13/data"
os.makedirs(output_folder, exist_ok=True)

# Get metadata
r = requests.get(f"https://zenodo.org/api/records/{record_id}", params={'access_token': ACCESS_TOKEN})
if r.status_code != 200:
    print("Error retrieving record:", r.status_code, r.text)
    exit()

# Filter for target files
files = r.json().get('files', [])
selected_files = [f for f in files if f['key'] in target_files]

print(f"Total files to download: {len(selected_files)}")

# Download selected files
for index, f in enumerate(selected_files):
    filename = f['key']
    url = f['links']['self']
    file_path = os.path.join(output_folder, filename)

    print(f"Downloading file {index+1}/{len(selected_files)}: {filename}")

    with requests.get(url, params={'access_token': ACCESS_TOKEN}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as out_file:
            for i, chunk in enumerate(r.iter_content(chunk_size=1048576)):  # 1 MB
                if chunk:
                    out_file.write(chunk)
                    if i % 10 == 0:
                        print(f"  Downloaded {i} MB chunks of {filename}")

    print(f"Completed: {filename}")

print("All selected downloads completed successfully!")

