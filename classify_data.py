import os
import shutil

# Define the source directory and target directories
video_fall_dir = os.path.join('dataset/HFD/videos/Fall')
video_adl_dir = os.path.join('dataset/HFD/videos/ADL')

csv_dir = os.path.join('dataset/HFD/csv_files')

# Create target directories if they don't exist
os.makedirs(video_fall_dir, exist_ok=True)
os.makedirs(video_adl_dir, exist_ok=True)

os.makedirs(csv_dir, exist_ok=True)
for i in range(4):
    source_dir = f'./collected_data/Subject {i+1}'
    # Walk through the directory tree
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            # Determine the target path based on file type
            if filename.lower().endswith('.csv'):
               
                    target_path = os.path.join(csv_dir, str(i+1)+"_"+filename)
            else:
                continue
            
            # Check if the file already exists in the target directory
            if os.path.exists(target_path):
                print(f"File {filename} already exists in the target directory. Skipping.")
            else:
                shutil.move(file_path, target_path)
    

    source_dir = f'./collected_data/Subject {i+1}/Fall'
    # Walk through the directory tree
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            # Determine the target path based on file type
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
      
                    target_path = os.path.join(video_fall_dir, str(i+1)+"_"+filename)
      
            else:
                continue
            
            # Check if the file already exists in the target directory
            if os.path.exists(target_path):
                print(f"File {filename} already exists in the target directory. Skipping.")
            else:
                shutil.move(file_path, target_path)

    source_dir = f'./collected_data/Subject {i+1}/ADL'
    # Walk through the directory tree
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            # Determine the target path based on file type
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                
                    target_path = os.path.join(video_adl_dir, str(i+1)+"_"+filename)
      
            else:
                continue
            
            # Check if the file already exists in the target directory
            if os.path.exists(target_path):
                print(f"File {filename} already exists in the target directory. Skipping.")
            else:
                shutil.copy(file_path, target_path)



