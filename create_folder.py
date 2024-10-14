import os
zip_f="Videos"
zip_f_noExt = os.path.splitext(os.path.basename(zip_f))[0]
featureBasePath1=zip_f_noExt+"_Dataset"
os.makedirs(featureBasePath1, exist_ok=True)
os.makedirs(os.path.join(featureBasePath1, "Fall"),exist_ok=True)
os.makedirs(os.path.join(featureBasePath1, "ADL"),exist_ok=True)
os.makedirs(os.path.join(featureBasePath1, "results"),exist_ok=True)
num_cameras = 4  # Adjust this based on the actual number of cameras
# Generate the camera folder names dynamically
cams = [f"cam{i+1}" for i in range(num_cameras)]
cams
for cam in cams:
    cam_pathF = os.path.join(featureBasePath1, f"Fall/{cam}")
    os.makedirs(cam_pathF, exist_ok=True)
    cam_pathA = os.path.join(featureBasePath1, f"ADL/{cam}")
    os.makedirs(cam_pathA, exist_ok=True)


featureBasePath1