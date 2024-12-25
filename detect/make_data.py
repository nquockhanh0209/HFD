import os
from typing import List
import cv2
import mediapipe as mp
import pandas as pd

class DataUtilities:
    data_video_paths: List[str]
    save_path: str
    def __init__(self, data_video_paths: List[str]):
        self.data_paths = data_video_paths
        self.save_path = "/home/khanh/HFD/dataset/"
    
    def convert_video_to_kinetics(self, label: str, saved_path_update: str = None):
        for data_path in self.data_paths:
            source_dir = data_path
            # Walk through the directory tree
            for root, dirs, files in os.walk(source_dir):
                for filename in files:
                    # Khởi tạo thư viện mediapipe
                    mpPose = mp.solutions.pose
                    pose = mpPose.Pose()
                    mpDraw = mp.solutions.drawing_utils
                    # Đọc ảnh từ video
                    file_path = os.path.join(root, filename)
                    cap = cv2.VideoCapture(file_path)

                    lm_list = []
                    label = label

                    def make_landmark_timestep(results):
                        
                        c_lm = []
                        for id, lm in enumerate(results.pose_landmarks.landmark):
                            c_lm.append(lm.x)
                            c_lm.append(lm.y)
                            c_lm.append(lm.z)
                            c_lm.append(lm.visibility)
                        return c_lm

                    def draw_landmark_on_image(mpDraw, results, img):
                        # Vẽ các đường nối
                        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                        # Vẽ các điểm nút
                        for id, lm in enumerate(results.pose_landmarks.landmark):
                            h, w, c = img.shape
                            print(id, lm)
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                        return img


                    while True:
                        ret, frame = cap.read()
                        if ret:
                            # Nhận diện pose
                            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(frameRGB)

                            if results.pose_landmarks:
                                # Ghi nhận thông số khung xương
                                lm = make_landmark_timestep(results)
                                lm_list.append(lm)
                                # Vẽ khung xương lên ảnh
                                frame = draw_landmark_on_image(mpDraw, results, frame)

                            cv2.imshow("image", frame)
                            if cv2.waitKey(1) == ord('q'):
                                break
                        else: break
                    if saved_path_update:
                        self.save_path = self.save_path + saved_path_update
                    # Write vào file csv
                    os.makedirs(self.save_path, exist_ok=True)
                    df  = pd.DataFrame(lm_list)
                    df.to_csv(self.save_path + label + ".txt")
                    cap.release()
                    cv2.destroyAllWindows()

DataUtilities(["/home/khanh/HFD/dataset/HFD/videos/ADL/"]).convert_video_to_kinetics(label="ADL", saved_path_update="HFD/kinetics/")