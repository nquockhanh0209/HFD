# Utility functions for the experiments (chunk count, video preprocessing, feature computation, )
from keras.models import model_from_json, Model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from C3D import create_C3D_model, frame_n
from create_folder import featureBasePath1, zip_f_noExt, cams

def getFeatureExtractor(weigthsPath, layer, verbose = False):


    model = create_C3D_model(verbose)
    model.load_weights(weigthsPath)
    model.compile(loss='mean_squared_error', optimizer='sgd')

    return Model(inputs=model.input,outputs=model.get_layer(layer).output)

def count_chunks(videoBasePath):


    folders = ['Fall', 'ADL']
    # # cams = ['cam1']
    #   # Number of cameras
    # num_cameras = 2  # Adjust this based on the actual number of cameras
    # # Generate the camera folder names dynamically
    # cams = [f"cam{i+1}" for i in range(num_cameras)]

    cnt = 0

    # TC=0

    for folder in folders:
        print("cams",cams)
        for camName in cams:
            path = os.path.join(videoBasePath, folder, camName)

            # if TC>=10:
            #   break
            # TC=TC+1

            try:
              # Attempt to access the subfolder
              videofiles = os.listdir(path)

            # try:
            # # Attempt to access the subfolder
            # videofiles = os.listdir(path)
            except FileNotFoundError:
              # Subfolder doesn't exist, continue with the next one
              continue
            for videofile in videofiles:
                filePath = os.path.join(path, videofile)
                video = cv2.VideoCapture(filePath)
                numframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                chunks = numframes//frame_n
                cnt += chunks


    return cnt

def preprocessVideos(videoBasePath, featureBasePath, verbose=True):


    folders = ['Fall', 'ADL']
    # Number of cameras
    # num_cameras = 2  # Adjust this based on the actual number of cameras
    # # Generate the camera folder names dynamically
    # cams = [f"cam{i+1}" for i in range(num_cameras)]
    # # cams = ['cam1']
    total_chunks = count_chunks(videoBasePath)
    npSamples = np.memmap(os.path.join(featureBasePath, 'samples.mmap'), dtype=np.float32, mode='w+', shape=(total_chunks, 16, 112, 112, 3))
    npLabels = np.memmap(os.path.join(featureBasePath, 'labels.mmap'), dtype=np.int8, mode='w+', shape=(total_chunks))
    cnt = 0

    # TC=0

    for folder in folders:
        for camName in cams:
            path = os.path.join(videoBasePath, folder, camName)
            # if TC>=10:
            #   break
            # TC=TC+1


            try:
              videofiles = os.listdir(path)

            except FileNotFoundError:
              # Subfolder doesn't exist, continue with the next one
              continue
            for videofile in videofiles:
                filePath = os.path.join(path, videofile)
                video = cv2.VideoCapture(filePath)
                numframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                chunks = numframes//frame_n
                if verbose:
                    print(filePath)
                    print("*** [Video Info] Number of frames: {} - fps: {} - chunks: {}".format(numframes, fps, chunks))
                vid = []
                videoFrames = []
                while True:
                    ret, img = video.read()
                    if not ret:
                        break
                    videoFrames.append(cv2.resize(img, (112, 112)))
                vid = np.array(videoFrames, dtype=np.float32)
                filename = os.path.splitext(videofile)[0]
                chunk_cnt = 0
                for i in range(chunks):
                    X = vid[i*frame_n:i*frame_n+frame_n]
                    chunk_cnt += 1
                    npSamples[cnt] = np.array(X, dtype=np.float32)
                    if folder == 'Fall':
                        npLabels[cnt] = np.int8(1)
                    else:
                        npLabels[cnt] = np.int8(0)
                    cnt += 1

    if verbose:
        print("** Labels **")
        print(npLabels.shape)
        print('\n****\n')
        print("** Samples **")
        print(npSamples.shape)
        print('\n****\n')

    del npSamples
    del npLabels

def extractFeatures(weigthsPath, videoBasePath, featureBasePath='', verbose=True):



    featureExtractor = getFeatureExtractor(weigthsPath, 'fc6', verbose)

    folders = ['Fall', 'ADL']
    # # cams = ['cam1']
    #   # Number of cameras
    # num_cameras = 2  # Adjust this based on the actual number of cameras
    # # Generate the camera folder names dynamically
    # cams = [f"cam{i+1}" for i in range(num_cameras)]
    labels = []
    features = []

    # TC=0
    AllVdoFrames=0

    for folder in folders:
        for camName in cams:
            path = os.path.join(videoBasePath, folder, camName)
            featurepath = os.path.join(featureBasePath, folder, camName)
            # if TC>=10:
            #   break
            # TC=TC+1

            try:
              videofiles = os.listdir(path)

            except FileNotFoundError:
              print("Subfolder doesn't exist, continue with the next one")
              # Subfolder doesn't exist, continue with the next one
              continue
            for videofile in videofiles:
                filePath = os.path.join(path, videofile)
                video = cv2.VideoCapture(filePath)
                numframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                AllVdoFrames=AllVdoFrames+numframes
                fps = int(video.get(cv2.CAP_PROP_FPS))
                chunks = numframes//frame_n
                if verbose:
                    print(filePath)
                    print("*** [Video Info] Number of frames: {} - fps: {} - chunks: {}".format(numframes, fps, chunks))
                vid = []
                videoFrames = []
                while True:
                    ret, img = video.read()
                    if not ret:
                        break
                    videoFrames.append(cv2.resize(img, (112, 112)))
                vid = np.array(videoFrames, dtype=np.float32)

                filename = os.path.splitext(videofile)[0]
                if featureBasePath:
                    featureFilePath = os.path.join(featurepath, filename + '.csv')
                    with open(featureFilePath, 'ab') as f:
                        for i in range(chunks):
                            X = vid[i*frame_n:i*frame_n+frame_n]
                            out = featureExtractor.predict(np.array([X]))
                            np.savetxt(f, out)
                            out = out.reshape(4096)
                            features.append(out)
                            if folder == 'Fall':
                                labels.append(1)
                            else:
                                labels.append(0)

                    if verbose:
                        print('*** Saved file: ' + featureFilePath)
                        print('\n')
                else:
                    for i in range(chunks):
                        X = vid[i*frame_n:i*frame_n+frame_n]
                        out = featureExtractor.predict(np.array([X]))
                        out = out.reshape(4096)
                        features.append(out)
                        if folder == 'Fall':
                            labels.append(1)
                        else:
                            labels.append(0)

    y = np.array(labels)
    X = np.array(features)

    if verbose:
        print("** Labels **")
        # print(y)
        print(y.shape)
        print('\n****\n')
        print("** Features **")
        # print(X)
        print(X.shape)
        print('\n****\n')
        print("AllVdoFrames:",AllVdoFrames,"AvgNoChunks:",AllVdoFrames//16)

    return X, y

def get_labels_and_features_from_files(basePath, verbose=True):

    folders = ['Fall', 'ADL']
    # # cams = ['cam1']
    #   # Number of cameras
    # num_cameras = 2  # Adjust this based on the actual number of cameras
    # # Generate the camera folder names dynamically
    # cams = [f"cam{i+1}" for i in range(num_cameras)]
    labels = []
    features = []

    # TC=0

    for folder in folders:
        for camName in cams:
            path = os.path.join(basePath, folder, camName)
            # if TC>=10:
            #   break
            # TC=TC+1
            try:
              textfiles = os.listdir(path)

            except FileNotFoundError:
              # Subfolder doesn't exist, continue with the next one
              continue
            for textfile in textfiles:
                filePath = os.path.join(path, textfile)
                chunks = np.loadtxt(filePath)
                for chunk in chunks:
                  features.append(chunk)
                  if folder == 'Fall':
                    labels.append(1)
                  else:
                    labels.append(0)

    y = np.array(labels)
    X = np.array(features)

    if verbose:
        print("** Labels **")
        print("*************cams",cams)
        # print(y)
        print(y.shape)
        print('\n****\n')
        print("** Features **")
        # print(X)
        print(X.shape)
        print('\n****\n')
        print("*************cams",cams)

    return X, y



extractFeatures('weights/weights.h5', zip_f_noExt, featureBasePath1, True)