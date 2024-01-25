import cv2
import dlib
from imutils import face_utils
import imutils
from scipy.spatial import distance
import numpy as np
import pandas as pd
import os
import warnings
import sys

# To suppress all warnings
warnings.filterwarnings("ignore")

# users_list = [ "person12", "person14", "person15", "person16", "person17", "person18", "person19", "person20", "person21", "person22", "person23"]
users_list = ["person1"]
blink_types = ["short_blink", "long_blink"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


for user_nm in users_list:
    for blink_type in blink_types:
        # path to frames
        dir_path = f"./data/videos/frames/{user_nm}/{blink_type}/"
        frames_num = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])


        # ##### Old code 
        # # define user and blink type to extract the ear for that category
        # user_nm = "person6"
        # blink_type = "short_blink" #switch between short_blink and long_blink
        # # blink_type = "long_blink"
        # frames_num = 606 # 0 to 599
        # ##### Old code end

        # create the folder to store ear vector(if not already present)
        folder_path = "data/ear_vecs_per_user/" + user_nm
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        ears = []

        for i in range(frames_num):
            frame = cv2.imread(filename="data/videos/frames/" + 
                            user_nm + "/" + blink_type + "/" + 
                            user_nm + "_" + blink_type + "_frame_" + str(i)+ ".jpg")
            # filename="data/videos/frames/" + \
                            #    user_nm + "/" + blink_type + "/" + \
                            #    user_nm + "_" + blink_type + "_frame_" + str(i)+ ".jpg"
            # print(f"frame name : {filename}")
            frame = imutils.resize(frame, width=540)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # find faces in frame
            subjects = detect(gray, 0)

            if len(subjects) == 0:
                print(f"No faces found in {i}th frame for {user_nm} {blink_type} category....exiting")
                sys.exit()

            for subject in subjects:
                # find location of both eyes
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)#converting to NumPy Array
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                ears.append(ear)
                # code to draw the eye hulls on the frame image and save it for analysis
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, f"EAR: {ear}", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
                labelled_frame_folder_name = "data/videos/frames/" + \
                               user_nm + "/" + blink_type + "-labelled/" 
                if not os.path.exists(labelled_frame_folder_name):
                    os.makedirs(labelled_frame_folder_name)
                # print(f"labelled_frame_filename: {labelled_frame_filename}")
                cv2.imwrite(labelled_frame_folder_name + "/" + user_nm + "_" + blink_type + "_frame_" + str(i)+ "_labelled.jpg", frame)

        ears_df = pd.DataFrame(ears)
        ears_df.to_excel("data/ear_vecs_per_user/" + user_nm + "/" + user_nm + "-" + blink_type + "-ears.xlsx", index=False)
        print(f"{user_nm} {blink_type}: Collected EARs for each frame... contructing EAR vectors now")

        ear_v_len = 15
        step_size = 5
        ear_vec = []

        i = 0
        # step_size * i + ear_v_len is the index of last ear value, should not exceed number of frames
        while step_size * i + ear_v_len <= frames_num:
            start = step_size * i 
            end = step_size * i + ear_v_len
            # print(f"ears len: {len(ears)}, i: {i}, start: {start}, end: {end}")
            ear_vec.append(ears[start:end])
            i = i+1

        columns = []

        for i in range(ear_v_len):
            column_name = "EAR" + str(i + 1)
            columns.append(column_name)

        df = pd.DataFrame(ear_vec, columns=columns)

        df.to_excel("data/ear_vecs_per_user/" + user_nm + "/" + user_nm + "-" + blink_type + ".xlsx", index=False)

        print(f"{user_nm} {blink_type}: Completed creating EAR vectors and saving the resulting data")
