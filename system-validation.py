import cv2
import pandas as pd
from datetime import datetime, timedelta
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import warnings
import sys

# To suppress all warnings
warnings.filterwarnings("ignore")


# Load trained models
scaler_filename = "models/feature_scaler_2.joblib"
scaler = joblib.load(scaler_filename)
model_filename = "models/svm_model_2.joblib"
svm_classifier = joblib.load(model_filename)

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def get_ear_from_frame(frame):
    frame = imutils.resize(frame, width=540)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    ear = None
    # find faces in frame
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)#converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        break
    return ear

# function that take the df of ear vectors as input and outputs whether the user is drowsy
def check_for_drowsiness(df):
    continued_long_blink = sum(df["EYE_STATE"].tail(5)) == 10
    if continued_long_blink:
        return 1
    time_now = datetime.now()
    last_minute_start = time_now - timedelta(minutes=1)

    filtered_df = df[df['TIMESTAMP'] > last_minute_start]
    short_blink_count = sum(filtered_df["EYE_STATE"] == 1)
    long_blink_count = sum(filtered_df["EYE_STATE"] == 2)
    total_blink_count = short_blink_count + long_blink_count
    drowsiness_ratio = 0
    if total_blink_count > 1:
        drowsiness_ratio = long_blink_count / (total_blink_count)
    if drowsiness_ratio > 0.25:
        return 2
    return 0


def run_validation_on_video(subject, pvt):
    """
    Inputs: subject id and pvt id
    Output: information on if user was alerted in the video
        0 means video could not be opened
        1 means user alerted
        2 means user not alerted
    """

    # Initialize video capture
    video_path = f'data/DROZY/DROZY/videos_i8/{subject}-{pvt}.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print(f"Error: Couldn't open the video file: {video_path}")
        return 0 # 0 means could not open the file

    # Create an empty DataFrame to store ears and timestamps
    ear_df_columns = ['TIMESTAMP', 'EAR']
    ear_df = pd.DataFrame(columns=ear_df_columns)
    ear_v_len = 15
    step_size = 5

    # i = 0 # temp code

    # df stores the ear vectors that are used for model prediction
    df_ear_columns = []
    for i in range(ear_v_len):
        column_name = "EAR" + str(i + 1)
        df_ear_columns.append(column_name)

    df = pd.DataFrame(columns= df_ear_columns + ["TIMESTAMP", "EYE_STATE"])

    while True:
        ret, frame = cap.read()
        # Break the loop if the video is over
        if not ret:
            break
        timestamp = datetime.now()

        ear = get_ear_from_frame(frame)

        # Handle None EAR since the frame is dark when camera is just turned on
        if ear is not None:
            ear_df = pd.concat([
                ear_df, 
                pd.DataFrame({'TIMESTAMP': [timestamp], 'EAR': [ear]})], 
                ignore_index=True)
            # cv2.putText(frame, f"EAR: {round(ear, 2)}", (10, 650),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        row_num = len(ear_df) - 1
        if row_num + 1 >= ear_v_len and (row_num + 1) % step_size == 0:
            ears_for_vec = ear_df.iloc[row_num - ear_v_len + 1 :row_num + 1, ear_df.columns.isin(["EAR"])]
            time = ear_df.iloc[row_num]["TIMESTAMP"]

            ears_for_vec_2 = ears_for_vec.T.reset_index(drop=True)
            ears_for_vec_2.columns = df_ear_columns

            ears_for_vec_2["TIMESTAMP"] = time
            ears_for_vec_2["TIMESTAMP"] = pd.to_datetime(ears_for_vec_2["TIMESTAMP"])
            ears_for_vec_2_scaled = scaler.transform(ears_for_vec_2[df_ear_columns])
            eye_state = svm_classifier.predict(ears_for_vec_2_scaled)[0]
            ears_for_vec_2["EYE_STATE"] = eye_state
            
            df = pd.concat([
                df, 
                ears_for_vec_2], 
                ignore_index=True) 
            
            drowsiness_ind = check_for_drowsiness(df)
            if drowsiness_ind!= 0:
                print(f"Subject {subject} PVT {pvt} alerted at frame {row_num}")
                cap.release()
                return 1 # 1 means user alerted
    
       
    cap.release()
    return 2 # 2 means user not alerted


# validation_data = [
#     (1, 1), 
#     (2, 1), 
#     (3, 1), 
#     (5, 1), 
#     (6, 1), 
#     (8, 1), 
#     (10, 1), 
#     (12, 1), 
#     (1, 3), 
#     (4, 3), 
#     (5, 3), 
#     (6, 3), 
#     (7, 3), 
#     (8, 3), 
#     (9, 3), 
#     (10, 3), 
#     (11, 3), 
#     (14, 3)]


validation_data = [
    (8, 1), 
    (12, 1),
    (5, 3),
    (10, 3)]

for subject, pvt in validation_data:
    print(f"Starting validation for subject {subject} pvt {pvt}")
    output = run_validation_on_video(subject, pvt)
    if output == 0:
        print("Error")
    if output == 1:
        print(f"Subject {subject} PVT {pvt} user alerted")
    if output == 2:
        print(f"Subject {subject} PVT {pvt} user not alerted")