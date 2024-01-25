import cv2
import os
import warnings

# To suppress all warnings
warnings.filterwarnings("ignore")

print(os.getcwd())
# os.chdir("real-time-drowsiness-detection-system/")

# define user and blink type to extract the ear for that category
user_nm = "person1"
blink_type = "short_blink" #switch between short_blink and long_blink
# blink_type = "long_blink" 

# Open the video file
video_path = f'data/videos/{blink_type}-{user_nm}.mov'
cap = cv2.VideoCapture(video_path)

# create the folder to store frames(if not already present)
folder_path = f"data/videos/frames/{user_nm}/{blink_type}"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open the video file at {video_path}")
    exit()

# Read and extract each frame from the video
frame_count = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video is over
    if not ret:
        break

    # Process the frame (you can perform any operations on the frame here)

    # Display the frame (optional)
    # cv2.imshow('Frame', frame)

    # Save the frame to a file (optional)
    frame_filename = f'frame_{frame_count}.jpg'
    cv2.imwrite(f"{folder_path}/{user_nm}_{blink_type}_" + frame_filename, frame)

    # Increment the frame count
    frame_count += 1

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
