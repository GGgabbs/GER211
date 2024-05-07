#
# Filename:     Video_Processor5.py
# Created by:   Gabrielle Ross
# Last updated: 5/7/2024
# Github:       https://github.com/GGgabbs/GER211
#
# GER211:       A system for breaking down and simplifying
#               video processing for media theory classes 
#               via individual video frame extraction
#
# Purpose:      Tracks the infintesmile movements of a
#               user's eyes to unveil the subconscious
#               biases in a witness' affective state



# Modules
import cv2
import numpy as np
import os.path
import sys
import time
import matplotlib
from matplotlib import pyplot as plot
import matplotlib.animation as animation



# Private functions
def detect_faces(img): # Face detection
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray images are easier to conduct image data analysis on
    
    # Face classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    coords = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    frame = None
    x, y = 0, 0
    
    if len(coords) > 1: # Making sure there is only 1 face
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None, [0, 0]
    
    for (x, y, w, h) in biggest:
        frame = img[y : y + h, x : x + w] # Frame is the image data of the face
    
    return frame, [x, y] # Also passing in original top left (x, y) points to find frame positioning from original image


def detect_eyes(img): # Eye detection
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray images are easier to conduct image data analysis on
    
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5) # Detect eyes
    
    # Face frame
    width = np.size(img, 1)
    height = np.size(img, 0)
    
    # Setting to None if no eyes are found
    left_eye = None
    right_eye = None
    x1, y1, x2, y2 = 0, 0, 0, 0
    
    for (x, y, w, h) in eyes:
        if y > height / 2: # Eyes are typically the top half of the face
            pass
        
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5: # Each eye is on each left and right side of the face split down the middle
            left_eye = img[y : y + h, x : x + w]
            x1, y1 = x, y
        else:
            right_eye = img[y : y + h, x : x + w]
            x2, y2 = x, y
            
    
    return left_eye, right_eye, [x1, y1, x2, y2] # (x1, y1) is top left corner of left eye, (x2, y2) is top left corner of right eye


def cut_eyebrows(img): # Removing the eyebrows from the eye frame
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4) # Eyebrows can safely be assumed to be 1/4 the height of the eye frame
    img = img[eyebrow_h : height, 0 : width] # Eye frames without eyebrows
    
    return img


def blob_process(img, threshold, detector): # Cleaning the rest of the eye and detecting the pure pupil eye blob
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    
    # Removing noise to get a pure blob
    img = cv2.erode(img, None, iterations = 2)
    img = cv2.dilate(img, None, iterations = 4)
    img = cv2.medianBlur(img, 5)
    
    keypoints = detector.detect(img) # Largest blob
    
    return keypoints


def eye_detection_analysis(img, eye, eye_pos, detector):
    eye_center_FINAL_x, eye_center_FINAL_y = None, None
    if eye is not None:
        threshold = cv2.getTrackbarPos("threshold", "image2")
        #cv2.rectangle(img, (eye_pos[0], eye_pos[1]), ((eye_pos[0] + len(eye[0])), (eye_pos[1] + len(eye))), (255, 255, 0), 2) # Image, (face top left xy), (face bottom right xy), (border color), (border width)
        
        
        # Cutting out the eyebrows from the frame
        eye = cut_eyebrows(eye)
        
        # Bounds of eye frame without eyebrow
        x1, x2 = eye_pos[0], (eye_pos[0] + len(eye[0]))
        #x1, x2 = eye_pos[0] + 10, (eye_pos[0] + len(eye[0])) - 10
        #x1, x2 = eye_pos[0] + (((eye_pos[0] + len(eye[0])) - eye_pos[0]) / 2), (eye_pos[0] + len(eye[0]) - (((eye_pos[0] + len(eye[0])) - eye_pos[0]) / 2))
        y1, y2 = (eye_pos[1] + int(len(eye) / 4)), (eye_pos[1] + len(eye) + int(len(eye) / 4))
        
        # Making new frame same aspect ratio as window "image2"
        center_eye_frame_y = int((y2 - y1) / 2)
        eye_frame_width = x2 - x1
        #eye_frame_width = (x2 - x1) / 2
        _, _, window_width, window_height = cv2.getWindowImageRect("image2")
        eye_frame_height = int((eye_frame_width * window_height) / window_width)
        
        
        # Final pupil blob outline
        keypoints = blob_process(eye, threshold, detector)
        eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        if keypoints != (): # Only draw the frame if the pupil is found
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2) # Eye frame without eyebrow. Image, (top left xy), (bottom right xy), (border color), (border width)
            
            y1, y2 = int(eye_pos[1] + int(len(eye) / 4) + center_eye_frame_y - (eye_frame_height / 2)), int(eye_pos[1] + int(len(eye) / 4) + center_eye_frame_y + (eye_frame_height / 2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2) # Eye frame with corrected aspect ratio to movie. Image, (top left xy), (bottom right xy), (border color), (border width)
            #cv2.circle(img, (int(x1 + keypoints[0].pt[0]), int(y1 + keypoints[0].pt[1])), 10, (0, 255, 255), 3) # Circle around center of found pupil
            #eye_center_FINAL_x, eye_center_FINAL_y = (keypoints[0].pt[0] * window_width / eye_frame_width), (keypoints[0].pt[1] * window_height / (y2 - y1))
            eye_center_FINAL_x, eye_center_FINAL_y = (keypoints[0].pt[0] * window_width / (x2 - x1)), (keypoints[0].pt[1] * window_height / (y2 - y1))
            print(keypoints[0].pt[0], keypoints[0].pt[1], eye_center_FINAL_x, eye_center_FINAL_y)
     
    return eye_center_FINAL_x, eye_center_FINAL_y
      

def nothing(x): # Sliders require track bar movement functions
    pass


def extract_frames(movie_name, movie_path): # This is the slow way but processes correctly. Processes the entire movie file and saves the frames individually
    video = cv2.VideoCapture(movie_path)
    
    print("Extracting frames...this may take a few minutes...")
    current_frame = 0
    while(True):
        success, frame = video.read() 
    
        if success: # If more vieo is still left to process
            cv2.imwrite(("data/" + movie_name.split(".")[0] + "/Frame" + str(current_frame) + ".png"), frame)
            current_frame += 1 # Increasing frame count
        else: 
            break
    
    print("Saved movie frames!")


def plot_eye_map(eyes_position_array):
    figure, ax = plot.subplots()
    
    line1 = ax.plot(eyes_position_array["x"][0], eyes_position_array["y"][0], c = "b")[0]
    ax.set(xlim=[min(eyes_position_array["x"]), max(eyes_position_array["x"])], ylim=[min(eyes_position_array["y"]), max(eyes_position_array["y"])], xlabel='Time [s]', ylabel='Z [m]')
    
    def update(frame):
        line1.set_xdata(eyes_position_array["x"][:frame])
        line1.set_ydata(eyes_position_array["y"][:frame])
        return line1
    
    ani = animation.FuncAnimation(fig = figure, func = update, frames = len(eyes_position_array["x"]), interval = 30)
    plot.show()


def plot_eye_map_on_video(movie_name, frame_counter, eyes_position_array):
    # Setting up the the movie window
    cv2.namedWindow("image3", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("image3", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Get initial frame
    video_frame_name = "data/" + movie_name + "/Frame" + str(frame_counter) + ".png"
    if os.path.isfile(video_frame_name):
        img = cv2.imread(video_frame_name)
    
    
    # MAIN LOOP
    while True:
        video_frame_name2 = "data/" + movie_name + "/Frame" + str(frame_counter) + ".png"
        
        if os.path.isfile(video_frame_name2): # If at the end of the video, end it
            img = cv2.imread(video_frame_name2)
            frame_counter += 1 # Increases next frame
            
            pixel_square_length = 50
            if frame_counter in eyes_position_array["x"].keys():
                img[int(eyes_position_array["y"][frame_counter] - (pixel_square_length / 2)) : int(eyes_position_array["y"][frame_counter] + (pixel_square_length / 2)), int(eyes_position_array["x"][frame_counter] - (pixel_square_length / 2)) : int(eyes_position_array["x"][frame_counter] + (pixel_square_length / 2))] = [255, 0, 0]
            else: # Getting average of last accessible closest frame counter within the position array and the next one
                if frame_counter > max(eyes_position_array["x"].keys()):
                    break
            #    img[int(eyes_position_array["y"][frame_counter] - (pixel_square_length / 2)) : int(eyes_position_array["y"][frame_counter] + (pixel_square_length / 2)), int(eyes_position_array["x"][frame_counter] - (pixel_square_length / 2)) : int(eyes_position_array["x"][frame_counter] + (pixel_square_length / 2))] = [255, 0, 0]
        
            cv2.imshow("image3", img)
            time.sleep(0.04166666666) # 24 fps
        else: # Want to only break when the entire screen is at the video end
            print("Video end!")
            break
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Refreshing and keeping everything in check
    cv2.destroyAllWindows()


def webcam_setup(movie_name, frame_counter): # MAIN CODE FUNCTIONALITY
    old_frame_counter = frame_counter
    
    
    # Setting up the webcam window
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("WebCam", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("WebCam", cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow("WebCam", 0, 50)
    _, _, webcam_window_width, webcam_window_height = cv2.getWindowImageRect("WebCam")
    webcam_set_height = 300
    cv2.resizeWindow("WebCam", int((webcam_set_height * webcam_window_width) / webcam_window_height), webcam_set_height)
    
    # Setting up the the movie window
    cv2.namedWindow("image2", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("image2", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.createTrackbar("threshold", "image2", 0, 255, nothing)
    
    
    # Setting up the detector
    detector_params = cv2.SimpleBlobDetector_Params()
    detector_params.filterByArea = True
    detector_params.maxArea = 1500
    detector = cv2.SimpleBlobDetector_create(detector_params)
    
    
    # Get initial frame
    video_frame_name = "data/" + movie_name + "/Frame" + str(frame_counter) + ".png"
    if os.path.isfile(video_frame_name):
        img = cv2.imread(video_frame_name)
    
    
    #eyes_position_array = {"x": [], "y": []}
    eyes_position_array = {"x": {}, "y": {}}
    
    
    # MAIN LOOP
    while True:
        _, frame = cap.read()
        
        face, face_pos = detect_faces(frame)
        if face is not None:
            #cv2.rectangle(frame, (face_pos[0], face_pos[1]), ((face_pos[0] + len(face[0])), (face_pos[1] + len(face))), (255, 255, 0), 2) # Face frame. Image, (face top left xy), (face bottom right xy), (border color), (border width)
            
            left_eye, right_eye, eye_pos = detect_eyes(face)
            eye_center_FINAL_x, eye_center_FINAL_y = eye_detection_analysis(frame, left_eye, np.array([eye_pos[0], eye_pos[1]]) + np.array(face_pos), detector)
            
            
            #pixel_square_length = 50
            if eye_center_FINAL_x is not None:
                eyes_position_array["x"][frame_counter] = eye_center_FINAL_x
                eyes_position_array["y"][frame_counter] = eye_center_FINAL_y
        
        video_frame_name2 = "data/" + movie_name + "/Frame" + str(frame_counter) + ".png"
        if os.path.isfile(video_frame_name2): # If at the end of the video, end it
            img2 = cv2.imread(video_frame_name2)
            frame_counter += 1 # Increases next frame
        else: # Want to only break when the entire screen is at the video end
            print("Video end!")
            break
        
        
        cv2.imshow("WebCam", frame)
        cv2.imshow("image2", img2)
        
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Refreshing and keeping everything in check
    cap.release()
    cv2.destroyAllWindows()
    
    
    plot_eye_map_on_video(movie_name, old_frame_counter, eyes_position_array)


def setup(movie_name, movie_path, starting_frame): # Running everything together
    # Fixing user input
    if not os.path.exists(movie_name): # Making sure movie file exists
        print("Invalid movie name inputted. Please input a movie or video name that exists.")
        sys.exit()
    
    if movie_path != "": # If the movie is from another path check the movie path to the file is valid
        if not os.path.isfile(movie_path):
            print("Invalid movie path inputted. Please input a movie or video path that exists.")
            sys.exit()
    else:
        movie_path = movie_name
    
    # If the starting frame is a valid integer
    if starting_frame != "":
        try:
            starting_frame = int(starting_frame)
        except ValueError:
            print("Invalid starting frame inputted. Please input a integer less than the number of frames in the video.")
            sys.exit()
    else:
        starting_frame = 0
    
    
    # Make the folder directories for the movie names
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/" + movie_name.split(".")[0]): # Movie has never been analyzed before
        os.makedirs("data/" + movie_name.split(".")[0])
        extract_frames(movie_name, movie_path)
    
    
    # Getting the starting frame
    number_of_frames = len(os.listdir("data/" + movie_name.split(".")[0]))
    print("There are " + str(number_of_frames) + " frames in video " + movie_name)
    
    frame_counter = 0
    if starting_frame != 0:
        if starting_frame < number_of_frames: # User inputted bad number
            frame_counter = starting_frame
        else:
            print("Invalid starting frame inputted. Please input a frame that exists.")
            sys.exit()
        
    
    webcam_setup(movie_name.split(".")[0], frame_counter)




setup(input("Movie name (including file extension): "), input("Movie file path (Optional): "), input("Starting frame (Optional): "))