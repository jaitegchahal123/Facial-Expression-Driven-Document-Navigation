import numpy as np
import cv2
import math
# Controls the mouse and keyboard without the user having to physically interact with it
import pyautogui
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import cv2
import time

#A funtion to clean up how to find the indices of landmarks that we need. 
# For example, if we need the left-eye landmarks (on the diagrma, 32,33...), this would tell us where of the 68 length array those landmarks would be found
def get_facial_landmark_ID(body_part):
    (start_point, end_point) = face_utils.FACIAL_LANDMARKS_IDXS[body_part]
    return (start_point, end_point)

#Calculate EAR - eye aspect ratio
    #The higher the ear, the more open the eye. Hence, we can set a threshold to determine whether or not the eye is blinking.
def calculate_ear(eye):
    #Given the array of points of a singular eye, we can calculate the ear of each eye
    point_2_minus_6 = dist.euclidean(eye[1], eye[5])
    point_3_minus_5 = dist.euclidean(eye[2], eye[4])
    point_1_minus_4 = dist.euclidean(eye[0], eye[3])
    #We need to multiply point 1 - point 4 by 2 to get a scaled average (because we have 2 subtractions in the numerator)
    return (point_2_minus_6 + point_3_minus_5)/(2*point_1_minus_4)




# 0 is the default camera, can use others if need
video = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

blink_counter = 0
ptime  =0 

while video.isOpened():

    

    # Ret is a boolean variable that returns true if the frame is available
    # Frame is the image array vector specified on the default frames per second

    ret, frame = video.read()
    #Make the frame smaller, so it's less laggy
    frame = imutils.resize(frame,width=640)

    #In order to pass something into face_cascade classifier, needs to be grayscaled
    grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #The classifier was trained on differnet sizes of images. It doesn't the dimensions of the CURRENT image/video
        #Hence, we give it a scale factor to tell it how much we should shrink the image/video per iteration. (1.05 => shrink image by 5%)
            #Higher the number, the more accurate, but way slower
    scale_factor = 1.2

    #Min neighbors is essentially the "accuracy" of your classifier. Basically saying how many rectangles do I need on a face for it to be considered a face.
        #The way the model works, is it creates a bunch of different rectangles over different features that pertain to 'a face', then it converges into 1 depending on min_neighbors
            #Higher value => More accurate faces/results => less detections
    min_neighbors = 5
    # Min size = the minimum size of the rectangle on the face
    min_size = [30, 30] 

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    #print(fps)
    # length = frames = video.get(cv2.CAP_PROP_FRAME_COUNT) #int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # print( length )
    #cv2.putText(frame, "FPS: " + str(fps), (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    
    #Also, a max size, but we don't need to set it manually
    #Faces will give us rectangles. More specifically, the x,y coordinates and the height/width
    #faces = face_cascade.detectMultiScale(grayed_frame, scale_factor, min_neighbors, minSize=(200, 200))
    faces = detector(grayed_frame)
    for face in faces:
        #Now we need to actually draw the rectangle around each rectangle it gives us
        #rectangle = cv2.rectangle(frame, start_point, end_point, color, thickness)
        #print(x, y)
        #print(face)
        color_blue = (255, 0, 0)
        color_red = (0, 0, 255)
        color_turq = (200, 200, 0)
        x = face.left()#face[0]
        y= face.top()#face[1]
        w= face.right()#face[2]
        h = face.bottom()#face[3]
        #center = (x + w//2, y + h//2)
        #frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        cv2.rectangle(frame, (x,y), (w, h), color_blue, 2)
        #now we need to find the area of just the face, becuase we have to ONLY pass the face into the eye classifier, so that it can classify eyes given face
        # roi = region of interest
        #Also, in the faces array, we index by y and then x
        #Supposed to be more accurate, to convert to grayscale, but seems slower and actually worse
        grayed_roi = grayed_frame[ y:h, x:w]
        colored_roi = frame[y:y+h, x:x+w]
        # dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # print(x,y,w,h)
        # print(dlib_rect)
        #This landmark will return an OBJECT: a landmark object, we need to convert it into an array to actually access each of the 68 points
            #Using the regular frame, instead of roi, because it's more effieicnet
        landmarks = predictor(grayed_frame, face)
        landmark_array = face_utils.shape_to_np(landmarks)
        left_eye_indices = landmark_array[get_facial_landmark_ID("left_eye")[0]:get_facial_landmark_ID("left_eye")[1]]
        right_eye_indices = landmark_array[get_facial_landmark_ID("right_eye")[0]:get_facial_landmark_ID("right_eye")[1]]
        #mouth_indices = landmark_array[get_facial_landmark_ID("mouth")[0]:get_facial_landmark_ID("mouth")[1]]
        # inner_mouth_indices = landmark_array[get_facial_landmark_ID("inner_mouth")[0]:get_facial_landmark_ID("inner_mouth")[1]]
        left_eyebrow_indices = landmark_array[get_facial_landmark_ID("left_eyebrow")[0]:get_facial_landmark_ID("left_eyebrow")[1]]
        right_eyebrow_indices = landmark_array[get_facial_landmark_ID("right_eyebrow")[0]:get_facial_landmark_ID("right_eyebrow")[1]]
        # nose_indices = landmark_array[get_facial_landmark_ID("nose")[0]:get_facial_landmark_ID("nose")[1]]
        # jaw_indices = landmark_array[get_facial_landmark_ID("jaw")[0]:get_facial_landmark_ID("jaw")[1]]

        color_green = (0, 255, 0)
        color_pink = (203, 192, 255)

        #From what I see, 0.2 seems to be a good threshold
        #print(calculate_ear(left_eye_indices))
        #print(calculate_ear(right_eye_indices))
        eyes_open = True
        frames_eyes_closed = 0
        
        
        ear = (calculate_ear(right_eye_indices)+ calculate_ear(left_eye_indices)) / 2.0
        #eyes_closed = False
        #print(ear)
        if(ear < 0.2):
            #blink
            eyes_open = False
            cv2.putText(frame, "BLINK DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #cv2.waitKey(50)
            blink_counter+=1

            #frames_eyes_closed+=1
            #print(frames_eyes_closed)
            #print(fps*3)
            if(frames_eyes_closed >= fps):
                pyautogui.press('right')
                break;
            #print(frames_eyes_closed)
            #pyautogui.click()
#            pyautogui.press("right")



            #eyes_closed = True
            #pass;
        else:#if (ear >= 0.2):
            #open
            eyes_open = True
            cv2.putText(frame, "EYES OPEN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
           # eyes_closed = False

            #continue;
            #pass;
        # else:
        #     #wink
        #     cv2.putText(frame, "WINK DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (33, 123, 232), 2)
            #eyes_closed = False
        
        cv2.putText(frame, "Total Blinks: " + str(blink_counter), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "EAR: " + str(round(ear, 1)), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        





        #FACIAL_LANDMARKS_68_IDXS = OrderedDict([(“mouth”, (48, 68)), 
        # (“inner_mouth”, (60, 68)), (“right_eyebrow”, (17, 22)),
        # (“left_eyebrow”, (22, 27)), (“right_eye”, (36, 42)),
        #  (“left_eye”, (42, 48)),(“nose”, (27, 36)), (“jaw”, (0, 17))])
        
        for pts1, pts2, pts3, pts4 in zip(left_eye_indices, right_eye_indices, left_eyebrow_indices, right_eyebrow_indices):
            #cv2.rectangle(frame, (xpt, ypt), (xpt, ypt), color_green, 3)
            cv2.circle(frame, (pts1), 1, color_green)
            cv2.circle(frame, (pts2), 1, color_green)
            cv2.circle(frame, (pts3), 1, color_turq)
            cv2.circle(frame, (pts4), 1, color_turq)

            # print("Left: " + calculate_ear(pts1))
            # print("Right: " + calculate_ear(pts2))

        





        # landmarks = predictor(grayed_roi, dlib_rect)
        # left_point = (landmarks.part(36).x, landmarks.part(36).y)
        # right_point = (landmarks.part(39).x, landmarks.part(39).y)
        # center_top = midpoint(landmarks.part(37), landmarks.part(38))
        # center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        #cv2.imshow("roi", grayed_roi)

        #Remember minSize doesn't effect the actual values of x2, y2 just sets a lower bound to filter
        # eyes = eye_cascade.detectMultiScale(grayed_roi, scale_factor, min_neighbors, minSize=(100, 100))
        # #print(eyes)

        # for(x2, y2, w2, h2) in eyes:
        #     #(x2, y2)
        #     color_red = (0, 0, 255)
        #     cv2.rectangle(colored_roi, (x2,y2), (x2+w2, y2+h2), color_red, 5)



    cv2.imshow("frame", frame)
                                                                                                                    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

#    # crop_image = frame[100:300, 100:300]
    
#     # Top left of ractangle
#     start_point = (100,100)

#     # Bottom right of rectangle
#     end_point = (300, 300)

#     cropped_rectangle = frame[100:300, 100:300]
#     color = (255, 0, 0)
#     thickness = 4ter
#     rectangle = cv2.rectangle(frame, start_point, end_point, color, thickness)
#     cv2.imshow("rectangle", rectangle)
    

#    Using delib
# import cv2
# import numpy as np
# import dlib

# cap = cv2.VideoCapture(0)

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# def midpoint(p1 ,p2):
#     return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = detector(gray)
#     for face in faces:
#         x, y = face.left(), face.top()
#         x1, y1 = face.right(), face.bottom()
#         cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
#         landmarks = predictor(gray, face)
#         left_point = (landmarks.part(36).x, landmarks.part(36).y)
#         right_point = (landmarks.part(39).x, landmarks.part(39).y)
#         center_top = midpoint(landmarks.part(37), landmarks.part(38))
#         center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

#         hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
#         ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
#         cv2.imshow("Frame", frame)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()