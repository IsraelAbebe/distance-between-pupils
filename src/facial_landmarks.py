from imutils import face_utils
from pupil_distance import *
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
from card_detect import get_card_width
from scipy.spatial import distance
# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True,
                help='path to facial landmark predictor')
ap.add_argument('-i', '--image', required=True,
                help='path to input image')
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")

# load the image, resize it, and conver it to grayscale
image = cv2.imread(args['image'])
# image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
  # determine the facial landmarks for the face region, then
  # convert the facial landmark (x, y)-coordinates to a NumPy
  # array
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)

  # convert dlib's retangle to a OpenCV-style bounding box
  # [i.e., (x, y, w, h)], then draw the face bounding box
  (x, y, w, h) = face_utils.rect_to_bb(rect)
  cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 2)

  # show the face number
  cv2.putText(image, 'Face #{}'.format(i + 1), (x - 10, y - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  # loop over the (x, y)-coordinates for the facial landmarks
  # and draw them on the image
  for (j, (x, y)) in enumerate(shape):
    if(j in [37, 38, 40, 41, 43, 44, 46, 47]):
      cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
    else:
      cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

  #-----------------------------------------------

  # length of left eyelash
  left_lash_pixel = [23,24,25,26,27]
  left_lash_dist = 0
  for i in range(len(left_lash_pixel)-1):
    left_lash_dist += np.linalg.norm(shape[i+1]-shape[i])

  
  # length of right eyelash
  right_lash_pixel = [18,19,20,21,22]
  right_lash_dist = 0
  for i in range(len(right_lash_pixel)-1):
    right_lash_dist += np.linalg.norm(shape[i+1]-shape[i])

  
  # length of jaw
  jaw_pixel = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
  jaw_dist = 0
  for i in range(len(jaw_pixel)-1):
    jaw_dist += np.linalg.norm(shape[i+1]-shape[i])

  # length of nose
  nose_pixel = [28,29,30,31,34]
  nose_dist = 0
  for i in range(len(nose_pixel)-1):
    nose_dist += np.linalg.norm(shape[i+1]-shape[i])

  # length of nose
  upper_lip_pixel = [49,50,51,52,53,54,55]
  upper_lip_dist = 0
  for i in range(len(upper_lip_pixel)-1):
    upper_lip_dist += np.linalg.norm(shape[i+1]-shape[i])

  # length of face end to end
  face_pixel = [1,17]
  face_dist = 0
  for i in range(len(face_pixel)-1):
    face_dist += np.linalg.norm(shape[i+1]-shape[i])

  

  # print('--------------left_lash_dist',left_lash_dist)
  # print('--------------right_lash_dist',right_lash_dist)
  # print('--------------jaw_dist',jaw_dist)
  # print('--------------nose_dist',nose_dist)
  # print('--------------upper_lip_dist',upper_lip_dist)
  # print('--------------face_pixel',face_dist)



  #-----------------------------------------------

  # estimating pupil center from known points 
  left_pupil = Pupil(shape[37], shape[40],
      shape[41], shape[38]).central_point

  right_pupil = Pupil(shape[43], shape[46],
      shape[47],shape[44]).central_point
  
  # drawing the lines to find the pupil
  for ((a, b), (c, d)) in [(shape[37], shape[40]), (shape[38], shape[41]), (shape[43], shape[46]), (shape[47], shape[44])]:
    cv2.line(image,
            (a, b),
            (c, d),
            (255,0,0), 1)

  # drawing central points and showing informations
  cv2.circle(image, (int(left_pupil.x), int(left_pupil.y)), 1, (0, 0, 255), -1)
  cv2.circle(image, (int(right_pupil.x), int(right_pupil.y)), 1, (0, 0, 255), -1)

  cv2.line(image,
          (int(left_pupil.x), int(left_pupil.y)),
          (int(right_pupil.x), int(right_pupil.y)),
          (0,0,255), 1)


  pupil_distance = Pupil.distance(left_pupil, right_pupil)
  

  card_coordinate = [768.69446,2121.0703,1420.283,2538.185]
  # card_coordinate = get_card_width(im) #[768.69446,2121.0703,1420.283,2538.185]

  x = card_coordinate[2]-card_coordinate[0]
  y = card_coordinate[3]-card_coordinate[1]

  if x > y:
    width = x
  else:
    width = y

  # width = card_coordinate[2]

  #---------------------------------------MESUREMENT OUTPUT

  real_mesurement = (pupil_distance * 3.375)/width
  REAL_left_lash_dist = (left_lash_dist *3.375)/width
  REAL_right_lash_dist = (right_lash_dist *3.375)/width
  REAL_jaw_dist = (jaw_dist *3.375)/width
  REAL_nose_dist = (nose_dist *3.375)/width
  REAL_upper_lip_dist = (upper_lip_dist *3.375)/width
  REAL_left_lash_dist = (left_lash_dist *3.375)/width
  REAL_face_dist = (face_dist *3.375)/width
  




  print('LEFT LASH',REAL_left_lash_dist)
  print('RIGHT LASH',REAL_right_lash_dist)
  print('REAL JAW',REAL_jaw_dist)
  print('REAL NOSE',REAL_nose_dist)
  print('REAL UPPER LIP ',REAL_upper_lip_dist)
  print('REAL FACE',REAL_face_dist)




  cv2.putText(image, 'distance: {:.2f} inch'.format(real_mesurement), (int(left_pupil.x), int(left_pupil.y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

# show the output image with the face detection + facial landmarks
# cv2.imshow('Output', image)
# cv2.waitKey(0)

cv2.imwrite('output/{}.jpg'.format(time.time()), image)
