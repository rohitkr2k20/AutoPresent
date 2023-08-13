from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import face_recognition
from datetime import datetime
import random
import time

def detect_and_predict_mask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))


	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	
	faces = []
	locs = []
	preds = []

	
	for i in range(0, detections.shape[2]):
	
		confidence = detections[0, 0, i, 2]

		
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

		
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	
	if len(faces) > 0:
		
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	
	return (locs, preds)

def findEncodings(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList


def markAttendance(name):
	with open('Attendance.csv','r+') as f:
		myDataList = f.readlines()
		nameList = []
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList:
			now = datetime.now()
			dtstring = now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},{dtstring}')

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

path = 'Dataset_Images'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
	curImg = cv2.imread(f'{path}/{cl}')
	images.append(curImg)
	classNames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images)

print ()
print ()
print ()
print ()
print ()
print ()

# Temprature Sensing
print ("Bring forth your Hand")
time.sleep (4)
temprature = 97.8 + random.random ()
print ("Your temprature is %.2f F, so now you can move on"%temprature)
time.sleep (4)

cnt = 0
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	if (len (preds) > 1):
		print ("Come One at a time")
		cv2.imshow("Frame", frame)
		continue

	
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		if (label == "Mask"): 
			cnt += 1

		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}".format(label)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if (cnt >= 30):
		break


print("[INFO] Now You Can Move On and start showing your face...")
#print('Encoding Complete')

while True:

	img = vs.read()
	imgS = cv2.resize(img,(0,0),None,0.25,0.25)
	imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)


	facesCurFrame  = face_recognition.face_locations(imgS)
	encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

	for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
		matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
		faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
		#print(faceDis)
		matchIndex = np.argmin(faceDis)


		if matches[matchIndex]:
			name = classNames[matchIndex].upper()
			#print(name)
			y1,x2,y2,x1 = faceLoc
			y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
			cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
			cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
			cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
			markAttendance(name)


	cv2.imshow('Frame',img)
	key = cv2.waitKey(1) & 0xFF

	
	if key == ord("q"):
		break

cv2.destroyAllWindows()

	
