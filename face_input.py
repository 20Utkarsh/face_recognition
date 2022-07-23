import numpy as np
import cv2
import sys
sys.path.append(
    'C:\\Users\\devansh raval\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages')


# initialize camera
cap = cv2.VideoCapture(0)
datapath = "C:\\Users\\HP\\Desktop\\data\\"


face_cascade = cv2.CascadeClassifier(
    "C:\\Users\\HP\\Desktop\\Pyhton\\haarcascade_frontalface_alt.xml")
face_data = []
skip = 0

file_name = input("Enter the name of the person = ")


while True:
    ret, frame = cap.read()
    g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(ret == False):
        continue

        # this will retrn list of faces
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    faces = sorted(faces, key=lambda f: f[2]*f[3])
    face_sec = 0
    for (x, y, w, h) in faces[-1:]:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # extract region of interest
        margin = 10
        face_sec = frame[y-margin:y+h+margin, x-margin:x+w+margin]
        face_sec = cv2.resize(face_sec, (100, 100))

        if skip % 10 == 0:
            face_data.append(face_sec)
            print(len(face_data))
        skip += 1

    cv2.imshow("croped", face_sec)
    cv2.imshow("v_frame", frame)

    # wait for user to input -w then u will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed) == ord('w'):
        break


face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(datapath+file_name+".npy", face_data)
print("data saved at"+datapath+file_name)

cap.release()
cv2.destroyAllWindows()
