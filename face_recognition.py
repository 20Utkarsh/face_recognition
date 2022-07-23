import numpy as np
import cv2
import sys
import os
sys.path.append(
    'C:\\Users\\devansh raval\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages')

cap = cv2.VideoCapture(0)
datapath = 'C:\\Users\\HP\\Desktop\\data'
face_cascade = cv2.CascadeClassifier(
    "C:\\Users\\HP\\Desktop\\Pyhton\\haarcascade_frontalface_alt.xml")
face_data = []
label = []
class_id = 0
dataset_path = 'C:\\Users\\HP\\Desktop\\data\\'
name = {}
# print(np.load("D:\\haarcascade_\\faces_data\\"+os.listdir(dataset_path)[0]))


def knn(avail_data, query_data, k=5):
    p = []
    for i in range(len(avail_data)):
        dist = np.sqrt(sum((avail_data[i][:-1]-query_data)**2))
        p.append((dist, avail_data[i][-1]))

    p = np.array(sorted(p))
    p = p[:k, :]
    uniq_val_and_count = (np.unique(p[:, 1], return_counts=True))
    # print(p)
    maxvalindx = uniq_val_and_count[1].argmax()
    predicted_val = uniq_val_and_count[0][maxvalindx]
    return predicted_val


for file_name in os.listdir(dataset_path):
    if file_name.endswith('.npy'):
        print("loaded = "+file_name)
        data_item = np.load("C:\\Users\\HP\\Desktop\\data\\"+file_name)
        face_data.append(data_item)
        print("data item")
        print(data_item)
        name[class_id] = file_name[:-4]
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)


print("face data")
print(face_data)
face_dataset = (np.concatenate(face_data, axis=0))
print("face dataser")
print(face_dataset)
face_labels = np.concatenate(label, axis=0).reshape((-1, 1))
train_set = np.concatenate((face_dataset, face_labels), axis=1)
print("train dataser")
print(train_set)


while True:
    ret, frame = cap.read()
    if(ret == False):
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face
        margin = 10
        face_sec = frame[y-margin:y+h+margin, x-margin:x+w+margin]
        try:

            face_sec = cv2.resize(face_sec, (100, 100))
        except Exception as e:
            print(e)
        out = int(knn(train_set, face_sec.flatten()))

        cv2.putText(frame, name[out], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#        cv2.putText(frame,names[int(out)],(x,y-10))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("v_frame", frame)

    # wait for user to input -w then u will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed) == ord('w'):
        break

cap.release()
cv2.destroyAllWindows()
