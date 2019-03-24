import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
fullbody_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_fullbody.xml')
profile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    #cap=cv2.VideoCapture(0)
    #while True:
    #    _,frame=cap.read()
    #    found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
    #    draw_detections(frame,found)
    #    cv2.imshow('feed',frame)
    #    if cv2.waitKey(20) & 0xFF == ord('q'):
    #        break
    #cv2.destroyAllWindows()


# Load in pickles form faces-train dump
labels = {"person_name:", 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
         #TODO: Fix recognizer for people that arent me
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "STRANGER"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Eye cascade - ONLY OTHER WORKING CASCADE... IDK WHY MAN
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0, 255, 0),2)

        #fullbody = fullbody_cascade.detectMultiScale(roi_gray)
        #for (fx, fy, fw, fh) in fullbody:
        #    cv2.rectangle(roi_color,(fx,fy),(fx+fw,fy+fh),(0, 255, 0),2)

        profile = profile_cascade.detectMultiScale(roi_gray)
        for (x, y, w, h) in profile:
            cv2.rectangle(roi_color,(x,y),(x+w,y+h),(255, 255, 0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
