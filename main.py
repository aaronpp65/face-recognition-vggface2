from face_recognizer import FaceRecognizer
import cv2
    
camera=cv2.VideoCapture(0)
fce=FaceRecognizer()

while True:
    ret,img=camera.read()
    img=fce.get_faces(img)
    draw = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('window',draw)
    if cv2.waitKey(1) == 27: 
        break # esc to quit

