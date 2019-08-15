from face_recognizer import FaceRecognizer
import cv2

fce=FaceRecognizer()
camera = cv2.imread('/home/phi/Neuroplex/face-recognition-vggface2/mals.jpg')
camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
fce.add_new_face(camera,'mals')