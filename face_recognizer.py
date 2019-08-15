import pickle as pk
import cv2
from pickle import dump
import os
import face_recognition
import numpy as np
from PIL import Image
from numpy import asarray
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

path = os.path.dirname(os.path.abspath(__file__))

def who_is_it(encodings, database, threshold = 0.5):
		"""
		Arguments:
		encodings -- encodings of faces detected 
		database -- database containing image encodings along with the name of the person on the image

		Returns:
		min_dist -- the minimum distance between image_path encoding and the encodings from the database
		identity -- string, the name prediction for the person on image_path
		"""
		results = []
		scores = []
		
		identity = None

		# db_vectors=np.array(db_vectors)
		for encoding in encodings:
			# Initialize "min_dist" to a large value, say 100 (≈1 line)
			min_dist = 100
			# Loop over the database dictionary's names and encodings.
			for (name, db_enc) in database:
		
				# Compute cosine distance between the target "encoding" and the encoding from the database. (≈ 1 line)
				dist = cosine(encoding,db_enc)

				# If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
				if dist<min_dist:
					min_dist = dist
					identity = name
	
			if min_dist > threshold:
				identity="Unknown"

			results.append(identity)
			scores.append(min_dist)

		return results, scores
# extract faces from a given photograph
def extract_faces(bounding_boxes, img, required_size=(224, 224)):
	"""
		Arguments:
		bounding_boxes -- bounding boxes of detected faces
		img -- the image

		Returns:
		face_array -- array of cropped images
		"""
	
	cropped_images = []

	for bb in bounding_boxes:
		y1,x2,y2,x1 = bb
		# extract the face
		image = img[y1:y2, x1:x2]
		image = cv2.resize(image,(required_size))
		cropped_images.append(image)
	face_array = np.array(cropped_images)
	return face_array


# extract a single face from a given photograph
def extract_face(img, required_size=(224, 224)):
	"""
		Arguments:
		img -- the image

		Returns:
		face_array -- array of cropped images
		"""

	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(img)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = img[y1:y2, x1:x2]
	# resize img to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

class FaceRecognizer:
	""" Class for recognising faces, adding new faces and delete existing faces from the database.
	"""

	def __init__(self):
		# database path
		self.db_path = os.path.join(path, 'database','db_enc.pkl')
		# create a vggface model
		self.model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
		# loading database
		if(os.path.exists(self.db_path)):
			self.database = pk.load(open(self.db_path, "rb"))
		else:
			self.database = []

	def add_new_face(self, img, name):
		""" Adds a new face to the add_new_facedatabase.
		"""
		# extract faces
		faces = extract_face(img)
		faces = np.expand_dims(faces,axis=0)
		# convert into an array of samples
		samples = asarray(faces, 'float32')
		# prepare the face for the model, e.g. center pixels
		samples = preprocess_input(samples, version=2)
		# perform prediction
		yhat = self.model.predict(samples)
		encodings=yhat
		for encoding in encodings:
			self.database.append((name, encoding))
			print("Added {} to the database".format(name))

		# Saving database
		dump(self.database, open(self.db_path, "wb"))
	
	def delete_a_face(self, name):
		""" Deletes an entry from the database.
		"""
		for i,(db_name, db_enc) in enumerate(self.database):
			print(name,db_name)
			if db_name == name:
				self.database.pop(i)
				print("Removed {} from database".format(name))

		# Saving database
		dump(self.database, open(self.db_path, "wb"))

	def get_faces(self, img):
			print("[INFO] recognizing faces...")
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			# detecting faces
			boxes = face_recognition.face_locations(img, model='hog')
			if(boxes):  
				# extract faces
				faces = extract_faces(boxes,img)
				print(np.shape(faces))
				# convert into an array of samples
				samples = asarray(faces, 'float32')
				# prepare the face for the model, e.g. center pixels
				samples = preprocess_input(samples, version=2)
				# perform prediction
				yhat = self.model.predict(samples)
				encodings=yhat
				print("get_face")
				print((encodings))
			
				# Predict output
				names, scores = who_is_it(encodings, self.database, threshold=0.5)
		
				# loop over the recognized faces
				for ((top, right, bottom, left), name) in zip(boxes, names):
					# draw the predicted face name on the image
					cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
					y = top - 15 if top - 15 > 15 else top + 15
					cv2.putText(img, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
						0.75, (0, 255, 0), 2)

			return img
