# Face recognition using vggface2

Face recognition is the general task of identifying and verifying people from photographs of their face.This is a ready to use face recognition code using vggface2.
The VGGFace refers to a series of models developed for face recognition and demonstrated on benchmark computer vision datasets by members of the Visual Geometry Group (VGG) at the University of Oxford.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

```
git clone https://github.com/aaronpp65/face-recognition-vggface2.git
```

### Prerequisites

What things you need to install and how to install them

```
pip install face_recognition
```
```
# Most Recent One (Suggested)
pip install git+https://github.com/rcmalli/keras-vggface.git
# Release Version
pip install keras_vggface
```
```
sudo pip install mtcnn
```
## Usage
### Adding a face to the database


```
from face_recognizer import FaceRecognizer
import cv2

fce=FaceRecognizer()
# path to the image of face
camera = cv2.imread('/home/phi/Neuroplex/face-recognition-vggface2/mals.jpg')
camera = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
# name of the person
fce.add_new_face(camera,'mals')

```
To run 
```
python add_new_face.py
```
### Running the recognizer

```
python main.py
```
### Deleting a face from the database


```
from face_recognizer import FaceRecognizer

fce=FaceRecognizer()
# name of the person to be deleted from the databse
fce.delete_a_face('mals')

```
To run 
```
python delete_face.py
```

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Aaron P P** - *Initial work* - [aaronpp65](https://github.com/aaronpp65)
* **Pranoy R** - *Initial work* - [aaronpp65](https://github.com/pranoyr)


See also the list of [contributors](https://github.com/aaronpp65/face-recognition-vggface2/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [VGGFace2: A dataset for recognising faces across pose and age](https://arxiv.org/abs/1710.08092)
* [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)


