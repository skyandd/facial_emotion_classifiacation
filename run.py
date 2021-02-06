import cv2
import tensorflow as tf
import numpy as np
from copy import deepcopy
import model


def cut_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector.detectMultiScale(rgb_image, 1.3, 5)
    fece_coor = faces[0]
    x, y, w, h = fece_coor
    face = rgb_image[y - 5:y + 5 + h, x - 5:x + 5 + w]
    return rgb_image, face, (x, y, w, h)


def preprocess_image(image, target_size):
    image = tf.image.resize(image, target_size, method='nearest')
    scale_image = model_function.preprocess_input(image)
    return scale_image


def predict_emotion(image):
    return decode_predict[np.argmax(base_model.predict(image[None, ...]), axis=1)[0]]


classes = {'anger': 0,
           'contempt': 1,
           'disgust': 2,
           'fear': 3,
           'happy': 4,
           'neutral': 5,
           'sad': 6,
           'surprise': 7,
           'uncertain': 8}

decode_predict = {j: i for i, j in classes.items()}

metrics = 'accuracy'
model_name = 'EfficientNetB0'
num_classes = 9
image_size = (224, 224)
freezing = False

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
base_model, model_function = model.get_model(model_name,
                                             image_size,
                                             num_classes,
                                             freezing=False)

base_model.load_weights('EfficientNetB0.weights.09-1.0309-1.3092.hdf5')
cam = cv2.VideoCapture(0)
assert cam.isOpened(), 'Камера не запущена'

while True:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector.detectMultiScale(frame, 1.3, 5)
    
    if len(faces) != 0:
        x, y, w, h = faces[0]
        face = frame[y - 5:y + 5 + h, x - 5:x + 5 + w]

        face = preprocess_image(face, image_size)
        emotion = predict_emotion(face)
        
        frame = cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 1)
        frame = cv2.putText(frame,
                            emotion,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("facial emotion recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# if __name__=='__main__':
