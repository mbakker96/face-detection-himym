import os

import cv2

from PIL import Image
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'names')

y_labels = []
x_train = []
i = 0
labels_ids = {}

recognizer = cv2.face.LBPHFaceRecognizer_create()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in labels_ids:
                labels_ids[label] = i
                i += 1
            id_ = labels_ids[label]
            pil_image = Image.open(path).convert("L")  # turn into gray scale
            image_array = np.array(pil_image, "uint8")

            x_train.append(image_array)
            y_labels.append(id_)

with open("cascade/trained_faces/labels.pickle", 'wb') as f:
    pickle.dump(labels_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("cascade/trained_faces/trainner.yml")
