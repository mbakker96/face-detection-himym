import pickle
import numpy as np
import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("cascade/trained_faces/trainner.yml")

labels = {}
with open("cascade/trained_faces/labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labels = {v: k for k, v, in labels.items()}


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def draw_predict(frame, conf, left, top, right, bottom, name):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

    text = name + " " + str(round(conf))

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

    top = max(top, label_size[1])
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1)


def post_process(frame, outs, conf_threshold, nms_threshold, count):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    x = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        id_, conf = recognizer.predict(frame_gray[top: top + height, left:left + width])
        # if conf <= 150:
        #     print(id_, labels[id_], conf)
        #     filename = 'result/{}/{}.png'.format(labels[id_], count)
        name = labels[id_]
        # else:
        #     filename = 'faces/{}-{}.png'.format(count, x)
        #     name = "unknown"
        #     x += 1
        # cv2.imwrite(filename, frame[top: top + height, left:left + width])
        box.append(confidences[i])
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height)
        draw_predict(frame, conf, left, top, right, bottom, name)
    return final_boxes


def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom
