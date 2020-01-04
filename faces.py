import time
import cv2
import os

from process import getOutputsNames, post_process

CONF_THRESHOLD = 0.99
NMS_THRESHOLD = 0.5

net = cv2.dnn.readNetFromDarknet('cascade/config/yolov3-face.cfg', 'cascade/model-weights/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

for e in range(3):
    cap = cv2.VideoCapture('episodes/How I Met Your Mother s01e0{}.avi'.format(e + 1))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = cv2.VideoWriter('output-{}.avi'.format(e + 1), fourcc, cap.get(cv2.CAP_PROP_FPS), (416, 416))

    i = 1
    while True:
        # Capture frame-by-frame
        try:
            ret, frame = cap.read()
            if not ret:
                print("No frame")
                break
            t1 = time.time()
            frame = cv2.resize(frame, (416, 416))
            original_frame = frame.copy()

            blob = cv2.dnn.blobFromImage(frame, 1 / 255, frame.shape[:2], [0, 0, 0], 1, crop=False)
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))
            faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD, i)
            print("Episode {} - {}% - {}/{}".format(e + 1, round((i / total * 100), 2), i, total))
            print('[i] ==> # detected faces: {}'.format(len(faces)))
            print('#' * 60)
            cv2.imshow("detection", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            video_writer.write(frame)

            i = i + 1
        except:
            print('error')

    cap.release()

# When everything done, release the capture
cv2.destroyAllWindows()
