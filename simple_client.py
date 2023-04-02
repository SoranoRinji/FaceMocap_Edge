import time
from numpysocket import NumpySocket
import cv2
import logging

cap = cv2.VideoCapture(0)
reciever = NumpySocket()
sender = NumpySocket()
reciever.bind(('', 20086))
reciever.listen()
sender.connect(("10.0.0.70", 13560))

conn, addr = reciever.accept()

# logger = logging.getLogger('Landmarks reciever.')
# logger.setLevel(logging.INFO)

while(cap.isOpened()):
    ret, frame = cap.read()

    # frame_resize = frame[::2, ::2]
    if ret is True:
        img = frame.copy()
        imgOut = frame.copy()
        try:
            sender.sendall(img)
            start = time.time()
            # logger.info(f"connected: {addr}")
            landmarks = conn.recv()
            end = time.time()
            print(f"[FPS]: {1/(end-start)}")
            # print(type(landmarks))
            # logger.info("array received")
            # logger.info(landmarks)
            if len(landmarks) > 0:
                for (x, y) in landmarks:
                    cv2.circle(imgOut, (int(x), int(y)), 2, (0, 0, 255), -1)
            print("showing image")
            cv2.imshow('annotated', imgOut)
            cv2.waitKey(1)
        except:
            continue

    else:
        break

reciever.close()
sender.close()
conn.close()
cap.release()
cv2.destroyAllWindows()


