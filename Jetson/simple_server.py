from time import sleep
from numpysocket import NumpySocket
import logging
import numpy as np

print("-----Init network-----")
reciever = NumpySocket()
sender = NumpySocket()
print("-----Server listen-----")
reciever.bind(('', 13560))
reciever.listen()
sleep(10)

print("-----Connect Client-----")
print("[IP:Port] 10.0.0.30:", 20086)
sender.connect(("10.0.0.30", 20086))

print("-----Loading Models-----")
from detect_imgs import FastFaceDetNet
from pfld_api import PFLDNet
faceNet = FastFaceDetNet()
pfldNet = PFLDNet()

def crop_roi(img, x1, y1, x2, y2, outCropRange=0):
    height_img = img.shape[0]
    width_img = img.shape[1]
    w = x2 - x1
    h = y2 - y1
    delta_w = w*outCropRange
    delta_h = h*outCropRange
    x1 -= delta_w/2
    x2 += delta_w/2
    y1 -= delta_h/2
    y2 += delta_h/2
    y1 = 0 if y1 < 0 else y1
    y2 = height_img if y2 > height_img else y2
    x1 = 0 if x1 < 0 else x1
    x2 = width_img if x2 > width_img else x2
    outputImg = img[int(y1):int(y2), int(x1):int(x2)]
    return outputImg

conn, addr = reciever.accept()

logger = logging.getLogger('Frame reciever.')
logger.setLevel(logging.INFO)
logger.info(f"connected: {addr}")
while True:
    img = conn.recv()
    # print(img)
    logger.info("array received")
    logger.info(img)
    landmarks = []
    if img is None:
        pass
    else:
        boxes, labels, probs = faceNet.run(img)
        if len(boxes) > 0:
            box = boxes[0].numpy()
            org_x = box[0]
            org_y = box[1]
            # print(box)
            # pfldImg = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            pfldImg = crop_roi(img, box[0], box[1], box[2], box[3], outCropRange=0)
            # cv2.imwrite('testout.jpg', pfldImg)
            landmarks_raw = pfldNet.run(pfldImg)
            for (x, y) in landmarks_raw:
                x = x + org_x
                y = y + org_y
                landmarks.append([x, y])
    # Test
    # landmarks = []
    try:
        landmarks = np.array(landmarks)
        print("[SHAPE]", landmarks.shape)
        sender.sendall(landmarks)
        print(conn)
    except:
        continue
reciever.close()
sender.close()



