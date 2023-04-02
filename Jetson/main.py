from detect_imgs import FastFaceDetNet
from pfld_api import PFLDNet

import cv2

print("-----Loading Models-----")
faceNet = FastFaceDetNet()
pfldNet = PFLDNet()

print("-----Initializing WebCam-----")
print("You can check camera status led light to comfirm its status.")
cap = cv2.VideoCapture(1)

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


    

print("Begin detection!")
while True:
    ret, img = cap.read()
    boxes, labels, probs = faceNet.run(img)
    outImg = img.copy()
    if len(boxes) > 0:
        box = boxes[0].numpy()
        org_x = box[0]
        org_y = box[1]
        print(box)
        # pfldImg = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        pfldImg = crop_roi(img, box[0], box[1], box[2], box[3], outCropRange=0.02)
        # cv2.imwrite('testout.jpg', pfldImg)
        landmarks = pfldNet.run(pfldImg)
        for (x, y) in landmarks:
            x = x + org_x
            y = y + org_y
            cv2.circle(outImg, (int(x), int(y)), 2, (0, 0, 255), -1)
    cv2.imshow('annotated', outImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
