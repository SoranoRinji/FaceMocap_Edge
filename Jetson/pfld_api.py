import numpy as np
import onnxruntime
import time
import cv2

class PFLDNet():
    def __init__(self):
        self.onnx_model_path = "model/lite.onnx"

        self.session = onnxruntime.InferenceSession(self.onnx_model_path, providers=['CUDAExecutionProvider'])  # 'CUDAExecutionProvider' 'TensorrtExecutionProvider'
        # get the name of the first input of the model
        self.input_name = self.session.get_inputs()[0].name

    def preprocessing(self, img):
        img = cv2.resize(img, (112, 112))
        image_data = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255
        return image_data

    def postprocessing(self, img, ret):
        landmarks = ret.reshape(-1, 2)
        landmarks[:, 0] = landmarks[:, 0] * img.shape[1]
        landmarks[:, 1] = landmarks[:, 1] * img.shape[0]
        return landmarks

    def run(self, img):
        image_data = self.preprocessing(img)
        import time
        tic = time.time()
	
        output = self.session.run([], {self.input_name: image_data})[1]

        t = (time.time() - tic) / 100

        print('average infer time: {:.4f}ms, FPS: {:.2f}'.format(t * 1000, 1 / t))
        print('output.shape: ', output.shape)
        landmarks = self.postprocessing(img, output)
        return landmarks

    # print(output[0])

    # if show_img:
    #     landmarks = output.reshape(-1, 2)
    #     landmarks[:, 0] = landmarks[:, 0] * img.shape[1]
    #     landmarks[:, 1] = landmarks[:, 1] * img.shape[0]
    #     img_copy = img.copy().astype(np.uint8)
    #     for (x, y) in landmarks:
    #         cv2.circle(img_copy, (int(x), int(y)), 2, (0, 0, 255), -1)
    #     # cv2.imshow('demo', img_copy)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #     cv2.imwrite('result1.jpg', img_copy)
