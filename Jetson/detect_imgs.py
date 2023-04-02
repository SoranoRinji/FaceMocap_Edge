"""
This code is used to batch detect images in a folder.
"""
import sys
import cv2

from vision.ssd.config.fd_config import define_img_size

class FastFaceDetNet():
    def __init__(self):
        self.threshold = 0.7    # 'score threshold'
        
        self.candidate_size = 1500  # nms candidate size

        define_img_size(160)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor' # define network input size,default optional value 128/160/320/480/640/1280

        from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
        from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

        self.label_path = "./models/voc-model-labels.txt"
        self.test_device = "cuda:0"  # 'cuda:0 or cpu'

        self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.model_path = "models/pretrained/version-slim-320.pth"
        # model_path = "models/pretrained/version-slim-640.pth"
        self.net = create_mb_tiny_fd(len(self.class_names), is_test=True, device=self.test_device)
        self.predictor = create_mb_tiny_fd_predictor(self.net, candidate_size=self.candidate_size, device=self.test_device)
        self.net.load(self.model_path)
    def run(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(image, self.candidate_size / 2, self.threshold)
        # print(f"Found {len(probs)} faces.")
        return boxes, labels, probs
        
    
