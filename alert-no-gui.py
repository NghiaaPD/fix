from time import time
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.email_sent = False

        self.model = YOLO("yolo11n_ncnn_model")

        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        self.device = "cpu"  # Chạy trên CPU

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / round(self.end_time - self.start_time, 2)
        print(f"FPS: {int(fps)}")

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)

            label = names[int(cls)]
            if label == "Amenazas - v1 2024-07-10 12-32am":
                label = "Fire"

            self.annotator.box_label(box, label=label, color=colors(int(cls), True))

        cv2.imwrite(f"output/frame_{int(time())}.jpg", im0)

        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0:
                if not self.email_sent:
                    print("Alert: Detected Fire!")
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)

            frame_count += 1
            if frame_count >= 100000:
                break

        cap.release()


detector = ObjectDetection(capture_index="test-1.mp4")
detector()
