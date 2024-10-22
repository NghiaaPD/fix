from time import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.email_sent = False

        self.model = YOLO("models/best.pt")

        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / round(self.end_time - self.start_time, 2)
        text = f"FPS: {int(fps)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(
            im0,
            (20 - gap, 70 - text_size[1] - gap),
            (20 + text_size[0] + gap, 70 + gap),
            (255, 255, 255),
            -1,
        )
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)

            # Kiểm tra và thay đổi nhãn
            label = names[int(cls)]
            if label == "Amenazas - v1 2024-07-10 12-32am":
                label = "Fire"

            self.annotator.box_label(box, label=label, color=colors(int(cls), True))
        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        cv2.namedWindow("Fire Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Fire Detection", 800, 800)

        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if len(class_ids) > 0:  # Chỉ gửi email nếu chưa gửi trước đó
                if not self.email_sent:
                    self.email_sent = True
            else:
                self.email_sent = False

            self.display_fps(im0)
            cv2.imshow("Fire Detection", im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index="test-1.mp4")
detector()
