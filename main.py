import cv2
import mediapipe as mp
import time


def init():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    mpFace = mp.solutions.face_detection
    face = mpFace.FaceDetection(0.70)
    mpDraw = mp.solutions.drawing_utils

    previousTime = 0
    while True:
        success, img = cap.read()
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face.process(imageRGB)
        if results.detections:
            for detection in results.detections:
                # mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (1, 255, 10), 2)
                cv2.putText(img, f'Soilihi: {int(detection.score[0]*100)}%', (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 255, 10), 2)
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 250), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    init()