from CamStable import Stabilizer
import cv2



cap = cv2.VideoCapture(1)

frame = 0
counter = 0

stabilizer = Stabilizer()

while True:
    image = cap.read()
    frame, result=stabilizer.stabilize(image, frame)

    cv2.imshow("Result", result)
    cv2.imshow("Image", image[1])
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()