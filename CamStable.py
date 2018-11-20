import cv2


class Stabilizer:
    def stabilize(self,image, old_frame):
            border_crop = 10
            feature_params = dict( maxCorners = 100 ,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )

            lk_params = dict( winSize  = (15, 15),
                              maxLevel = 1,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            try:
                if old_frame==0:
                    ret, old_frame = image
            except:
                print("\n")

            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            rows,cols = old_gray.shape
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


            ret,frame = image
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            good_new = p1[st==1]
            good_old = p0[st==1]

            h=cv2.findHomography(good_old,good_new)
            result=cv2.warpPerspective(frame,h[0], (frame.shape[1],frame.shape[0]))
            cropped_stabilized_frame = result[border_crop:rows-border_crop, border_crop:cols-border_crop]

            return frame, cropped_stabilized_frame