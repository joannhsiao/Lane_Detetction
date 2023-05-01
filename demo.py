import cv2
from lane_line_detect import init, img_seg

model, checkpoint, palette = init()

image = "Data/video_images/30427_hd_Trim_Trim/150.jpg"
img = cv2.imread(image)

right_line, left_line, direction, top_point = img_seg(img, model, checkpoint, palette)

# x1, y1, x2, y2 = right_line
# x3, y3, x4, y4 = left_line
# cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
# cv2.circle(img, (x1, y1), 5, (255, 0, 0), 0)
# cv2.circle(img, (x2, y2), 5, (0, 255, 0), 0)
# cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 2)
# cv2.circle(img, (x3, y3), 5, (255, 0, 0), 0)
# cv2.circle(img, (x4, y4), 5, (0, 255, 0), 0)

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()