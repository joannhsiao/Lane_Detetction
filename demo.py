import cv2
from lane_detection import init, img_seg

model, checkpoint, palette = init()

image = "Data/video_images/30427_hd_Trim_Trim/150.jpg"
img = cv2.imread(image)

right_line, left_line, direction, top_point = img_seg(img, model, checkpoint, palette)
