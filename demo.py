import cv2
from lane_line_detect import initialize, img_seg


config = "configs/cityscapes/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.py"
checkpoint = "checkpoint_dir/seg/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.pth"
initialize(config=config, checkpoint=checkpoint)

image = "Data/video_images/30427_hd_Trim_Trim/150.jpg"
img = cv2.imread(image)

right_line, left_line, direction, highest_point, included_angle = img_seg(img)
