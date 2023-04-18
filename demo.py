from image_demo import img_seg

image = "Data/video_images/30427_hd_Trim_Trim/130.jpg"
right_line, left_line, direction, point = img_seg(image)
print(right_line, left_line, direction, point)