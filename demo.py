from image_demo import img_seg

image = "Data/video_images/30427_hd_Trim_Trim/150.jpg"
right_line, left_line, direction, top_point = img_seg(image)
print(right_line, left_line, direction, top_point)