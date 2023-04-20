from lane_detection import init, img_seg

image = "Data/video_images/30427_hd_Trim_Trim/150.jpg"
model, checkpoint, palette = init()
right_line, left_line, direction, top_point = img_seg(image, model, checkpoint, palette)
print(right_line, left_line, direction, top_point)