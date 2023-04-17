# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import numpy as np
import math
import time
    
def cal_slope(x1, y1, x2, y2):
    if x2 == x1:  # devide by zero
        return None
    else:
        return ((y2 - y1) / (x2 - x1))

def intercept(x, y, slope):
    return y - x * slope

def cal_intersection(lines, img):
    coordinates = list(zip(lines[:, 0, 0], lines[:, 0, 1], lines[:, 0, 2], lines[:, 0, 3]))

    x1, y1, x2, y2 = coordinates[0]
    x3, y3, x4, y4 = coordinates[-1]

    for coordinate in coordinates:
        if coordinate[0] < x1 and coordinate[2] < x2 and coordinate[1] > y1 and coordinate[3] > y2:
            x1, y1, x2, y2 = coordinate
    
    for coordinate in coordinates:
        if coordinate[0] > x3 and coordinate[2] > x4 and coordinate[1] > y3 and coordinate[3] > y4:
            if coordinate != [x1, y1, x2, y2]:
                x3, y3, x4, y4 = coordinate

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.line(img, (x3, y3), (x4, y4), (0, 0, 255), 2)

    # y = mx + b
    slope_l = cal_slope(x1, y1, x2, y2)
    slope_r = cal_slope(x3, y3, x4, y4)
    b1 = intercept(x1, y1, slope_l)
    b2 = intercept(x3, y3, slope_r)

    if slope_l is None:
        if not slope_r is None:
            x = x1
            y = slope_r * x1 + b2
        else:   # slope_r & slope_l are both none
            if x1 == x3:
                return int(x1), int(y1)
            else:
                return None, None
    elif slope_r is None:
        x = x3
        y = slope_l * x3 + b1
    else:
        if slope_r == slope_l:
            if b1 == b2:
                return int(x1), int(y1)
            else:
                return None, None
        else:
            x = (b2 - b1) / (slope_l - slope_r)
            y = slope_l * x + b1

    return int(x), int(y)

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def pipline(model, img):
    segment_image = inference_segmentor(model, img)

    # 把非0的全部統一為1
    segment_image = np.array(segment_image[0]).astype(np.uint8)
    segment_image[segment_image == 1] = 0   # 把sideroad和road 合成一個類別
    gray_img = np.minimum(segment_image, 1)*255

    """ Canny """
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)

    """ find contours """
    # contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0]

    res = np.where(edges == 255)
    contours = list(zip(res[1], res[0]))

    """ contour smoothing """
    contours = np.mat(contours)
    cv2.approxPolyDP(contours, 3, closed=False)
    edges = np.zeros_like(segment_image)
    cv2.drawContours(edges, contours, -1, 255, 2, 8)
    # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    """ mask """
    # imshape = edges.shape
    # vertices = np.array([[(0, imshape[0]), (0, imshape[0]/2), (imshape[1], imshape[0]/2), (imshape[1], imshape[0])]], dtype=np.int32)
    # mask = np.zeros_like(edges)
    # cv2.fillPoly(mask, vertices, 255)
    # masked_img = cv2.bitwise_and(edges, mask)

    """ hough transform """
    rho = 1     # 原點到直線的最短直線距離
    theta = np.pi / 180     # 最短直線與X軸的夾角
    threshold = 100
    min_line_len = 25
    max_line_gap = 25
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # line_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    """ the highest of y-axis """
    # res = np.where(edges == 255)
    # coordinates= list(zip(res[1], res[0]))
    # origin = (int(edges.shape[1]/2), edges.shape[0])
    # top_point = (edges.shape[1],edges.shape[0])
    # for coordinate in coordinates:
    #     if coordinate[1] < top_point[1]:
    #         top_point = coordinate
    # cv2.line(edges, origin, top_point, 255, 2)

    test_images_output = weighted_img(edges, img, α=0.8, β=1., γ=0.)

    return test_images_output, lines

def find_highest(lines, center_x, center_y):
    coordinates = list(zip(lines[:, 0, 0], lines[:, 0, 1]))

    point = coordinates[0]
    # distance = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)

    for coordinate in coordinates:
        # if coordinate[1] < point[1] and math.sqrt((coordinate[0] - center_x)**2 + (coordinate[1] - center_y)**2) > distance:
        if coordinate[1] < point[1]:
            point = coordinate

    return point[0], point[1]

def Drawline(img, x1, y1, x2, y2, color=(255, 0, 0)):
    cv2.line(img, (x1, y1), (x2, y2), color, 2)

def cal_angle(point_x, point_y, center_x, center_y):
    if point_x is None or point_y is None:
        return None

    radian = math.atan2(point_y - center_y, point_x - center_x)
    angle = (radian * 180) / math.pi
    return angle

def determine_direction(included_angle):
    if included_angle > 5:
        return "right"
    elif included_angle < -5:
        return "left"
    else:
        return "straight"

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="../../Data/video_images/30427_hd_Trim_Trim/130.jpg", help='Image file')
    parser.add_argument('--config', default="configs/cityscapes/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.py", help='Config file')
    parser.add_argument('--checkpoint', default="checkpoint_dir/seg/upernet_internimage_l_512x1024_160k_mapillary2cityscapes.pth", help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', choices=['ade20k', 'cityscapes', 'cocostuff'], help='Color palette used for segmentation map')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    """ Test a single image """
    # img = cv2.imread(args.img)
    # output_img = pipline(model, img)

    # # segment_image = inference_segmentor(model, img)
    # # #show segment result
    # # if hasattr(model, 'module'):
    # #     model = model.module
    # # seg_result = model.show_result(img, segment_image,
    # #             palette=get_palette(args.palette),
    # #             show=False, opacity=args.opacity)

    """ Test video """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo.avi', fourcc, 20.0, (1920, 1080))
    cap = cv2.VideoCapture("../../Data/30427_hd_Trim_Trim.mp4")

    i = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        test_images_output, lines = pipline(model, frame)
    
        # inter_x, inter_y = cal_intersection(lines, test_images_output)
        center_x, center_y = int(frame.shape[1] / 2), frame.shape[0]
        inter_x, inter_y = find_highest(lines, center_x, center_y)  # highest point
        point_x, point_y = int(frame.shape[1] / 2), int(frame.shape[0] / 2)
        
        Drawline(test_images_output, center_x, center_y, inter_x, inter_y, (255, 0, 0))
        Drawline(test_images_output, center_x, center_y, point_x, point_y, (0, 255, 0))

        angle1 = cal_angle(inter_x, inter_y, center_x, center_y)
        angle2 = cal_angle(point_x, point_y, center_x, center_y)
        if angle1 is not None:
            included_angle = round(angle1 - angle2, 2)
            direction = determine_direction(included_angle)
            cv2.putText(test_images_output, direction, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
            # cv2.putText(test_images_output, str(included_angle), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.imshow("demo", test_images_output)
        cv2.waitKey(1)

        out.write(test_images_output)
        print("[{}], cost: {}".format(i, time.time() - start_time))

        i += 1
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()