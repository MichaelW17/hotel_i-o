'''
11/15/2019:
10/30/2019: 试验第一种方法：门框全部简化为直线，根据框消失/出现时，框的中心点离哪条门框直线的距离最近，认为是进入/离开了哪个房间
1. 在手动选择每个门框的两个点之后，将两个点的横坐标取均值作为门框x位置，y方向先不用；
2. 明确跟踪算法的消失时间：max_age in tracker.py
'''

#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker_wmh import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import defaultdict
from math import sqrt

warnings.filterwarnings('ignore')

dumb_area = []  # 记录走廊尽头的范围，在这个区域内出现或消失的人相框不算进出
doors = []  # 记录门框位置的list，按下小写L输入四个点（四边形），按下i输入两个点（直线）
door_frame = []
door_counter = 0

ix, iy = -1, -1
x_pos = 0
y_pos = 0
key_pressed = ord('p')  # press 'a' to enter 4 points as a door frame, press 'd' to enter 2 points as a door "line"


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    '''
    :param param: param[0]: img under process, param[1]: imshow name like 'cool', param[2]: key_pressed
    :return:
    '''
    global dumb_area, doors, door_counter, door_frame, key_pressed

    # dumb area
    if param[2] == ord('a'):  # take 4 input points (left kick) as a dumb area
        if (event == cv2.EVENT_LBUTTONDOWN) & (door_counter < 4):
            dumb_area.append((x, y))
            door_counter += 1
            if door_counter == 4:
                door_counter = 0
                key_pressed = ord('p')
    if param[2] == ord('z'):  # 按下z重新设置dumb area
        if door_counter == 0:
            dumb_area = []
        if (event == cv2.EVENT_LBUTTONDOWN) & (door_counter < 4):
            dumb_area.append((x, y))
            door_counter += 1
            if door_counter == 4:
                door_counter = 0
                key_pressed = ord('p')

    # doors
    if param[2] == ord('d'):  # take 2 input points (left kick)
        if (event == cv2.EVENT_LBUTTONDOWN) & (door_counter < 2):
            door_frame.append((x, y))
            door_counter += 1
            if door_counter == 2:
                doors.append(door_frame)
                door_frame = []
                door_counter = 0
                key_pressed = ord('p')
    # cv2.imshow(param[1], param[0])

def point_in_rect(bbox, rect):  # 判断一个bbox的中心是否在一个矩形（可以是非规则四边形）内部
    """
    :param bbox: in the form like (xmin, ymin, xmax, ymax)
    :param rect: in the form like [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], points are randomly arranged
    :return: true if the center of the bbox lies inside the rectangle
    """
    # 得到人相框中心
    if len(rect) < 4:
        return False
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    # 得到矩形边界，(以rect指定的任意四边形的外接矩形为准)
    lr = sorted(rect, key=lambda arg: arg[0])  # left-right, 将rect的四个点按水平位置排序, lr[1][0]就是左边第二个点的x坐标
    ud = sorted(rect, key=lambda arg: arg[1])  # up-down, 将rect的四个点按垂直位置排序
    # print('rect: ', rect, 'lr: ', lr, 'ud: ', ud)
    if lr[0][0] < x < lr[3][0] and ud[0][1] < y < ud[3][1]:
        return True
    else:
        return False


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # 给bbox返回bbox内的特征嵌入？

    metric = nn_matching.NearestNeighborDistanceMetric(metric="cosine", matching_threshold=max_cosine_distance, budget=nn_budget)
    tracker = Tracker(metric, max_age=10)

    video = 'C:/Users/Minghao/Desktop/hotel3.mp4'
    video_capture = cv2.VideoCapture(video)

    frame_ID = 0
    fps = 0.0
    global key_pressed
    draw_newly_add = defaultdict(list)
    draw_newly_del = defaultdict(list)
    while 1:
        t1 = time.time()
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            print('again')
            video_capture = cv2.VideoCapture(video)
            frame_ID = 0
            continue
        else:
            frame_ID += 1

        if frame_ID < 1:
            continue

        frame_w, frame_h = frame.shape[:2]


        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)  # ndarray, 128-D

        # score to 1.0 here.
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]   # 这里好像只是初始化用来做坐标转换的类
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])  # 所有的bbox
        scores = np.array([d.confidence for d in detections])  # 都是1啊
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)


        # for track in tracker.tracks:  # 白框是跟踪结果
        #
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue
        #     bbox = track.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
        #     cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

        # for det in detections:  # 蓝框是检测结果
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)


        cv2.namedWindow("cool")
        cv2.resizeWindow("cool", 800, 480)
        # 画门框
        cv2.setMouseCallback("cool", on_EVENT_LBUTTONDOWN, [frame, 'cool', key_pressed])
        global doors, door_frame, door_counter
        for door in doors:
            if len(door) > 2:  # 较近的门，标出了4个点，画一个平行四边形
                points = np.array(door, np.int32).reshape((-1, 1, 2))
                # pts = points.reshape((-1, 1, 2))
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)
            else:  # 较远的门，标出了2个点，画一条直线
                if door[0][1] > door[1][1]:  # 比较哪个点在上面
                    top, bottom = door[0], door[1]
                else:
                    top, bottom = door[1], door[0]
                cv2.line(frame, top, bottom, (0, 255, 0), 2)

        if len(dumb_area):  # 画出dumb_area
            points = np.array(dumb_area, np.int32).reshape((-1, 1, 2))
            # pts = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], True, (255, 255, 0), 2)

        # 处理新增/消失的人相框
        # 新增
        if len(tracker.newly_add):
            print('new tracks: ', tracker.newly_add)  # 记录
            for key in tracker.newly_add.keys():
                # print('dumb_area: ', dumb_area)
                if not point_in_rect(tracker.newly_add[key], dumb_area):  # 如果新出现的框在dumb_area内部，就丢弃
                    draw_newly_add[key] = [tracker.newly_add[key], 100]  # 100是在图上显示100帧
            tracker.newly_add = defaultdict(list)
        for key in draw_newly_add.keys():
            if draw_newly_add[key][1] > 0:
                draw_newly_add[key][1] -= 1
                newbox = draw_newly_add[key][0]
                nbox_w, nbox_h = abs(newbox[2] - newbox[0]), abs(newbox[3] - newbox[1])
                nbox_ratio = round(nbox_h / nbox_w, 1)
                nbox_center = (int((newbox[0] + newbox[2]) / 2), int((newbox[1] + newbox[3]) / 2))
                if nbox_ratio > 3:
                    cv2.rectangle(frame, (int(newbox[0]), int(newbox[1])), (int(newbox[2]), int(newbox[3])), (0, 255, 0), 1)
                    cv2.putText(frame, 'outward', (int(newbox[0])-20, int(newbox[1])), 0, 1, (0, 255, 0), 2)
                    # 计算人相框中心dbox_center到各个门框的距离
                    min_dis, min_door, min_door_center = 1e10, [(1, 1), (frame_w, frame_h)], (1, 1)
                    for door in doors:  # 默认每个门框都只用两个点标出来
                        door_center = (int((door[0][0] + door[1][0]) / 2), int((door[0][1] + door[1][1]) / 2))
                        dis = sqrt((nbox_center[0] - door_center[0]) ** 2 + (nbox_center[1] - door_center[1]) ** 2)
                        if dis < min_dis:
                            min_dis = dis
                            min_door_center = door_center
                            min_door = door
                    if (min_door != (1, 1)) and (min_dis < abs(min_door[0][1] - min_door[1][1]) * 0.3):
                        cv2.line(frame, nbox_center, min_door_center, (0, 255, 255), 2)
        for key in draw_newly_add.keys():  # 出现/消失的人相框显示100帧后，删除
            if draw_newly_add[key][1] < 0:
                del draw_newly_add[key]
        # 消失
        if len(tracker.newly_del):
            print('del tracks: ', tracker.newly_del)  # 记录
            for key in tracker.newly_del.keys():
                if not point_in_rect(tracker.newly_del[key], dumb_area):
                    draw_newly_del[key] = [tracker.newly_del[key], 100]  # 100是在图上显示100帧
            tracker.newly_del = defaultdict(list)
        for key in draw_newly_del.keys():
            if draw_newly_del[key][1] > 0:
                draw_newly_del[key][1] -= 1
                delbox = draw_newly_del[key][0]
                dbox_w, dbox_h = abs(delbox[2] - delbox[0]), abs(delbox[3] - delbox[1])
                dbox_ratio = round(dbox_h / dbox_w, 2)
                dbox_center = (int((delbox[0] + delbox[2]) / 2), int((delbox[1] + delbox[3]) / 2))
                if dbox_ratio > 3:
                    # 先把人相框画出来
                    cv2.rectangle(frame, (int(delbox[0]), int(delbox[1])), (int(delbox[2]), int(delbox[3])), (0, 0, 255), 1)
                    cv2.putText(frame, 'inward', (int(delbox[0])-20, int(delbox[1])), 0, 1, (0, 0, 255), 2)
                    # 计算人相框中心dbox_center到各个门框的距离
                    min_dis, min_door, min_door_center = 1e10, [(1, 1), (frame_w, frame_h)], (1, 1)
                    for door in doors:  # 默认每个门框都只用两个点标出来
                        door_center = (int((door[0][0] + door[1][0]) / 2), int((door[0][1] + door[1][1]) / 2))
                        dis = sqrt((dbox_center[0] - door_center[0])**2 + (dbox_center[1] - door_center[1])**2)
                        if dis < min_dis:
                            min_dis = dis
                            min_door_center = door_center
                            min_door = door
                    if (min_door != (1, 1)) and (min_dis < abs(min_door[0][1] - min_door[1][1]) * 0.3):
                        cv2.line(frame, dbox_center, min_door_center, (255, 255, 0), 2)



        for key in draw_newly_del.keys():  # 出现/消失的人相框显示100帧后，删除
            if draw_newly_del[key][1] < 0:
                del draw_newly_del[key]




        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(frame, 'frame ID: {},  fps: {},  key_pressed: {}'.format(frame_ID, round(fps, 1), chr(key_pressed)),
                    (20, 30), 0, 1, (0, 255, 0), 2)
        cv2.imshow('cool', frame)

        # Press Q or Esc to stop!
        if key_pressed == ord('q') or key_pressed == 27:
            break
        # else:
            # print('doors: ', doors)
            # print('door_frame: ', door_frame, 'door_counter: ', door_counter)

        key = cv2.waitKey(1)
        if key != -1:
            key_pressed = key

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())