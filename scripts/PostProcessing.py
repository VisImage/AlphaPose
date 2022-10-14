# importing libraries
import os
from re import X
import cv2
from PIL import Image
import math

import json
# import cv2
import numpy as np
import sys
from collections import defaultdict
import copy

# json_obj_list: the personal json object in one frame, input
# img: the image of the frame, input and output
def distance (p1,p2):
  dx = p1[0] - p2[0]
  dy = p1[1] - p2[1]
  return math.sqrt( dx**2 + dy**2 )

def drawPose(json_obj_list, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(img_gray)
    img_out = img

    for json_obj in json_obj_list:
        #if frame_id > 0 :
        keypoints = json_obj['keypoints']
        joints = []
        for i in range(int(len(keypoints)/3)):
            joint = (keypoints[i*3],keypoints[i*3 + 1])
            joints.append(joint)
        #print(f'keypoints:{joints}')
        #cPts = np.array([joints[2], joints[5], joints[12], joints[9],joints[2]])
        cPts = np.array([joints[5], joints[6], joints[12], joints[11],joints[5]])
        # maskImage=np.zeros_like(img_gray)
        # cv2.drawContours(maskImage,[cPts.astype(int)],0,255,-1)
        # local_mean = cv2.mean(img_gray, mask=maskImage)
        #print(f'person_id:{person_id}   mean:{mean}    localMean:{local_mean}')
        #print(f'cPots: {cPts}')
        #exit()

        cv2.polylines(img_out,[cPts.astype(int)],True,(255,0,0), 2)

    return img_out

# Video Generating function
def generate_video(image_folder,video_name):
    #image_folder = '.' # make sure to use your folder
    #video_name = 'mygeneratedvideo.avi'
    #os.chdir("C:\\Python\\Geekfolder2")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")]

    # Array images should only consider
    # the image files ignoring others if any

    images.sort()
    #print(f'ImageSize={len(images)}; video_name = {video_name}  images= {images}')

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape
    print(f'h={height}; w={width}')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #video = cv.VideoWriter(file_path, fourcc, fps, (w, h))

    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    # Appending the images to the video one by one
    for image in images:
        #fileName = os.path.join(image_folder, image)
        #print(f'fileName={fileName}')
        video.write(cv2.imread(os.path.join(image_folder, image)))

    #print(images)
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated
    print(f'{video_name} is generated.')

def get_frame_dict(pose_list):
    frame_dict = defaultdict()
    for pose in pose_list:
        frame_id = pose["image_id"]
        if frame_id in frame_dict:
            IV = int(pose["idx"])
            frame_dict[frame_id].append(IV)
        else:
            frame_dict[frame_id] = [int(pose["idx"])]
    return frame_dict

#generate pose dictionary
def get_pose_dict(pose_list):

    frame_dict = get_frame_dict(pose_list)
    # for pose in pose_list:
    #     frame_id = pose["image_id"]
    #     if frame_id in frame_dict:
    #         IV = int(pose["idx"])
    #         frame_dict[frame_id].append(IV)
    #     else:
    #         frame_dict[frame_id] = [int(pose["idx"])]

    pose_dict = defaultdict()
    for pose in pose_list:
        image_id = int(pose['image_id'].replace('.jpg',''))
        pose_id = pose['idx']
        pose_id = str(int(float(pose_id)))  # with --detector tracker  "idx": 9.0
        #print( "pose_id 2 =", pose_id)
        if pose_id == '765'  and image_id == 7 :
            ii = 99
        if pose_id == '1153'  and image_id == 2328:
            ii = 99
        if pose_id in pose_dict:
            pose_list = pose_dict[pose_id]
            frame_id = str(image_id - 1)+".jpg"
            if frame_id in frame_dict:
                if int(pose_id) in frame_dict[str(image_id - 1)+".jpg"]:
                    pose_list[len(pose_list)-1][1] = image_id
                else:
                    pose_list.append([image_id, image_id])
            else:
                pose_list.append([image_id, image_id])
        else:
            pose_dict[pose_id] = [[image_id, image_id]]

    return pose_dict

#generate pose dictionary
def get_pose_dict_2(pose_list):

    pose_dict = defaultdict()
    for pose in pose_list:
        image_id = int(pose['image_id'].replace('.jpg',''))
        pose_id = pose['idx']
        pose_id = str(int(float(pose_id)))  # with --detector tracker  "idx": 9.0
        if pose_id in pose_dict:
            pose_list = pose_dict[pose_id]
            pose_dict[pose_id].append(image_id)
        else:
            pose_dict[pose_id] = [image_id]

        # frame_dict_name = result +"/frame_dict.txt"
        # with open(frame_dict_name, 'w') as f:
        #     print(frame_dict, file=f)

        # pose_dict_name = result +"/pose_dict.txt"
        # with open(pose_dict_name, 'w') as f:
        #     print(pose_dict, file=f)

    return pose_dict

def get_overlap_dict(frame_dict):
    overlap_dict = defaultdict()
    for i in range(len(frame_dict)):
        pose_list= frame_dict[i]
        list_len = len(pose_list)
        for j in range(list_len):
            box_j = pose_list[j]['box']
            for k in range(j+1, list_len):
                box_k = pose_list[k]['box']
                over_lap_rate = float(int(overLap_rate(box_j, box_k)*1000))/1000
                if over_lap_rate > 0:
                    overlap_dict[i] = [j,k, over_lap_rate]

    return overlap_dict

def removeDarkClothing(json_obj_list, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(img_gray)
    json_list_output = []

    for json_obj in json_obj_list:
        #if frame_id > 0 :
        keypoints = json_obj['keypoints']
        joints = []
        for i in range(int(len(keypoints)/3)):
            joint = (keypoints[i*3],keypoints[i*3 + 1])
            joints.append(joint)
        #print(f'keypoints:{joints}')
        #cPts = np.array([joints[2], joints[5], joints[12], joints[9],joints[2]])
        cPts = np.array([joints[5], joints[6], joints[12], joints[11],joints[5]])
        maskImage=np.zeros_like(img_gray)
        cv2.drawContours(maskImage,[cPts.astype(int)],0,255,-1)
        local_mean = cv2.mean(img_gray, mask=maskImage)

        if mean < local_mean:  #fencing clothing is white so the brightness value is high
            json_list_output.append(json_obj)

    return json_list_output

# json_obj_list: the personal json object in one frame, input
# img: the image of the frame, input and output

def removeFlat(json_obj_list, img):
#    remove pose a flat width > 2 * height bbox
    json_list_output = []

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj['box']
        width = box[2]
        height = box[3]
        if width < height * 2 :
            json_list_output.append(json_obj)
    img_out = img
    return json_list_output, img_out

def removeFlat_2(json_obj_list):
#    remove pose a flat width > 2 * height bbox
    json_list_output = []

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj['box']
        width = box[2]
        height = box[3]
        if width < height * 2 :
            json_list_output.append(json_obj)
    return json_list_output


def removeSmall_2(json_obj_list):
#    if image_id > 285:
#        print(f'image_id = {image_id}, list_len={json_obj_list}')
    json_list_output = []
    #sort list by box area
    #index = len(json_obj_list) - 1
    while len(json_obj_list) > 0:
        max_obj_area = 0
        for i in range(len(json_obj_list)):
            json_obj = json_obj_list[i]
            box = json_obj['box']
            if box[2] * box[3] > max_obj_area:
                II = i
                max_obj_area = box[2] * box[3]
        json_list_output.append(json_obj_list[II])
        json_obj_list.pop(II)

    json_obj_list = json_list_output

    json_list_output = []
    max_box = json_obj_list[0]['box']
    max_obj_area = max_box[2] * max_box[3]

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj['box']
        p1 = [box[0],box[1]]
        p2 = [box[0],box[1]+box[3]]
        p3 = [box[0] + box[2],box[1]+box[3]]
        p4 = [box[0] + box[2],box[1]]
        pts = np.array([p1,p2,p3,p4], np.int32)
        if box[2] * box[3] > max_obj_area / 9:
            json_list_output.append(json_obj)

            str_id = json_obj["image_id"]
            # print(f'image_id={str_id}')
            # if str_id == "130.jpg":
            # print(json_obj)
            # sys.exit()
        # else:
        #    cv2.polylines(img_out,[pts],True,(255,0,0), 1)

    return json_list_output


def removeSmall(json_obj_list, img):
#    if image_id > 285:
#        print(f'image_id = {image_id}, list_len={json_obj_list}')
    json_list_output = []
    #sort list by box area
    #index = len(json_obj_list) - 1
    while len(json_obj_list) > 0:
        max_obj_area = 0
        for i in range(len(json_obj_list)):
            json_obj = json_obj_list[i]
            box = json_obj['box']
            if box[2] * box[3] > max_obj_area:
                II = i
                max_obj_area = box[2] * box[3]
        json_list_output.append(json_obj_list[II])
        json_obj_list.pop(II)

    img_out = img
    json_obj_list = json_list_output

    json_list_output = []
    max_box = json_obj_list[0]['box']
    max_obj_area = max_box[2] * max_box[3]

    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        box = json_obj['box']
        p1 = [box[0],box[1]]
        p2 = [box[0],box[1]+box[3]]
        p3 = [box[0] + box[2],box[1]+box[3]]
        p4 = [box[0] + box[2],box[1]]
        pts = np.array([p1,p2,p3,p4], np.int32)
        if box[2] * box[3] > max_obj_area / 9:
            json_list_output.append(json_obj)
            cv2.polylines(img_out,[pts],True,(128,128,128), 1)
            if int(json_obj['idx']) == 765:
                cv2.polylines(img_out,[pts],True,(255,0,0), 3)
            if int(json_obj['idx']) == 816:
                cv2.polylines(img_out,[pts],True,(0,255,0), 3)
            if int(json_obj['idx']) == 1016:
                cv2.polylines(img_out,[pts],True,(0,0,255), 3)

            V1 = 0 #pose_dict[json_obj['idx']][0]
            V2 = 3 #V1 + pose_dict[json_obj['idx']][1]
            #cv2.putText(img_out, f'{V1}-{V2}', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            cv2.putText(img_out, str(json_obj['idx']), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            str_id = json_obj["image_id"]
            # print(f'image_id={str_id}')
            # if str_id == "130.jpg":
            # print(json_obj)
            # sys.exit()
        # else:
        #    cv2.polylines(img_out,[pts],True,(255,0,0), 1)

    return json_list_output, img_out

def overLap_rate (b1,b2):
    left = max(b1[0], b2[0])
    right = min(b1[0]+b1[2], b2[0]+b2[2])
    bottom = max(b1[1], b2[1])
    top = min(b1[1]+b1[3], b2[1]+b2[3])
    if left < right and bottom < top:
        area1 = (right-left)*(top - bottom)
        area2 = b1[2]*b1[3]
        area3 = b2[2]*b2[3]
        return area1/min(area2,area3)
    else:
        return 0

def _get_bbox(keypoints_merge):
    Xmin = 1000000000
    Xmax = 0
    Ymin = 1000000000
    Ymax = 0
    size = int (len(keypoints_merge)/3)
    for i in range(size):
        x = keypoints_merge[i*3]
        y = keypoints_merge[i*3 + 1]
        if Xmin > x :
            Xmin = x
        elif Xmax < x :
            Xmax = x
        if Ymin > y :
            Ymin = y
        elif Ymax < y :
            Ymax = y
    return Xmin, Ymin, Xmax - Xmin, Ymax - Ymin

def get_Overlap_list(json_obj_list):
    overlap_list = []
    no_pose_in_frame = len(json_obj_list)
    for j in range(no_pose_in_frame):
        Jbox = json_obj_list[j]['box']
        for k in range(j+1, no_pose_in_frame):
            over_lap_rate = float(int(overLap_rate(Jbox, json_obj_list[k]['box'])*1000))/1000
            if over_lap_rate > 0:
                overlap_list.append([json_obj_list[j]["idx"],json_obj_list[k]["idx"], over_lap_rate])

    return overlap_list

def remove_overlap_by_size(json_obj_list,overlap_list, ratio):

    for i in range(len(overlap_list)):
        overlap = overlap_list[i]

        box_idx_0 = 0
        box_idx_1 = 0
        json_list_out = []
        over_lap_out = []
        for pose in json_obj_list:
            if pose['idx'] == overlap[0]:
                box_idx_0 = pose['idx']
                box_area_0 = pose['box'][2] * pose['box'][3]
            elif pose['idx'] == overlap[1]:
                box_idx_1 = pose['idx']
                box_area_1 = pose['box'][2] * pose['box'][3]
        if box_area_0 * ratio < box_area_1:
            for pose in json_obj_list:
                if pose['idx'] != box_idx_0:
                    json_list_out.append(pose)
        elif box_area_1 * ratio < box_area_0:
            for pose in json_obj_list:
                if pose['idx'] != box_idx_1:
                    json_list_out.append(pose)
        else:
            over_lap_out.append(overlap)
            for pose in json_obj_list:
                json_list_out.append(pose)

    return json_list_out, over_lap_out

def keypoint_dis(list1, list2):
    return 0
def keypoint_average(list1, list2):
    return list1
def get_diff(list1):
    return list1
def merge_pose_list(list1, list2):
    return list1


def merge_overlap_pose(pose_list, overlap_list):
    overlap_out = []
    pose_out = []
    for overlap in overlap_list:
        idx0 = overlap[0]
        idx1 = overlap[1]
        for pose in pose_list:
            if pose['idx'] == idx0:
                pose_0 = pose
            elif pose['idx'] == idx1:
                pose_1 = pose
            else:
                pose_out.append(pose)

        keypoints_0 = pose_0['keypoints']
        keypoints_1 = pose_1['keypoints']
        size = int(len(keypoints_0)/3)
        keypoints_diff = [0] * size
        pose_merge = pose_0
        keypoints_merge = keypoints_0
        for index in range(size):
            i = index * 3
            score_thre = 0.6
            keypoints_merge[i] = (keypoints_0[i] + keypoints_1[i] )/2
            keypoints_merge[i+1] = (keypoints_0[i+1] + keypoints_1[i+1] )/2
            keypoints_merge[i+2] = min(keypoints_0[i+2] , keypoints_1[i+2] )

            if keypoints_0[i + 2] > score_thre and keypoints_1[i + 2] > score_thre:
                keypoints_diff[index] = max((keypoints_0[i] - keypoints_1[i]), (keypoints_0[i+1] - keypoints_1[i+1]))

        merge_thre = 15
        merge_status = True
        for diff in keypoints_diff:
            if diff > merge_thre:
                merge_status = False
        
        if merge_status == True:
            if pose_0['score'] > pose_1['score']:
                pose_merge = pose_0
            else:
                pose_merge = pose_1
            pose_merge['keypoints'] = keypoints_merge
            pose_merge['box'] = _get_bbox(keypoints_merge)
            pose_out.append(pose_merge)    
        else:
            overlap_out.append(overlap)
            pose_out.append(pose_0)
            pose_out.append(pose_1)

    return pose_out, overlap_out

def remove_overlap_by_merge(pose_list, thre):
    pose_list_out = []
    length = len(pose_dict)
    for i in range(length):
        list_i = pose_dict[i]
        for j in range( i + 1, length):
            list_j = pose_dict[j]
            comman_frame_list =  list_i & list_j
            if comman_frame_list != []:
                continue
        if comman_frame_list == []:
            pose_list_out.append(list_i)
        else:
            comman_frame_list =  list_i & list_j
            comman_frame_dis_list = get_diff(comman_frame_list)
            # if comman_frame_list == []
            #     pose_list_out.append(merge_pose_list(list_i, list_j)
            # pose_1 = -1
            # pose_list_out = []
        #     for pose in json_obj_list:
        #         if pose['idx'] == overlap[0]:
        #             box_idx_0 = pose['idx']
        #             keypoints_0 = pose['keypoints']
        #         elif pose['idx'] == overlap[1]:
        #             box_idx_1 = pose['idx']
        #             keypoints_1 = pose['keypoints']
        #     if keypoint_dis (keypoints_0,keypoints_1) < thre:
        #         for pose in json_obj_list:
        #             if pose['idx'] != box_idx_0:
        #                 json_list_out.append(pose)        json_list_out = []
        # over_lap_out = []
        # else:
        #     over_lap_out.append(overlap)

    return pose_list_out

def remove_overlap_by_sequence(pose_list, overlap_list,pose_dict,frame_list):
    pose_list_out = []
    frame_list_out=[]
    pose_dict_out = []

    for i in range(len(overlap_list)):
        overlap = overlap_list[i]
        box_idx_0 = 0
        box_idx_1 = 0
        json_list_out = []
        over_lap_out = []


    return pose_list_out, pose_dict_out


def checkForFencer(json_obj):
    box = json_obj['box']

    keypoints = json_obj['keypoints']
    numOfPoint = int(len(keypoints)/3)
    joints = []
    for i in range(numOfPoint):
        joint = (keypoints[i*3],keypoints[i*3 + 1])
        joints.append(joint)

    LElbow = joints[7]
    RElbow = joints[8]
    LWrist = joints[9]
    RWrist = joints[10]
    LAnkle = joints[15]
    RAnkle = joints[16]
    LAnkle = joints[15]
    RAnkle = joints[16]

    #print(f'LElbow={LElbow}, RElbow={RElbow}, LWrist={LWrist},  RWrist={RWrist}, LAnkle={LAnkle}, RAnkle={RAnkle}, LAnkle={LAnkle}, RAnkle={RAnkle}')
    isFeetDisOk = False
    #print(f'distance(LAnkle,RAnkle)={distance(LAnkle,RAnkle)}, box[3]/4={box[3]/4}')
    if distance(LAnkle,RAnkle) > box[3]/4:
        isFeetDisOk = True
    isLArmOk = False

    isArmOk = False
    if abs(LElbow[1] - LWrist[1]) < abs(LElbow[0]  - LWrist[0]) :
        isLArmOk = True

    if abs(RElbow[1] - RWrist[1]) < abs(RElbow[0]  - RWrist[0]) :
        isLArmOk = True

    firstfencerFound = False
    if isArmOk or isFeetDisOk:
      firstFencingID = i
      firstfencerFound = True

    return firstfencerFound

def findCloseKeypoint(json_list_output):
  obj1 = json_list_output[0]
  obj2 = json_list_output[1]

  keypoint1 = obj1['keypoints']
  keypoint2 = obj2['keypoints']

  numOfPoint = int(len(keypoint2)/3)
  joints_1 = []
  joints_2 = []
  for i in range(numOfPoint):
      joint_1 = (keypoint1[i*3],keypoint1[i*3 + 1])
      joints_1.append(joint_1)
      joint_2 = (keypoint2[i*3],keypoint2[i*3 + 1])
      joints_2.append(joint_2)

  average_1_x = joints_1[-1][0]
  average_1_y = joints_1[-1][1]
  average_2_x = joints_2[-1][0]
  average_2_y = joints_2[-1][1]

  for i in range(numOfPoint - 1):
     #print(f'joints_1[i]:{joints_1[i]}')
     average_1_x +=joints_1[i][0]
     average_1_y +=joints_1[i][1]
     average_2_x +=joints_2[i][0]
     average_2_y +=joints_2[i][1]

  average_1_x = average_1_x/numOfPoint
  average_1_y = average_1_y/numOfPoint
  average_2_x = average_2_x/numOfPoint
  average_2_y = average_2_y/numOfPoint

  min_1 = 100000
  min_2 = 100000
  min_1_id = -1
  min_2_id = -1

  average_1 = (int(average_1_x),int(average_1_y))
  average_2 = (int(average_2_x),int(average_2_y))
  for i in range(numOfPoint):
     #print(f'distance(average_2, joints_1[{i}])={distance(average_2, joints_1[i])}')
     if min_1 > distance(average_2, joints_1[i]):
         min_1 = distance(average_2, joints_1[i])
         min_1_id = i
     #print(f'distance(average_1, joints_2[{i}])={distance(average_1, joints_2[i])}')
     if min_2 > distance(average_1, joints_2[i]):
         min_2 = distance(average_1, joints_2[i])
         min_2_id = i

  return min_1_id, min_2_id, min_1, min_2, average_1,average_2

def kneeDirectionCheck(Knee,Hip,Ankle):
    # point on the line (x,y)
    # (x - Ankle[0])*(y-Hip[1]) = (x-Hip[0])*(y-Ankle[1])
    # x * (y-Hip[1]) - x * (y-Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * ((y-Hip[1]) - (y-Ankle[1])) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (-Hip[1] + Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (Ankle[1] -Hip[1] + ) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])

    y = Knee[1]  
    if Ankle[1] != Hip[1]:
        x = int((Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1]))/(Ankle[1] -Hip[1]))
        if Knee[0] > x:
            return 'toLeft'
        elif Knee[0] < x:
            return 'toRight'
        else:
            return "unknow"
    else:
        return "unknow"

def elbowDireectionCheck(Knee,Hip,Ankle):
    # point on the line (x,y)
    # (x - Ankle[0])*(y-Hip[1]) = (x-Hip[0])*(y-Ankle[1])
    # x * (y-Hip[1]) - x * (y-Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * ((y-Hip[1]) - (y-Ankle[1])) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (-Hip[1] + Ankle[1]) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    # x * (Ankle[1] -Hip[1] + ) = Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1])
    x = 0
    y = Knee[1]  
    if Ankle[1] != Hip[1]:
        x = int((Ankle[0]*(y-Hip[1]) - Hip[0]*(y-Ankle[1]))/(Ankle[1] -Hip[1]))

    if x > 0:
        result = 'toLeft'
    elif x < 0:
        result = 'toRight'
    else:
        result = 'none'
    return result
def getFencerStatus_inPose(pose):

    keypoints = pose['keypoints']
    numOfPoint = int(len(keypoints)/3)
    joints = []
    for i in range(numOfPoint):
        joint = (keypoints[i*3],keypoints[i*3 + 1])
        joints.append(joint)

    LElbow = joints[7]
    RElbow = joints[8]
    LWrist = joints[9]
    RWrist = joints[10]
    LShouder = joints[10]
    RShoulder = joints[10]
    LHip = joints[15]
    RHip = joints[16]
    LKnee = joints[15]
    RKnee = joints[16]
    LAnkle = joints[15]
    RAnkle = joints[16]

    leg_D = "N"
    arm_D = "N"
    if  checkForFencer(json_obj):
        check_1 = kneeDirectionCheck(LKnee,LHip,LAnkle)
        check_2 = kneeDirectionCheck(RKnee,RHip,RAnkle)
        check_3 = elbowDireectionCheck(LElbow,LShoulder, LWrist)
        check_4 = elbowDireectionCheck(LElbow,LShoulder, LWrist)
        if  check_1 == "toLeft" and check_3: # right fencer Knee is toward left
            result = 9



    return 0
    # status 0: pose un-classified
    # status 1: pose of the fencer on the left
    # status 2: pose of the fencer on the right
    # status 3: pose in dark clothing

def getFencerStatus_inFrame(json_obj_list):
    json_list_output = []

    for pose in json_obj_list:
        fencerStatus = getFencerStatus_inPose(pose)
    #print('strp 1================')
    #sort list by box area
    index = len(json_obj_list) - 1
    while len(json_obj_list) > 0:
      max_obj_area = 0
      for i in range(len(json_obj_list)):
          json_obj = json_obj_list[i]
          box = json_obj['box']
          if box[2] * box[3] > max_obj_area:
              II = i
              max_obj_area = box[2] * box[3]
      json_list_output.append(json_obj_list[II])
def _get_center(pose):
    x = 0
    y = 0
    size = int (len(pose["keypoints"])/3)
    for i in range(size):
        x = x + pose["keypoints"][i*3]
        y = y + pose["keypoints"][i*3 + 1]

    return [x/size, y/size]
# 'pelvis', 'left_hip', 'right_hip',      # 2
# 'spine1', 'left_knee', 'right_knee',    # 5
# 'spine2', 'left_ankle', 'right_ankle',  # 8
# 'spine3', 'left_foot', 'right_foot',    # 11
# 'neck', 'left_collar', 'right_collar',  # 14
# 'jaw',                                  # 15
# 'left_shoulder', 'right_shoulder',      # 17
# 'left_elbow', 'right_elbow',            # 19
# 'left_wrist', 'right_wrist',            # 21
# 'left_thumb', 'right_thumb',            # 23
# 'head', 'left_middle', 'right_middle',  # 26
# 'left_bigtoe', 'right_bigtoe'           # 28

# "keypoints": [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
# "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", 
# "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", 
# "left_ankle", "right_ankle" ]

def getFencerPose(pose_list, img):

    if len(pose_list) < 2:
        return [],[], img
    dict_bodyLength = defaultdict()
    for pose in pose_list:  
        keyP = pose['keypoints']
        HipL = (int(keyP[11 * 3]), int(keyP[11 * 3 + 1]))
        HipR = (int(keyP[12 * 3]), int(keyP[12 * 3 + 1]))
        KneeL = (int(keyP[13 * 3]), int(keyP[13 * 3 + 1]))
        KneeR = (int(keyP[14 * 3]), int(keyP[14 * 3 + 1]))
        AnkleL = (int(keyP[15 * 3]), int(keyP[15 * 3 + 1]))
        AnkleR =(int(keyP[16 * 3]), int(keyP[16 * 3 + 1]))
        body_length_L = distance(HipL,KneeL) + distance(KneeL, AnkleL) 
        body_length_R = distance(HipR,KneeR) + distance(KneeR, AnkleR) 
        score_L = min(keyP[11*3+2],keyP[13*3+2],keyP[15*3+2])
        score_R = min(keyP[12*3+2],keyP[14*3+2],keyP[16*3+2])
        if score_L > score_R :
            body_length = body_length_L
        else:
            body_length = body_length_R
            
        dict_bodyLength[int(body_length)] = pose
    
    sorted_bodyLength_list = sorted(dict_bodyLength, reverse = True)

    img_out = img
    pose_fencer_L = []
    pose_fencer_R = []

    for body_length in sorted_bodyLength_list:
        pose = dict_bodyLength[body_length]
        keyP = pose['keypoints']

        HipL = (int(keyP[11 * 3]), int(keyP[11 * 3 + 1]))
        HipR = (int(keyP[12 * 3]), int(keyP[12 * 3 + 1]))
        KneeL = (int(keyP[13 * 3]), int(keyP[13 * 3 + 1]))
        KneeR = (int(keyP[14 * 3]), int(keyP[14 * 3 + 1]))
        AnkleL = (int(keyP[15 * 3]), int(keyP[15 * 3 + 1]))
        AnkleR =(int(keyP[16 * 3]), int(keyP[16 * 3 + 1]))
        
        # WHITE = (255, 255, 255)
        # DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, 'hl', HipL, DEFAULT_FONT, 1, WHITE, 2)
        # cv2.putText(img, 'hr', HipR, DEFAULT_FONT, 1, WHITE, 2)posesorted_dict_list
        result1 = kneeDirectionCheck(KneeL,HipL,AnkleL)
        result2 = kneeDirectionCheck(KneeR,HipR,AnkleR)
        if  result1 == "toLeft" or result2 == "toLeft":
            if pose_fencer_R == []:
                pose_fencer_R = pose
            # elif _get_center(pose)[0] > _get_center(pose_fencer_L)[0]: # fencer_L.x is smaller
            #     pose_fencer_R = pose
        elif  result1 == "toRight" or result2 == "toRight":
            if pose_fencer_L == []:
                pose_fencer_L = pose
            # elif _get_center(pose)[0] > _get_center(pose_fencer_L)[0]: # fencer_L.x is smaller
            #     pose_fencer_R = pose
    return pose_fencer_L, pose_fencer_R, img_out

def get2fencingStatus(json_obj_list, img):
    img_out = img
    json_list_output = []
    max_obj_area = 0
    #print('strp 1================')
    #sort list by box area
    index = len(json_obj_list) - 1
    while len(json_obj_list) > 0:
      max_obj_area = 0
      for i in range(len(json_obj_list)):
          json_obj = json_obj_list[i]
          box = json_obj['box']
          if box[2] * box[3] > max_obj_area:
              II = i
              max_obj_area = box[2] * box[3]
      json_list_output.append(json_obj_list[II])
      json_obj_list.pop(II)

    #print('strp 2================')
    json_obj_list = json_list_output
    json_list_output = []

    # search for the first fencer
    firstfencerFound = False
    for i in range(len(json_obj_list)):
        json_obj = json_obj_list[i]
        if checkForFencer(json_obj) == True:
            firstFencingID = i
            firstfencerFound = True
            box = json_obj['box']
            cv2.putText(img, "1st", (int(box[0]), int(box[1] + box[3])), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
            break

    #print('strp 3================')
    if firstfencerFound == False:
        img = cv2.putText(img, "NOFencer", (img.shape[0] - 100,50), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0,255), 2, cv2.LINE_AA)
        return img, 0 #

    json_list_output.append(json_obj_list[firstFencingID])
    json_obj_list.pop(firstFencingID)
    secondFencingID = False
    secondfencerFound = False
    if len(json_obj_list) > 0:
      for i in range(len(json_obj_list)):
          json_obj = json_obj_list[i]
          if checkForFencer(json_obj) == True:
              sencondFencingID = i
              secondfencerFound = True
              box = json_obj['box']
              cv2.putText(img, "2nd", (int(box[0]), int(box[1] + box[3])), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
              break

    if secondfencerFound == False:
        cv2.putText(img, "NNNNNN", (img.shape[1] - 100,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4, cv2.LINE_AA)
        return img, 0

    json_list_output.append(json_obj_list[secondFencingID])
    max_1_id, max_2_id, max_1, max_2, A1, A2 = findCloseKeypoint(json_list_output)
    cv2.circle(img, A1, 15, (0,0,255), -1)
    cv2.circle(img, A2, 15, (0,0,255), -1)

    if max_1_id == 9 or max_1_id == 10 and max_2_id == 9 or max_2_id == 10:
        img = cv2.putText(img, "FFFFFF", (img.shape[1] - 100,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4, cv2.LINE_AA)
        return img, 1

    img = cv2.putText(img, "UUUUUU", (img.shape[1] - 100,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (128,128,128), 4, cv2.LINE_AA)
    return img, -1


# def search_overLap_pose(pose_dict,frame_dict,idx1, idx2):
#     frame_dict_len = len(frame_dict)
#     _result = []
#     for i in range(frame_dict_len):
#         frame_pose = frame_dict[i]
#         no_pose_in_frame = len(frame_pose)

# def removeOverlap(frame_dict,idx1, idx2):
#     frame_dict_len = len(frame_dict)
#     _result = []
#     for i in range(frame_dict_len):
#         frame_pose = frame_dict[i]
#         no_pose_in_frame = len(frame_pose)
#         for j in range(no_pose_in_frame):
#             Jbox = frame_pose[j]['box']
#             for k in range(j+1, no_pose_in_frame):
#                 over_lap_rate = overLap_rate(Jbox, frame_pose[k]['box'])
#                 _result.append([frame_pose[0],frame_pose[j]["idx"],frame_pose[k]["idx"], over_lap_rate])

#     return _result
#     return _result




        #     if idx1 == idx:
        #         idx1_exist = True
        #     if idx2 == idx:
        #         idx2_exist = True
        # if idx1_exist and idx2_exist:
        #     print ("i = ", i, " =============  frame_id:", frame_pose[0]["image_id"] )
        #     exit()
alphaPose_resuslt_path = '/home/yin/gitSources/AlphaPose/testResults/'

input_results = os.listdir(alphaPose_resuslt_path)
image_status = {}
print(f'The folder {alphaPose_resuslt_path} has {len(input_results)} video files to be processed')
print(input_results)
for path in input_results:

   result = alphaPose_resuslt_path + path

   print(result)
   if os.path.isdir(result) :
        alphaPose_resuslt_image_path = result + "/vis_orig/"
        alphaPose_resuslt_json_name = result + "/precision_results.json"
        vis_ok = os.path.isdir(alphaPose_resuslt_image_path)
        json_ok = os.path.isfile(alphaPose_resuslt_json_name)

        numberOfImage = len(os.listdir(alphaPose_resuslt_image_path))
        padSize = len(str(numberOfImage))
        if vis_ok == False or json_ok == False:
            print(f'aphapose result {result} is not ready!')
            continue

        #filtered_video_name = result +"/filtered.avi"
        filtered_json_name = result +"/filtered.json"
        fencer_image_dir = result+"/fencer_image_dir"
        #filtered_image_dir = result+"/filtered_image_dir"
        #S1_remove_dark_image_dir = result+"/s1_image_dir"
        #S2_remove_small_image_dir = result+"/s2_image_dir"

        if  os.path.isfile(filtered_json_name) == True:
            print(f'{result} already processed!')
            continue

        if os.path.isdir(fencer_image_dir) == False:
            os.mkdir(fencer_image_dir)
        # if os.path.isdir(filtered_image_dir) == False:
        #     os.mkdir(filtered_image_dir)
        # if os.path.isdir(S1_remove_dark_image_dir) == False:
        #     os.mkdir(S1_remove_dark_image_dir)


        #print(alphaPose_resuslt_json_name)
        rawText = ""
        with open(alphaPose_resuslt_json_name,encoding = 'utf-8') as f:
            rawText = f.readline()
            #print(str1[:50])
        str2 = rawText[1:][:-1]
        str3 = str2.split('},{')
        #print(f'start to process {# Video Generating functionresult}, {len(str3)} poses and {numberOfImage} images are in this video.')
        print(f'({alphaPose_resuslt_json_name} file contains {len(str3)} poses')
        #print(str3[:50])
        # json_objs_array = []
        # json_objs_array_orig = []
        #idxes = {}
        #idx_used = 3167
        json_obj_init_list = []

        for i in range(len(str3)):
            if i == 0:
                jsonStr = str3[i]+"}"
            elif i == len(str3) -1:
                jsonStr = "{" + str3[i]
            else:
                jsonStr = "{" + str3[i] + "}"
            #print("-------")
            #print("i=",i,"    ", jsonStr, "    ++++    ", jsonStr[425:440])
            json_obj = json.loads(jsonStr)
            json_obj_init_list.append(json_obj)
        #print("json_obj_init_list:", json_obj_init_list[:20])
        # image_id, 1st frame, frame No.

        # pose_dict1 = get_pose_dict(json_obj_init_list)

        final_frame_list = []
        json_obj_dark_list = []
        json_obj_small_list = []
        json_obj_final_list = []
        json_obj_frame_list = []
        json_obj = json_obj_init_list[0]
        json_obj_frame_list.append(json_obj)
        obj_prev = json_obj
        final_overlap_list = []
        frame_dict = defaultdict()
        overlap_dict = defaultdict()
        # process pose within a frame, and another frame
        # the loop below uses json_obj_init_list to generate 1) pose_dict, a dictionary containing all the poses 
        # groupped by pose idx; 2) a frame_dict, a dictionary containing for the poses groupped by frame idx; and 
        # 3) a overlap_list, a list containing pose (box) overlap information. overlap_list is stored inside frame_dict
        # frame_dict = {frame_idx: [pose_list, overlap_list]}
        for i in range(1, len(json_obj_init_list)):
            json_obj = json_obj_init_list[i]
            if json_obj['image_id'] == obj_prev['image_id'] and i != len(json_obj_init_list) - 1:
                json_obj_frame_list.append(json_obj)
            else:
                # if i % 1000 == 0:
                #     print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')

                #process the josn_obj of the (prev) frame
                fileName = obj_prev['image_id'] #.replace(".jpg","")+"_orig.jpg"
                frame_no = int(fileName.replace(".jpg",""))
                if frame_no == 90:
                    t = 0
                #print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')
                img = cv2.imread(alphaPose_resuslt_image_path + fileName)
                img = cv2.putText(img, str(frame_no), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)

                out_list_f = removeFlat_2(json_obj_frame_list)

                out_list_s = removeSmall_2(out_list_f)
                # outF = f'{S1_remove_dark_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                # cv2.imwrite(outF, img_out)

                out_list_b= removeDarkClothing(out_list_s, img)

                overlap_list = get_Overlap_list(out_list_b)

                if len(overlap_list) > 0:
                    out_list_b, overlap_list = remove_overlap_by_size(out_list_b, overlap_list,3)

                if len(overlap_list) > 0:
                    out_list_b, overlap_list = merge_overlap_pose(out_list_b, overlap_list)
                
                if len(overlap_list) > 0:
                    overlap_dict[frame_no] = overlap_list

                #Pose_L, Pose_R, img= getFencerPose(out_list_b,img)

                #img_out = drawPose(out_list_b, img)
                json_obj_final_list = json_obj_final_list + out_list_b
                # outF = f'{fencer_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                # cv2.imwrite(outF, img_out)

                frame_list = []
                for pose in out_list_b:
                 #   frame_list.append(int(pose['idx']))
                    frame_list.append(pose)

                frame_id = obj_prev["image_id"]
                frame_dict[frame_no] = frame_list

                obj_prev = json_obj
                json_obj_frame_list = []
                json_obj_frame_list.append(obj_prev)
                if frame_no % 500 == 0:
                    print(" image frames processed: ", frame_no)

        for key, pose_list in frame_dict.items():
            #frame_pose = frame_dict[frame]
            fileName = str(key)+".jpg" #obj_prev['image_id'] #.replace(".jpg","")+"_orig.jpg"
            img = cv2.imread(alphaPose_resuslt_image_path + fileName)
            cv2.putText(img, str(key), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)

            Pose_L, Pose_R, img = getFencerPose(pose_list, img)

            if Pose_L != [] and Pose_R != []:

                box = Pose_L['box']
                p = (int(box[0]), int(box[1] + box[3]))
                cv2.putText(img, "fencer_L", p, cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
                box = Pose_R['box']
                p = (int(box[0]), int(box[1] + box[3]))
                cv2.putText(img, "fencer_R", p, cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)

            # frame_no = key
            # if frame_no == 90:
            #     t = 0
            #print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')

            outF = f'{fencer_image_dir}/img_{str(key).zfill(padSize)}.jpg'
            cv2.imwrite(outF, img)


        f = open(filtered_json_name, "w")
        json_string = json.dumps(json_obj_final_list)
        f.writelines(json_string)
        f.close()


