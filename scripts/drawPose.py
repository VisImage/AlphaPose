# importing libraries
import os
import cv2
from PIL import Image
import math
import argparse

import json
import numpy as np
import sys
from collections import defaultdict


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def get_color_fast(idx):
    color_pool = [RED, GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, WHITE]
    color = color_pool[idx % 8]

    return color


def vis_frame_fast(frame, im_res, bbox):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    kp_num = 17
    format = 'coco'
    if len(im_res['result']) > 0:
        kp_num = len(im_res['result'][0]['keypoints']) / 3
    if kp_num == 17:
        if format == 'coco':
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                       (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                       (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                          (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                          (77, 222, 255), (255, 156, 127),
                          (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
        elif format == 'mpii':
            l_pair = [
                (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                (13, 14), (14, 15), (3, 4), (4, 5),
                (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
            ]
            p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # im_name = os.path.basename(im_res['imgname'])
    img = frame.copy()
    height, width = img.shape[:2]
    for human in im_res['result']:
        part_line = {}
        kp_preds = {}
        kp_scores = []
        keypoints = human['keypoints']
        for n in range (17):
            kp_preds[n,0] = keypoints[n*3]
            kp_preds[n,1] = keypoints[n*3 + 1]
            kp_scores.append(keypoints[n*3 + 2])
        # if kp_num == 17:
        #     kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        #     kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        #     vis_thres.append(vis_thres[-1])
        # if opt.pose_track or opt.tracking:
        #     while isinstance(human['idx'], list):
        #         human['idx'].sort()
        #         human['idx'] = human['idx'][0]
        color = get_color(int(abs(human['idx'])))
        # else:
        #     color = BLUE

        # Draw bboxes
        if bbox:
            if 'box' in human.keys():
                bbox = human['box']
                bbox = [bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]]#xmin,xmax,ymin,ymax
            else:
                from trackers.PoseFlow.poseflow_infer import get_box
                keypoints = []
                for n in range(kp_scores.shape[0]):
                    keypoints.append(float(kp_preds[n, 0]))
                    keypoints.append(float(kp_preds[n, 1]))
                    keypoints.append(float(kp_scores[n]))
                bbox = get_box(keypoints, height, width)
            
            cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), color, 2)
            #if opt.tracking:
            cv2.putText(img, str(human['idx']), (int(bbox[0]), int((bbox[2] + 26))), DEFAULT_FONT, 1, RED, 2)
        # Draw keypoints
        #for n in range(kp_scores.shape[0]):
        for n in range(17):
        #     if kp_scores[n] <= vis_thres[n]:
        #         continue

            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            if n < len(p_color):
                #if opt.tracking:
                if True:
                    cv2.circle(img, (cor_x, cor_y), 3, color, -1)
                else:
                    cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
            else:
                cv2.circle(img, (cor_x, cor_y), 1, (255,255,255), 2)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    if True: #opt.tracking:
                        cv2.line(img, start_xy, end_xy, color, 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                    else:
                        cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                else:
                    cv2.line(img, start_xy, end_xy, (255,255,255), 1)

    return img



parser = argparse.ArgumentParser(description='Pose proessing')

parser.add_argument('--result_dir', type=str, required=False, default = "/home/yin/gitSources/AlphaPose/testResults/alphaPose_sample1",
                    help='result path from inference including a json file and original image frame')
parser.add_argument('--bbox', default=True, action='store_true',
                    help='draw bonding box')
parser.add_argument('--joint', default=True, action='store_true',
                    help='draw joint (keypoint)')
parser.add_argument('--save_img', default=True, action='store_true',
                    help='save result as image')
parser.add_argument('--save_video', default=True, action='store_true',
                    help='save final video file')

args = parser.parse_args()


print("args=",args)
    # # _result = []
    # # for k in range(len(scores)):
    # #     _result.append(
    # #         {
    # #             'keypoints':preds_img[k],
    # #             'kp_score':preds_scores[k],
    # #             'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
    # #             'idx':ids[k],
    # #             'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
    # #         }
    # #     )

    # # result = {
    # #     'imgname': im_name,
    # #     'result': _result
    # # }


def drawbbox(img, result, color, lineWidth, pose_id = False, fontScale = 1.0):

    for human in result['result']:
        box = human['box']
        p1 = [box[0],box[1]]
        p2 = [box[0],box[1]+box[3]]
        p3 = [box[0] + box[2],box[1]+box[3]]
        p4 = [box[0] + box[2],box[1]]
        pts = np.array([p1,p2,p3,p4], np.int32)

    return img

if __name__ == "__main__":

    result = args.result_dir
    print("result_path = ",result)


    frame = 0
    if os.path.isdir(result):
        alphaPose_resuslt_image_path = result + "/vis_orig/"
        alphaPose_resuslt_json_name = result + "/precision_results.json" 
        vis_ok = os.path.isdir(alphaPose_resuslt_image_path)
        json_ok = os.path.isfile(alphaPose_resuslt_json_name)

        print("alphaPose_resuslt_image_path=",alphaPose_resuslt_image_path)
        print("alphaPose_resuslt_json_name=",alphaPose_resuslt_json_name)
        
        if vis_ok == False or json_ok == False:
            print(f'aphapose result {result} is not ready! exit program')
            exit()
        input_images = os.listdir(alphaPose_resuslt_image_path)
        frame = cv2.imread(alphaPose_resuslt_image_path+input_images[0])
    else: 
        print(f' path {result} does not exist! exit program')
        exit()


    numberOfImage = len(os.listdir(alphaPose_resuslt_image_path))
    padSize = len(str(numberOfImage))

    filtered_video_name = ""
    video = 1
    if args.save_video:
        filtered_video_name = result +"/result.avi"
        #initial video generatiion
        height, width, layers = frame.shape
        print(f'h={height}; w={width}')
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(filtered_video_name, fourcc, 30, (width, height))

    
    result_image_dir = ""
    if args.save_img:
        result_image_dir = result+"/result_image_dir/"

        print("result_image_dir",result_image_dir)
        if os.path.isdir(result_image_dir) == False:
            os.mkdir(result_image_dir)

    if True:
        rawText = ""
        with open(alphaPose_resuslt_json_name,encoding = 'utf-8') as f:
            rawText = f.readline()
            #print(str1[:50])
        str2 = rawText[1:][:-1]
        str3 = str2.split('},{')
        #print(f'start to process {# Video Generating functionresult}, {len(str3)} poses and {numberOfImage} images are in this video.')
        print(f'({alphaPose_resuslt_json_name} file contains {len(str3)} poses')

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


        _result = []
        obj_prev = json_obj_init_list[0]
        for i in range(1, len(json_obj_init_list)):
            json_obj = json_obj_init_list[i]
            if json_obj['image_id'] == obj_prev['image_id'] and i != len(json_obj_init_list) - 1:
                _result.append(
                {
                    'keypoints':json_obj['keypoints'],
                    'kp_score':json_obj['score'],
                    'proposal_score': 0,
                    'idx':json_obj['idx'],
                    'box':json_obj['box'],
                }
            )
            else:
                #process the josn_obj of the (prev) frame
                fileName = obj_prev['image_id'] #.replace(".jpg","")+"_orig.jpg"
                frame_no = int(fileName.replace(".jpg",""))
                #print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')
                img = cv2.imread(alphaPose_resuslt_image_path + fileName)
                img = cv2.putText(img, str(frame_no), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)

                result = {
                    'imgname': fileName,
                    'result': _result
                }

                # if args.bbox:
                #     color = { 255,0,0}
                #     lineWidth = 2
                #     img = drawbbox(img, result, color, lineWidth, True, 1)

                if args.joint:
                    opt = 0.5
                    vis_thres = 0.5
                    if args.bbox:
                        img = vis_frame_fast(img, result, True)
                    else:
                        img = vis_frame_fast(img, result, False)
                
                if args.save_img:
                    outF = f'{result_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                    cv2.imwrite(outF, img)

                if args.save_video:
                    video.write(img)

                _result = []
                obj_prev = json_obj
                if frame_no % 500 == 0:
                    print("i=", i, "  ", fileName)

        if args.save_video:
             video.release()  # releasing the video generated
             print(f'{filtered_video_name} is generated')
