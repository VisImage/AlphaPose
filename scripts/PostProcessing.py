# importing libraries
import os
import cv2
from PIL import Image
import math


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


def removeDarkClothing(json_obj_list, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(img_gray)
    img_out = img
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
        #print(f'person_id:{person_id}   m    img_out = img

        if mean < local_mean:  #fencing clothing is white so the brightness value is high
            json_list_output.append(json_obj)
            cv2.polylines(img_out,[cPts.astype(int)],True,(255,0,0), 3)

    return json_list_output, img_out
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

#generate pose dictionary
def get_pose_dict(pose_list, result):
    
    frame_dict = defaultdict()
    for pose in pose_list:
        frame_id = pose["image_id"]
        if frame_id in frame_dict:
            IV = int(pose["idx"])
            frame_dict[frame_id].append(IV)
        else:
            frame_dict[frame_id] = [int(pose["idx"])]

    pose_dict = defaultdict()
    for pose in pose_list:
        image_id = int(pose['image_id'].replace('.jpg',''))
        pose_id = str(pose["idx"])     
        if pose_id == '765'  and image_id == 7 :
            ii = 99 
        if pose_id == '1153'  and image_id == 2328:
            ii = 99
        if pose_id in pose_dict: 
            pose_list = pose_dict[pose_id]
            if int(pose_id) in frame_dict[str(image_id - 1)+".jpg"]:
                pose_list[len(pose_list)-1][1] = image_id
            else:
                pose_list.append([image_id, image_id])
        else:
            pose_dict[pose_id] = [[image_id, image_id]]

        frame_dict_name = result +"/frame_dict.txt"
        with open(frame_dict_name, 'w') as f:
            print(frame_dict, file=f)

        pose_dict_name = result +"/pose_dict.txt"
        with open(pose_dict_name, 'w') as f:
            print(pose_dict, file=f)

    return pose_dict

def new_func():
    i = 9
    ###from IPython.lib.display import IFrame
# To check and verify the result from pose_track alphaPose_resuslt.json
# alphaPose_resuslt_image_path = '/gdrive/MyDrive/AlphaPose_result/20220304_133434000_iOS/vis/'
alphaPose_resuslt_path = '/home/yin/gitSources/AlphaPose/testResults/'

import json
# import cv2
import numpy as np
import sys
from collections import defaultdict

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

        filtered_video_name = result +"/filtered.avi"
        filtered_json_name = result +"/filtered.json"
        filtered_image_dir = result+"/filtered_image_dir"
        S1_remove_dark_image_dir = result+"/s1_image_dir"
        S2_remove_small_image_dir = result+"/s2_image_dir"

        if  os.path.isfile(filtered_video_name) == True and os.path.isfile(filtered_json_name) == True:
            print(f'{result} already processed!')
            continue

        if os.path.isdir(filtered_image_dir) == False:
            os.mkdir(filtered_image_dir)
        if os.path.isdir(S1_remove_dark_image_dir) == False:
            os.mkdir(S1_remove_dark_image_dir)


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

        json_obj_dark_list = []
        json_obj_small_list = []
        json_obj_final_list = []
        json_obj_frame_list = []
        json_obj = json_obj_init_list[0]
        json_obj_frame_list.append(json_obj)
        obj_prev = json_obj

        # process pose within a frame, and another frame
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
                #print(f'i={i}, img_id={frame_no}, {len(json_obj_frame_list)}')
                img = cv2.imread(alphaPose_resuslt_image_path + fileName)
                img = cv2.putText(img, str(frame_no), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)

                out_list_f, img_out = removeFlat(json_obj_frame_list, img)

                out_list_s, img_out = removeSmall(out_list_f, img_out)
                outF = f'{S1_remove_dark_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                cv2.imwrite(outF, img_out)

                out_list_b, img_out= removeDarkClothing(out_list_s, img_out)
                outF = f'{S2_remove_small_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                cv2.imwrite(outF, img_out)

                json_obj_final_list = json_obj_final_list + out_list_b
                outF = f'{filtered_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                cv2.imwrite(outF, img_out)

                obj_prev = json_obj
                json_obj_frame_list = []
                json_obj_frame_list.append(obj_prev)
                if frame_no % 500 == 0:
                    print("i=", i, "  ", outF)

        print(" length of lists:", len(json_obj_init_list), ", ", len(out_list_f), ", ", len(out_list_s),  len(out_list_b), ", ", len(json_obj_final_list))
        pose_dict = get_pose_dict(json_obj_final_list, result)

        # pose_dict_name = result +"/pose_dict.txt"

        # with open(pose_dict_name, 'w') as f:
        #     print(pose_dict, file=f)

        # print("pose_dict:",pose_dict)
                # if len(out_list_s) > 1:
                #     img_out, status = get2fencingStatus(out_list_b, img_out)
                #     image_status[f'{fileName}'] = "Allez"
                # else:
                #     status = False
                #     image_status[f'{fileName}'] = ""

                # outF = f'{filtered_image_dir}/img_{str(frame_no).zfill(padSize)}.jpg'
                # cv2.imwrite(outF, img_out)

                # #cv2.imwrite(outF, img_out)
                # obj_prev = json_obj
                # json_obj_frame_list = []
                # json_obj_frame_list.append(obj_prev)

                # ## tempary stop for developmenet
                # #print(f'img_id={no_images}, len of json_obj_final_list {len(json_obj_final_list)}')
                # if frame_no % 500 == 0:
                #     print(outF)
            #if no_images > 300:
                #break
                #sys.exit()

        #if len(json_obj_final_list) > 100:

        f = open(filtered_json_name, "w")
        json_string = json.dumps(json_obj_final_list)
        f.writelines(json_string)
        f.close()

        print(f'image_status_file_path:{result}/image_status.json')
        out_file = open( result+"/image_status.json", "w")
        json.dump(image_status, out_file, ensure_ascii = True)
        out_file.close()

        print(f'filtered_image_dir={filtered_image_dir}')
        filtered_video_name = result +"/filtered.avi"
        generate_video(filtered_image_dir,filtered_video_name)
        print(f'{filtered_video_name} is generated')
        # S1_video_name = result +"/s1.avi"
        # generate_video(S1_remove_dark_image_dir,S1_video_name)
        # print(f'{S1_video_name} is generated')
        #else:
        #print(f'video:{result} does not has enough valid frames ({len(json_obj_final_list)}<100)')

print('filtering the videos are completed.')
