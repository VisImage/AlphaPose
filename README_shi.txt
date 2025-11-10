ReadME_Yin

INSTALLATION:

ubuntnu 20.04.5 from flashh drive
nvidia drive 470: desktop -> software & update > additional Driver n> nvidia 470
cuda-toollits: https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

follow the instruction in Download Installer for Linux Ubuntu 20.04 x86_64

copy AlphaPose folder from  008a648c-6902-4612-88db-e918639d2875/home/yin/gitSources/AlphaPose 

Follow installation instruction from alphapose github, except git clone part.

There are a little bit twicking needed fduring or python setup.py

Verification:

(alphapose) yin@yin-Dell: ~/gitSources/AlphaPose$ python scripts/fencingVideo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --save_img --pose_track

(alphapose) yin@yin-Dell:~/gitSources/AlphaPose$ python scripts/pose_precision_json.py  --inDir testResults

(alphapose) yin@yin-Dell:~/gitSources/AlphaPose$ python scripts/PostProcessing.py
The optical flow is calculated inside PostProcessing.py using cpu (opencv) effort to use cuda has break the alphapose installation :-(


==================== Nov. 10, 2025 ====================
############  Step 1: setup 
1) git clone https://github.com/VisImage/AlphaPose.git
2) update the following data:
    detector/yolo/data/
    detector/yolox/data/
    pretrained_models/
    racker/weights/

############  Step 2: process video dir 

(alphapose) cd AlphaPose
(alphapose) yin@yin-Dell: VisImage/AlphaPose$ python scripts/fencingVideo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --save_img --pose_track
        // 4 video files in testVideos are processed and the result are saved in testResults
(alphapose) yin@yin-Dell: VisImage/AlphaPose$ python scripts/pose_precision_json.py  --inDir testResults
        // alphapose-results.json is compressed and the smaller file is precesion_results.json
(alphapose) yin@yin-Dell: VisImage/AlphaPose$ python scripts/PostProcessing.py
        // 2 fencers, fencing on the strip, are detected from the images in the folder testResults

############  Step 3: process image dir 
#
#       images in the folder can come from different enviroment and easier to test veriaty situations.. --post_track is needed for a reasonable idx value, which is used in the fencer detection algorithm afterward. The images 
#   are named as numbers, following the convention of frames sampled from a video. 
#
#

(alphapose) cd AlphaPose
(alphapose) yin@yin-Dell: VisImage/AlphaPose$ python scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/fencing_images/ --outdir examples/res --save_img --pose_track
        // or to use .vscode/launch.json from MS code
(alphapose) yin@yin-Dell: VisImage/AlphaPose$ python scripts/PP_ImgDir.py
        // 2 fencers, fencing on the strip, are detected from the images in the folder examples/res. the fencer imgage are in fencer_image_dir, the pose is stored in  filtered.json
        // precision adjustment is integrated inside PP_ImgDir.py