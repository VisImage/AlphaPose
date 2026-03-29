# Fencing AlphaPose

This fork of AlphaPose contains custom scripts and post-processing tools specifically designed to track and extract the poses of two fencers in video or image sequences.

## 1. Installation & Environment Setup

### System Requirements
*   **OS:** Ubuntu 20.04.5
*   **GPU Driver:** NVIDIA Driver 470 (`Desktop > Software & Updates > Additional Drivers > nvidia-driver-470`)
*   **CUDA Toolkit:** CUDA 11.3 (Download Link)

### Setup Process
1. Clone the repository:
   ```bash
   git clone https://github.com/VisImage/AlphaPose.git
   cd AlphaPose
   ```
2. Follow the standard AlphaPose installation instructions in docs/INSTALL.md to set up the Conda environment and PyTorch dependencies. Ensure you activate the `alphapose` environment when done.
   *(Note: Depending on your exact environment setup, slight modifications may be required during the `python setup.py` step).*

3. Download and ensure the required weights are placed into the following directories:
   * `detector/yolo/data/`
   * `detector/yolox/data/`
   * `pretrained_models/`
   * `tracker/weights/`

---

## 2. Usage

You can process fencing data either from a directory of videos or a directory of extracted images. Ensure your Conda environment is active (`conda activate alphapose`) and you are in the root `AlphaPose` directory.

### Option A: Process a Video Directory
This method processes video files (e.g., `.avi` or `.mp4`) located in the `testVideos/` folder and outputs the tracking of the two fencers into `testResults/`.

1. **Run the initial video inference with pose tracking:**
   ```bash
   python scripts/fencingVideo_inference.py \
       --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
       --checkpoint pretrained_models/fast_res50_256x192.pth \
       --save_img \
       --pose_track
   ```
   *(This will process all videos in `testVideos/` and save the outputs to `testResults/`)*

2. **Compress the output results:**
   ```bash
   python scripts/pose_precision_json.py --inDir testResults
   ```
   *(This compresses `alphapose-results.json` into a smaller `precision_results.json`)*

3. **Run post-processing to trace the 2 fencers:**
   ```bash
   python scripts/PostProcessing.py
   ```
   *(This script extracts the 2 fencers fencing on the strip from the images in `testResults/`)*

### Option B: Process an Image Directory
If you have a sequence of images (e.g., in `examples/fencing_images/`), you can use this method. 

*Note: Images should be named as sequential numbers, following the convention of frames sampled from a video. This is useful for testing a variety of different environments. The `--pose_track` flag is heavily relied upon for returning a reasonable identity value (idx) which is used in the fencer detection algorithm.*

1. **Run the image inference with pose tracking:**
   ```bash
   python scripts/demo_inference.py \
       --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
       --checkpoint pretrained_models/fast_res50_256x192.pth \
       --indir examples/fencing_images/ \
       --outdir examples/res \
       --save_img \
       --pose_track
   ```
   *(Alternatively, you can run this via `.vscode/launch.json` if using VS Code)*

2. **Run the image directory post-processing script:**
   ```bash
   python scripts/PP_ImgDir.py
   ```
   *(This script will isolate the 2 fencers from the processed images in `examples/res/`. The fencer images are placed in a `fencer_image_dir/` and their poses are stored in `filtered.json`)*
   
   *Note: Precision adjustments and json compression are automatically integrated within `PP_ImgDir.py`, so a separate compression step is not needed.*

---

## 3. Known Issues & Notes
* **Optical Flow Acceleration:** The optical flow calculations inside `PostProcessing.py` and `PP_ImgDir.py` are executed on the CPU via OpenCV. Attempts to configure optical flow processing to use CUDA have broken the AlphaPose installation and dependencies in the past, so CPU processing is currently required.