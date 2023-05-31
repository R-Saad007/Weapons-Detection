# Weapons Detector (work in progress)
## Need to clone YOLO repository:
```git clone https://github.com/ultralytics/yolov5.git```

```pip install -r requirements.txt```
## Test videos have been added from YouTube due to the temporary unavailability of the custom dataset requirements
- vid1.mp4
- vid2.mp4
## Execution:
### YOLOv8:
Weight file has been added as well (yolov8n.pt)

```python handler.py -vid_path```
### YOLOv5:
Weight file (yolov5m_Objects365.pt) can be downloaded from this link:  
https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m_Objects365.pt  
```python handler_yolov5.py```
