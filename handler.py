import ultralytics
import torch
import cv2
import argparse
import time
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# yolov8 handler for weapons detection in videos
# handler class
class handler():
    def __init__(self, vid):
        self.model = None                       # yolov8 model
        self.vid_path = vid                     # video path

    def load_model(self):
        # loading yolov8 model for inference
        self.model = ultralytics.YOLO(f'yolov8n.pt')
        ultralytics.checks()
        # shifting model to GPU
        self.model.to(device)
        # training model on Objects365 dataset
        self.model.train(data="Objects365.yaml") 
        
    def frame_processing(self):
        print("Starting Video Processing...")
        # fps calculators
        prev_time = 0.0
        new_frame_time = 0.0
        font = cv2.FONT_HERSHEY_SIMPLEX
        # video capture object
        cap = cv2.VideoCapture(self.vid_path)
        frame_count = 0
        # video processing loop
        while cap.isOpened():
            # capturing first frame to check whether video exists for processing below
            ret, frame = cap.read()
            # if no frame exists, simply end the capturing process
            if not ret:
                break
            new_frame_time = time.time()
            frame_count +=1
            # inference
            results = self.model(frame, classes=[84, 147])  # using the Knife and Gun classes only classes = [84,147]
            frame = results[0].plot()   # rendering results on the frame
            fps = 'FPS: ' + str(int(1/(new_frame_time-prev_time)))
            # FPS text
            cv2.putText(frame, fps, (7, 50), font, 2, (0, 255, 255), 5, cv2.LINE_AA)
            # frame = cv2.resize(frame, (1080,720))
            cv2.imshow("YOLOv8 Output" , frame)
            '''Remove the statement below(only for testing purposes)'''
            # cv2.imwrite('frame.jpg',frame)
            prev_time = new_frame_time
            cv2.waitKey(1)
        # release the video capture object
        cap.release()
        # Closes all the windows currently opened.
        cv2.destroyAllWindows()
        print("Video Processing Completed!")
        print("Inferencing Completed!")
        print('Number of frames: ', frame_count)

    def __del__(self):
        # object destructor
        self.model = None                                                   # yolov8 path
        self.vid_path = None                                                # video path
        print("Handler destructor invoked!")

# main function
if __name__ == '__main__':
    # Argument from CLI
    parser = argparse.ArgumentParser(description = 'I/O file paths required.')
    parser.add_argument('-vid_path', type = str, dest = 'vid_path', required =True)
    args = parser.parse_args()

    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # whatever you are timing goes here
    vid_handler = handler(args.vid_path)
    vid_handler.load_model()
    vid_handler.frame_processing()
    del vid_handler
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds