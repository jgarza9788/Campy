import os,time
import json5 as json
from datetime import datetime
import cv2
import pandas as pd
from collections import Counter
import numpy as np
import torch

import shutil

import logging 
from logging import Logger
from logging.handlers import RotatingFileHandler

# info on YOLO
#https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#load-yolov5-with-pytorch-hub
#https://docs.ultralytics.com/yolov5/
#https://github.com/ultralytics/yolov5

# this is used to convert classes into numbers
classes_dict = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'boat': 8,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'bench': 13,
    'bird': 14,
    'cat': 15,
    'dog': 16,
    'horse': 17,
    'sheep': 18,
    'cow': 19,
    'elephant': 20,
    'bear': 21,
    'zebra': 22,
    'giraffe': 23,
    'backpack': 24,
    'umbrella': 25,
    'handbag': 26,
    'tie': 27,
    'suitcase': 28,
    'frisbee': 29,
    'skis': 30,
    'snowboard': 31,
    'sports ball': 32,
    'kite': 33,
    'baseball bat': 34,
    'baseball glove': 35,
    'skateboard': 36,
    'surfboard': 37,
    'tennis racket': 38,
    'bottle': 39,
    'wine glass': 40,
    'cup': 41,
    'fork': 42,
    'knife': 43,
    'spoon': 44,
    'bowl': 45,
    'banana': 46,
    'apple': 47,
    'sandwich': 48,
    'orange': 49,
    'broccoli': 50,
    'carrot': 51,
    'hot dog': 52,
    'pizza': 53,
    'donut': 54,
    'cake': 55,
    'chair': 56,
    'couch': 57,
    'potted plant': 58,
    'bed': 59,
    'dining table': 60,
    'toilet': 61,
    'tv': 62,
    'laptop': 63,
    'mouse': 64,
    'remote': 65,
    'keyboard': 66,
    'cell phone': 67,
    'microwave': 68,
    'oven': 69,
    'toaster': 70,
    'sink': 71,
    'refrigerator': 72,
    'book': 73,
    'clock': 74,
    'vase': 75,
    'scissors': 76,
    'teddy bear': 77,
    'hair drier': 78,
    'toothbrush': 79
    }

class Campy():
    """
    this class connects to the security cameras, finds objects, and saves the frames
    """
    def __init__(
            self,
            cam_nums=['1','2','3','4'],
            classes = [0,1,2,3,4,5,6,7,8,14,15,16],
            min_confidence=0.50,
            model_size='m',
            interval=1.0,
            history_size:int = 120,
            demo=False
            ):
        """creates a campy object

        Args:
            cam_nums (list, optional): Defaults to ['1','2','3','4'].
            classes (list, optional): Defaults to [0,1,2,3,4,5,6,7,8,14,15,16], see list above
            min_confidence (float, optional): Defaults to 0.40. (accuracy)
            model_size (str, optional): Defaults to 'm', should be n,s,m,l, or x.
            interval (float, optional): 
            demo (bool, optional): Defaults to False., used for demo only
        """
        self.url_template = 'rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={cam}&subtype=0'
        self.DIR = os.path.dirname(os.path.realpath(__file__))

        # logging stuff
        self.log = logging.getLogger()
        self.log.setLevel(logging.DEBUG)
        log_folder = os.path.join(self.DIR,'log')
        if os.path.exists(log_folder) == False:
            os.mkdir(log_folder)
        fh = RotatingFileHandler(filename=os.path.join(log_folder,'log'),encoding='utf-8',maxBytes=1024*1024*1024*1,backupCount=100)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
            )
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

        self.log.info('campy - started')
        self.log.info(f'{cam_nums=}')
        self.log.info(f'{classes=}')
        self.log.info(f'{min_confidence=}')
        self.log.info(f'{model_size=}')
        self.log.info(f'{interval=}')
        self.log.info(f'{history_size=}')
        self.log.info(f'{demo=}')

        self.interval = interval
        self.history_size = history_size

        # convert strings to ints
        # print(classes)
        for index,c in enumerate(classes):
            if isinstance(c,str):
                classes[index] = classes_dict[c]
        # print(classes)

        self.log.info( 'classes (named)' + ','.join([k for k,v in classes_dict.items() if v in classes]) )

        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

        model_name = f"yolov5{model_size}"

        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.conf = min_confidence
        self.model.classes = classes

        # # self.classes = classes
        # # self.min_confidence = min_confidence

        s = self.load(os.path.join(self.DIR,'secret.json'))
        self.user = s['user']
        self.password = s['password']
        self.ip = s['ip']
        self.port = s['port']

        self.cam_nums = cam_nums
        self.cams = self.get_cams()

        # this will be used to store a history of object counts for each camera
        self.object_count_history = self.load(os.path.join(self.DIR,'och.json'))
        if self.object_count_history == {}:
            for c in self.cam_nums:
                self.object_count_history[c] = [] * self.history_size
            
        if demo:
            self.demo()
        else:
            self.run_loop()

    def run_loop(self):
        """
        runs an infinate loop to monitor the cameras
        """
        self.log.info('starting')
        n = 0 
        while True:

            n += 1
            if n == 120:
                n = 0
                self.save(
                    os.path.join(self.DIR,'och.json'),
                    self.object_count_history
                    )
                
                # clear runs
                shutil.rmtree(os.path.join(self.DIR,'runs'))

            for k in self.cams.keys():
                self.log.info(f'{k=} | {self.object_count_history[k]=}')

                if self.cams[k].isOpened():
                    ret,frame = self.cams[k].read()

                    if not ret:
                        self.log.warning('not ret')
                        self.cams[k] = self.get_cam(k)
                        continue

                    # print(k)
                    # print(frame.shape)

                    results = self.model(frame)                    

                    rdf = results.pandas().xyxy[0]
                    
                    records = rdf.to_dict(orient='records')
                    print(records)
                    
                    rlist = sorted([r['name'] for r in records])
                    
                    # # update the history for this camera
                    self.object_count_history[k].append(rlist)
                    self.object_count_history[k] = self.object_count_history[k][-self.history_size:]

                    # compare the number of objects with 
                    # 1 sec ago, 15 sec ago, and 30 sec ago
                    for m in [-1,0,30]:
                        try:
                            if self.object_count_history[k][m] != rlist:
                                self.log.info(results.__repr__())
                                for r in records:
                                    self.log.info(str(r))
                                self.save_frames(frame,k,results)
                                break
                        except:
                            pass


                else:
                    self.log.warning(f'{k} cam is not open')
            time.sleep(self.interval)

    def demo(self):
        """
        used to demo this class
        """

        self.log.info('starting')

        for k in self.cams.keys():
            if self.cams[k].isOpened():
                ret,frame = self.cams[k].read()

                if not ret:
                    break

                print(k)
                print(frame.shape)

                results = self.model(frame)

                rdf = results.pandas().xyxy[0]

                # x = rdf['name'].value_counts()
                # print(x,type(x))
                # print(x.tolist())


                # crdf =rdf[['name']].apply(pd.value_counts).fillna(0) 
                # print(crdf,type(crdf))
                # print(crdf.to_dict(orient='records'))
                    
                records = rdf[['name']].to_dict(orient='records')
                print(records)

                
                l = []
                for r in records:
                    l.append(r['name'])
                print(list(Counter(l).most_common()))
                
                # results.print()
                # results.show()
                # cv2.imwrite(self.get_filename(['unmarked',k]),frame)
                # results.save()
                # cv2.imwrite(self.get_filename(['marked',k]),frame)

                self.save_frames(frame,k,results)

        self.log.info('done')

    def save_frames(self,frame:np.ndarray,cam_num:str,results):
        filename = self.get_filename(['u',cam_num])
        cv2.imwrite(filename,frame)
        self.log.info(f'saved: {filename}')

        h = frame.shape[0]
        w = frame.shape[1]
        miniframe = cv2.resize(frame, (int(w/2), int(h/2)) )
        filename = self.get_filename(['s',cam_num])
        cv2.imwrite(filename,miniframe)
        self.log.info(f'saved: {filename}')

        results.save()
        filename = self.get_filename(['m',cam_num])
        cv2.imwrite(filename,frame)
        self.log.info(f'saved: {filename}')

    def get_filename(self,prefixes:list[str]):
        """returns a file name to save the image

        Args:
            cam (str): the camera number as a string

        Returns:
            str: the file name for the new file
        """
        fn = '_'.join(prefixes) + '_' + datetime.now().strftime('%Y%m%d%H%M_%S') + '.png'

        folder = os.path.join(self.DIR,'frames')
        if os.path.exists(folder) == False:
            os.mkdir(folder)

        return  os.path.join(folder,fn)

    def get_cam(self,c:str):
        """returns a new cam connection

        Args:
            c (str): cam_numer

        Returns:
            _type_: a rtsp connection
        """
        url = self.url_template.format(
            user=self.user,
            password=self.password,
            ip=self.ip,
            port=self.port,
            cam = c
            )
        return cv2.VideoCapture(url)

    def get_cams(self):
        """returns a dict of cameras

        Returns:
            dict: a dict of cameras using the camera num as keys
        """
        cams = {}
        for c in self.cam_nums:
            url = self.url_template.format(
                user=self.user,
                password=self.password,
                ip=self.ip,
                port=self.port,
                cam = c
                )
            cams[c] = cv2.VideoCapture(url)
        
        self.log.info( f'{str(len(cams.keys()))} cameras' )
        return cams

    def load(self, filepath,default={}):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            # Handle the case when the file is not found
            self.log.warning(f"Error: File '{filepath}' not found.")
            return default
        except Exception as e:
            # Handle any other unexpected errors
            self.log.warning(f"Error: An unexpected error occurred while loading '{filepath}': {e}")
            return default

    def save(self,filepath, data):
        with open(filepath,'w') as file:
            file.write(json.dumps(data,indent=4))

# ## this is base code for the whole thing
# def display_rtsp_stream(rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url)

#     # if cap.isOpened():
#     #     ret, frame = cap.read()

#     #     if not ret:
#     #         return

#     #     print(type(frame))
#     #     print(frame)

#     #     cv2.imshow("RTSP Stream", frame)


#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             print("Failed to receive frames from the stream.")
#             break

#         cv2.imshow("RTSP Stream", frame)

#         # Exit the loop and close the window if 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



if __name__ == "__main__":
    # display_rtsp_stream('rtsp://admin:SecCams3832@192.168.1.215:554/cam/realmonitor?channel=4&subtype=0')


    # print([k for k,v in classes_dict.items() if v in [1,2,3]])

    # campy = Campy(demo=True)
    campy = Campy(demo=False)

    # x = [0,1,2,3,4,5,6,7,8]
    # x.append(9)
    # x = x[5:]
    # print(x)



