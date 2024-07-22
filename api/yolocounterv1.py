import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple

class YoloOnnx:
    def __init__(self, weigths_path, class_names, cuda = False):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weigths_path, providers=providers)
        self.class_names = class_names
        self.colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(class_names)}

    def inference(self, img_path):        
        img = Image.open(img_path)
        if img.mode != 'RGB': 
            img = img.convert('RGB')
        img = np.asarray(img)
        image = img.copy()
        image, self.ratio, self.dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]:im}
        outputs = self.session.run(outname, inp)[0]
        c_classes = self.counting(outputs)
        return img, outputs, c_classes
    
    def counting(self, outputs):
        countings = {self.class_names[int(cls_id)]:0 for _, _, _, _, _, cls_id,_ in outputs}  #(batch_id,x0,y0,x1,y1,cls_id,score)
        for _, _, _, _, _, cls_id,_ in outputs:
            countings[self.class_names[int(cls_id)]] += 1  
            
        countings_list = list(countings.items())
        countings_list.sort(key = lambda x: x[1], reverse=True)
        countings = {c:n for (c,n) in countings_list}
        return countings
        
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)
    
    def visualize_detections(self, img, outputs): 
        ori_images = [img.copy()]
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = self.convertbox([x0,y0,x1,y1])
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = self.class_names[cls_id]
            color = self.colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
        return ori_images[0]
    
    def convertbox(self, box0):
        box = np.array(box0)
        box -= np.array(self.dwdh*2)
        box /= self.ratio
        box = box.round().astype(np.int32).tolist()
        return box 
