import urllib.request
from flask import Flask, jsonify, request

### Load the models' paths
local_dir = "../"
cloud_dir = "https://yolos3bucket.s3.us-east-2.amazonaws.com/"
s3_files = {'yolopath': "yolov7_training.onnx",}
for k, v in s3_files.items():
    try:# Local files
        s3_files[k] = local_dir + v
        open(s3_files[k])
        print('Loading the local file:', v)
    except: # Cloud files
        s3_files[k] = '/tmp/' + v
        urllib.request.urlretrieve(cloud_dir+v, s3_files[k])

### Load recognition system
from yolocounterv1 import YoloOnnx
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
yolo = YoloOnnx(weigths_path = s3_files['yolopath'], 
                class_names = class_names, 
                cuda = False)

### EB looks for an 'application' callable by default.
application = Flask(__name__)

@application.route('/')
def index():
    return "Hello, I'm ready to count objects using Yolo"

@application.route('/detect&count', methods=['POST'])
def predict():
    ### Inference
    _, outputs, c_classes = yolo.inference(request.files['image'])
    ### Prepare data for serialization
    detections = [(yolo.convertbox([x0,y0,x1,y1]), int(y_pred), str(y_prob), yolo.class_names[int(y_pred)]) for (batch_id,x0,y0,x1,y1,y_pred,y_prob) in outputs]
    return jsonify(countings = c_classes,
                   detections = detections)

if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    application.run()
