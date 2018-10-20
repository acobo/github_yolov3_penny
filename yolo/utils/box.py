# -*- coding: utf-8 -*-

import numpy as np
import cv2


def correct_yolo_boxes(boxes, image_h, image_w):
    """
    # Args
        boxes : array, shape of (N, 4)
            [0, 1]-scaled box
    # Returns
        boxes : array shape of (N, 4)
            ([0, image_h], [0, image_w]) - scaled box
    """
    for i in range(len(boxes)):

        boxes[i].x = int(boxes[i].x * image_w)
        boxes[i].w = int(boxes[i].w * image_w)
        boxes[i].y = int(boxes[i].y * image_h)
        boxes[i].h = int(boxes[i].h * image_h)


# Todo : BoundBox & its related method extraction
class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

    def get_label(self):
        return np.argmax(self.classes)
    
    def get_score(self):
        return self.classes[self.get_label()]
    
    def iou(self, bound_box):
        b1 = self.as_centroid()
        b2 = bound_box.as_centroid()
        return centroid_box_iou(b1, b2)

    def as_centroid(self):
        return np.array([self.x, self.y, self.w, self.h])

    def as_minmax(self):
        centroid_boxes = self.as_centroid().reshape(-1,4)
        minmax_box = to_minmax(centroid_boxes)[0]
        return minmax_box
    

def boxes_to_array(bound_boxes):
    """
    # Args
        boxes : list of BoundBox instances
    
    # Returns
        centroid_boxes : (N, 4)
        probs : (N, nb_classes)
    """
    centroid_boxes = []
    probs = []
    for box in bound_boxes:
        centroid_boxes.append([box.x, box.y, box.w, box.h])
        probs.append(box.classes)
    return np.array(centroid_boxes), np.array(probs)


def nms_boxes(boxes, nms_threshold=0.3, obj_threshold=0.3):
    """
    # Args
        boxes : list of BoundBox
    
    # Returns
        boxes : list of BoundBox
            non maximum supressed BoundBox instances
    """
    # suppress non-maximal boxes
    n_classes = len(boxes[0].classes)
    for c in range(n_classes):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if boxes[index_i].iou(boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    return boxes


def draw_boxes(image, boxes, labels, obj_thresh=0.0, desired_size=None):
    
    def _set_scale_factor():
        if desired_size:
            img_size = min(image.shape[:2])
            if img_size < desired_size:
                scale_factor = float(desired_size) / img_size
            else:
                scale_factor = 1.0
        else:
            scale_factor = 1.0
        return scale_factor
    
    scale_factor = _set_scale_factor()
    h, w = image.shape[:2]
    img_scaled = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    
    for box in boxes:
        label = np.argmax(box.classes)
        label_str = labels[label]
        if box.classes[label] > obj_thresh:
            print(label_str + ': ' + str(box.classes[label]*100) + '%')
                
            # Todo: check this code
            if img_scaled.dtype == np.uint8:
                img_scaled = img_scaled.astype(np.int32)
            x1, y1, x2, y2 = (box.as_minmax() * scale_factor).astype(np.int32)
            
            cv2.rectangle(img_scaled, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(img_scaled, 
                        "{}:  {:.2f}".format(label_str, box.get_score()),
                        (x1, y1 - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * img_scaled.shape[0], 
                        (0,255,0), 2)
    return img_scaled      


def centroid_box_iou(box1, box2):
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
    
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
    
    _, _, w1, h1 = box1.reshape(-1,)
    _, _, w2, h2 = box2.reshape(-1,)
    x1_min, y1_min, x1_max, y1_max = to_minmax(box1.reshape(-1,4)).reshape(-1,)
    x2_min, y2_min, x2_max, y2_max = to_minmax(box2.reshape(-1,4)).reshape(-1,)
            
    intersect_w = _interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = _interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    intersect = intersect_w * intersect_h
    union = w1 * h1 + w2 * h2 - intersect
    
    return float(intersect) / union


def to_centroid(minmax_boxes):
    """
    minmax_boxes : (N, 4)
    """
    minmax_boxes = minmax_boxes.astype(np.float)
    centroid_boxes = np.zeros_like(minmax_boxes)
    
    x1 = minmax_boxes[:,0]
    y1 = minmax_boxes[:,1]
    x2 = minmax_boxes[:,2]
    y2 = minmax_boxes[:,3]
    
    centroid_boxes[:,0] = (x1 + x2) / 2
    centroid_boxes[:,1] = (y1 + y2) / 2
    centroid_boxes[:,2] = x2 - x1
    centroid_boxes[:,3] = y2 - y1
    return centroid_boxes

def to_minmax(centroid_boxes):
    centroid_boxes = centroid_boxes.astype(np.float)
    minmax_boxes = np.zeros_like(centroid_boxes)
    
    cx = centroid_boxes[:,0]
    cy = centroid_boxes[:,1]
    w = centroid_boxes[:,2]
    h = centroid_boxes[:,3]
    
    minmax_boxes[:,0] = cx - w/2
    minmax_boxes[:,1] = cy - h/2
    minmax_boxes[:,2] = cx + w/2
    minmax_boxes[:,3] = cy + h/2
    return minmax_boxes

def create_anchor_boxes(anchors):
    """
    # Args
        anchors : list of floats
    # Returns
        boxes : array, shape of (len(anchors)/2, 4)
            centroid-type
    """
    boxes = []
    n_boxes = int(len(anchors)/2)
    for i in range(n_boxes):
        boxes.append(np.array([0, 0, anchors[2*i], anchors[2*i+1]]))
    return np.array(boxes)

def find_match_box(centroid_box, centroid_boxes):
    """Find the index of the boxes with the largest overlap among the N-boxes.

    # Args
        box : array, shape of (1, 4)
        boxes : array, shape of (N, 4)
    
    # Return
        match_index : int
    """
    match_index = -1
    max_iou     = -1
    
    for i, box in enumerate(centroid_boxes):
        iou = centroid_box_iou(centroid_box, box)
        
        if max_iou < iou:
            match_index = i
            max_iou     = iou
    return match_index



if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()
    import os
    from yolo import PROJECT_ROOT, COCO_ANCHORS
    from yolo.net.yolonet import Yolonet
    import matplotlib.pyplot as plt
    from yolo.frontend import YoloDetector
    # 1. Define Parameters
    
    # Pretrained weight file is at https://pjreddie.com/media/files/yolov3.weights
    # Download darknet weight file and locate it to PROJECT ROOT
    YOLOV3_WEIGHTS = os.path.join(PROJECT_ROOT, "yolov3.weights")
    LABELS = ["1", "2", "3", "9"]

    # 2. create model & load darknet pretrained weighted file
    model = Yolonet(n_classes=len(LABELS))
    model.load_darknet_params(YOLOV3_WEIGHTS, True)
    # 6. Load test images
    DEFAULT_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "samples", "svhn", "imgs")
    
    img_files = [os.path.join(DEFAULT_IMAGE_FOLDER, "1.png"), os.path.join(DEFAULT_IMAGE_FOLDER, "2.png")]
    # 7. predict 
    
    model.load_weights("svhn.h5")
    detector = YoloDetector(model)

    DEFAULT_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "samples", "svhn", "imgs")
    img_files = [os.path.join(DEFAULT_IMAGE_FOLDER, "1.png"), os.path.join(DEFAULT_IMAGE_FOLDER, "2.png")]
    imgs = []
    for fname in img_files:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    
    for img in imgs:
        boxes = detector.detect(img, COCO_ANCHORS)
        image = draw_boxes(img, boxes, labels=LABELS, desired_size=400)
        plt.imshow(image)
        plt.show()



