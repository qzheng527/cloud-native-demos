
import logging
import numpy as np, math
import cv2
from openvino.inference_engine import IENetwork

from clcn.nn.nn import NNBase

logger = logging.getLogger(__name__)

class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence

class NNYolov3Tiny(NNBase):

    camera_width = 320
    camera_height = 240

    m_input_size = 416

    yolo_scale_13 = 13
    yolo_scale_26 = 26
    yolo_scale_52 = 52

    classes = 80
    coords = 4
    num = 3
    anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

    LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

    label_text_color = (255, 255, 255)
    label_background_color = (125, 175, 75)
    box_color = (255, 128, 0)
    box_thickness = 1

    def __init__(self, model_dir, model_name):
        NNBase.__init__(self, model_dir, model_name)
        self.weight = int(self.camera_width * min(self.m_input_size/self.camera_width, self.m_input_size/self.camera_height))
        self.height = int(self.camera_height * min(self.m_input_size/self.camera_width, self.m_input_size/self.camera_height))
        logger.debug("Yolo new weight: %d, height: %d" % (self.weight, self.height))

    def load(self):
        logger.debug("Model XML: " + self.model_xml_path)
        logger.debug("Model weight: " + self.model_weight_path)

        self._net = IENetwork(model=self.model_xml_path, weights=self.model_weight_path)
        assert len(self._net.inputs.keys()) == 1, "Sample supports only single input topologies"
        #assert len(self._net.outputs) == 1, "Sample supports only single output topologies"    

        # Read and pre-process input image
        self.batch_size, self.channel, self.height, self.weight = self._net.inputs[self.input_blob].shape
        logger.debug("Network input shape: " + str(self._net.inputs[self.input_blob].shape))
        logger.debug("Network output shape: " + str(self._net.outputs[self.output_blob].shape))

    def process_input(self, frame):
        resized_image = cv2.resize(frame, (self.weight, self.height), interpolation = cv2.INTER_CUBIC)
        #in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        #in_frame = in_frame.reshape((self.batch_size, self.channel, self.height, self.weight))
        canvas = np.full((self.m_input_size, self.m_input_size, 3), 128)
        canvas[(self.m_input_size-self.height)//2:(self.m_input_size-self.height)//2 + self.height,(self.m_input_size-self.weight)//2:(self.m_input_size-self.weight)//2 + self.weight,  :] = resized_image
        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW        
        return prepimg

    def process_output(self, frame, result):
        objects = []

        for output in result.values():
            objects = self.ParseYOLOV3Output(output, self.height, self.weight, self.camera_height, self.camera_width, 0.4, objects)

        # Filtering overlapping boxes
        objlen = len(objects)
        for i in range(objlen):
            if (objects[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (self.IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                    if objects[i].confidence < objects[j].confidence:
                        objects[i], objects[j] = objects[j], objects[i]
                    objects[j].confidence = 0.0

        # Drawing boxes
        for obj in objects:
            if obj.confidence < 0.2:
                continue
            label = obj.class_id
            confidence = obj.confidence
            #if confidence >= 0.2:
            label_text = self.LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            cv2.rectangle(frame, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), self.box_color, self.box_thickness)
            cv2.putText(frame, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.label_text_color, 1)

        cv2.putText(frame, "", (self.camera_width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)

        return frame

    def EntryIndex(self, side, lcoords, lclasses, location, entry):
        n = int(location / (side * side))
        loc = location % (side * side)
        return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

    def ParseYOLOV3Output(self, blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):
        out_blob_h = blob.shape[2]
        out_blob_w = blob.shape[3]

        side = out_blob_h
        anchor_offset = 0

        if len(self.anchors) == 18:   ## YoloV3
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_52:
                anchor_offset = 2 * 0

        elif len(self.anchors) == 12: ## tiny-YoloV3
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 0

        else:                    ## ???
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_52:
                anchor_offset = 2 * 0

        side_square = side * side
        output_blob = blob.flatten()

        for i in range(side_square):
            row = int(i / side)
            col = int(i % side)
            for n in range(self.num):
                obj_index = self.EntryIndex(side, self.coords, self.classes, n * side * side + i, self.coords)
                box_index = self.EntryIndex(side, self.coords, self.classes, n * side * side + i, 0)
                scale = output_blob[obj_index]
                if (scale < threshold):
                    continue
                x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
                y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
                height = math.exp(output_blob[box_index + 3 * side_square]) * self.anchors[anchor_offset + 2 * n + 1]
                width = math.exp(output_blob[box_index + 2 * side_square]) * self.anchors[anchor_offset + 2 * n]
                for j in range(self.classes):
                    class_index = self.EntryIndex(side, self.coords, self.classes, n * side_square + i, self.coords + 1 + j)
                    prob = scale * output_blob[class_index]
                    if prob < threshold:
                        continue
                    obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                    objects.append(obj)
        return objects

    def IntersectionOverUnion(self, box_1, box_2):
        width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
        height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
        area_of_overlap = 0.0
        if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
            area_of_overlap = 0.0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
        box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
        area_of_union = box_1_area + box_2_area - area_of_overlap
        retval = 0.0
        if area_of_union <= 0.0:
            retval = 0.0
        else:
            retval = (area_of_overlap / area_of_union)
        return retval