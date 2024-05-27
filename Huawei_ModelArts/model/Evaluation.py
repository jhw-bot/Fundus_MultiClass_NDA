# -*- coding: utf-8 -*-
# Copyright 2020 ModelArts Authors from Huawei Cloud. All Rights Reserved.
# https://www.huaweicloud.com/product/modelarts.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import logging
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


import moxing as mox
from moxing.framework.file import file_io
from moxing.framework.common import task_enum
from moxing.framework.data import manifest_metadata
from moxing.tensorflow.datasets import dataset_utils

# ===================== 1. ImageClassification PreProcess in here ========

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def keep_ratio_resize(im, base=256):
    short_side = min(float(im.size[0]), float(im.size[1]))
    resize_ratio = base / short_side
    resize_sides = int(
        round(resize_ratio * im.size[0])), int(round(resize_ratio * im.size[1]))
    im = im.resize(resize_sides)
    return im


def central_crop(im, base=224):
    width, height = im.size
    left = (width - base) / 2
    top = (height - base) / 2
    right = (width + base) / 2
    bottom = (height + base) / 2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def preprocess_vgg_224(image):
    image = keep_ratio_resize(image, base=256)
    image = central_crop(image, base=224)
    image = np.asarray(image, dtype=np.float32)
    split_image = np.split(image, 3, axis=-1)
    mean = [_R_MEAN, _G_MEAN, _B_MEAN]
    nl_image = []
    for i in range(len(split_image)):
        nl_image.append(split_image[i] - mean[i])
    image = np.concatenate(nl_image, axis=-1)
    return image


def preprocess_inception_224(image):
    image = keep_ratio_resize(image, base=256)
    image = central_crop(image, base=224)
    image = np.asarray(image, dtype=np.float32)
    image = image * 1.0 / 255
    image = (image - 0.5) * 2
    return image


def preprocess_inception_299(image):
    image = image.resize((299, 299))
    image = np.asarray(image, dtype=np.float32)
    image = image * 1.0 / 255
    image = (image - 0.5) * 2
    return image


def preprocess_inception_331(image):
    image = image.resize((331, 331))
    image = np.asarray(image, dtype=np.float32)
    image = image * 1.0 / 255
    image = (image - 0.5) * 2
    return image

# ===================== 2. ImageClassification PostProcess in here =======
# Todo

# ===================== 3. ObjectDetection PreProcess in here ============


retina_ori_width = 0
retina_ori_height = 0
net_h = 640
net_w = 640
conf_threshold = 0.3
iou_threshold = 0.45
_scale_factors = [10.0, 10.0, 5.0, 5.0]
channel_means = [123.68, 116.779, 103.939]
class_names = ''
class_num = 0
labels_to_names = {}


def preprocess_default(data):
    preprocessed_data = {}
    for k, v in data.items():
        for file_name, file_content in v.items():
            image = Image.open(file_content)
            if image.mode != 'RGB':
                logging.warning(
                    'Input image is not RGB mode, it will cost time to convert to RGB!'
                    'Input RGB mode will reduce infer time.')
                image = image.convert('RGB')
            image = np.asarray(image, dtype=np.float32)
            image = image[np.newaxis, :, :, :]
            preprocessed_data[k] = image
    return preprocessed_data


def preprocess_retina_640(data):
    preprocessed_data = {}
    for k, v in data.items():
        for file_name, file_content in v.items():
            image = Image.open(file_content)
            if image.mode != 'RGB':
                logging.warning(
                    'Input image is not RGB mode, it will cost time to convert to RGB!'
                    'Input RGB mode will reduce infer time.')
                image = image.convert('RGB')
            global retina_ori_width
            global retina_ori_height
            retina_ori_width, retina_ori_height = image.size
            image = image.resize((net_h, net_w), Image.ANTIALIAS)
            image = np.asarray(image, dtype=np.float32)
            image = image - channel_means
            image = image[np.newaxis, :, :, :]
            preprocessed_data[k] = image
    return preprocessed_data

# ===================== 4. ObjectDetection PostProcess in here ===========


def postprocess_default(data):
    def _nms(
            boxes,
            scores,
            classes,
            picked_boxes,
            picked_classes,
            picked_score,
            nms_iou_threshold=0.3):
        bounding_boxes = boxes
        confidence_score = scores
        # Bounding boxes
        boxes = np.array(bounding_boxes)
        if len(boxes) != 0:
            # coordinates of bounding boxes
            start_x = boxes[:, 0]
            start_y = boxes[:, 1]
            end_x = boxes[:, 2]
            end_y = boxes[:, 3]
            # Confidence scores of bounding boxes
            score = np.array(confidence_score)
            # Picked bounding boxes
            # Compute areas of bounding boxes
            areas = (end_x - start_x + 1) * (end_y - start_y + 1)
            # Sort by confidence score of bounding boxes
            order = np.argsort(score)
            # Iterate bounding boxes
            while order.size > 0:
                # The index of largest confidence score
                index = order[-1]
                # Pick the bounding box with largest confidence score
                picked_boxes.append(bounding_boxes[index])
                picked_score.append(confidence_score[index])
                for num_label in range(len(labels_to_names)):
                    if labels_to_names[str(
                            num_label)] == classes[index]:
                        picked_classes.append(num_label)

                # Compute ordinates of intersection-over-union(IOU)
                x1 = np.maximum(start_x[index], start_x[order[:-1]])
                x2 = np.minimum(end_x[index], end_x[order[:-1]])
                y1 = np.maximum(start_y[index], start_y[order[:-1]])
                y2 = np.minimum(end_y[index], end_y[order[:-1]])
                # Compute areas of intersection-over-union
                w = np.maximum(0.0, x2 - x1 + 1)
                h = np.maximum(0.0, y2 - y1 + 1)
                intersection = w * h
                # Compute the ratio between intersection and union
                ratio = intersection / \
                    (areas[index] + areas[order[:-1]] - intersection)
                left = np.where(ratio < nms_iou_threshold)
                order = order[left]

    labels_list = list(labels_to_names.values())
    detection_classes = data['detection_classes'][0]
    detection_scores = data['detection_scores'][0]
    detection_boxes = data['detection_boxes'][0]
    num_boxes = len(detection_classes)
    classes = [[] for _ in range(len(labels_list))]
    boxes = [[] for _ in range(len(labels_list))]
    scores = [[] for _ in range(len(labels_list))]
    prob_threshold = 0.3
    result_return = dict()
    for i in range(num_boxes):
        if detection_scores[i] > prob_threshold:
            for index, label_name in enumerate(labels_list):
                class_id = detection_classes[i] - 1
                if class_id == index:
                    classes[index].append(labels_list[int(class_id)])
                    boxes[index].append(detection_boxes[i])
                    scores[index].append(detection_scores[i])

    # #########add NMS#######################################
    picked_classes = []
    picked_boxes = []
    picked_score = []
    for i in range(len(labels_list)):
        if boxes[i] and scores[i] and classes[i]:
            _nms(
                boxes=boxes[i],
                scores=scores[i],
                classes=classes[i],
                picked_boxes=picked_boxes,
                picked_classes=picked_classes,
                picked_score=picked_score,
                nms_iou_threshold=0.3,
            )
        else:
            continue

    result_return['detection_classes'] = picked_classes
    result_return['detection_boxes'] = picked_boxes
    result_return['detection_scores'] = picked_score
    return result_return


def postprocess_retina_640(data):
    def _sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def get_center_coordinates_and_sizes(bbox):
        ymin, xmin, ymax, xmax = np.transpose(bbox)
        width = xmax - xmin
        height = ymax - ymin
        ycenter = ymin + height / 2.
        xcenter = xmin + width / 2.
        return [ycenter, xcenter, height, width]

    def combined_static_and_dynamic_shape(tensor):
        static_tensor_shape = list(tensor.shape)
        dynamic_tensor_shape = tensor.shape
        combined_shape = []
        for index, dim in enumerate(static_tensor_shape):
            if dim is not None:
                combined_shape.append(dim)
            else:
                combined_shape.append(dynamic_tensor_shape[index])
        return combined_shape

    def batch_decode(box_encodings, anchors, class_predictions, ori_h, ori_w):
        combined_shape = combined_static_and_dynamic_shape(box_encodings)
        batch_size = combined_shape[0]
        tiled_anchor_boxes = np.tile(
            np.expand_dims(anchors, 0), [batch_size, 1, 1])
        tiled_anchors_boxlist = np.reshape(tiled_anchor_boxes, [-1, 4])
        decoded_boxes = decode_bbox_frcnn(np.reshape(
            box_encodings, [-1, 4]), tiled_anchors_boxlist, ori_h, ori_w)
        decoded_boxes = np.reshape(decoded_boxes, np.stack(
            [combined_shape[0], combined_shape[1], 4]))
        return decoded_boxes

    def overlap(x1, x2, x3, x4):
        left = max(x1, x3)
        right = min(x2, x4)
        return right - left

    def cal_iou(box, truth):
        w = overlap(box[0], box[2], truth[0], truth[2])
        h = overlap(box[1], box[3], truth[1], truth[3])
        if w <= 0 or h <= 0:
            return 0
        inter_area = w * h
        union_area = (box[2] - box[0]) * (box[3] - box[1]) + \
            (truth[2] - truth[0]) * (truth[3] - truth[1]) - inter_area
        return inter_area * 1.0 / union_area

    def apply_nms(all_boxes, thres):
        res = []

        for cls in range(class_num):
            cls_bboxes = all_boxes[cls]
            sorted_boxes = sorted(cls_bboxes, key=lambda d: d[5])[::-1]

            p = dict()
            for i in range(len(sorted_boxes)):
                if i in p:
                    continue

                truth = sorted_boxes[i]
                for j in range(i + 1, len(sorted_boxes)):
                    if j in p:
                        continue
                    box = sorted_boxes[j]
                    iou = cal_iou(box, truth)
                    if iou >= thres:
                        p[j] = 1

            for i in range(len(sorted_boxes)):
                if i not in p:
                    res.append(sorted_boxes[i])
        return res

    def decode_bbox_frcnn(conv_output, anchors, ori_h, ori_w):
        ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(
            anchors)
        ty, tx, th, tw = conv_output.transpose()
        ty /= _scale_factors[0]
        tx /= _scale_factors[1]
        th /= _scale_factors[2]
        tw /= _scale_factors[3]

        w = np.exp(tw) * wa
        h = np.exp(th) * ha

        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a

        scale_h = 1
        scale_w = 1
        ymin = np.maximum((ycenter - h / 2.) * scale_h * ori_h, 0)
        xmin = np.maximum((xcenter - w / 2.) * scale_w * ori_w, 0)
        ymax = np.minimum((ycenter + h / 2.) * scale_h * ori_h, ori_h)
        xmax = np.minimum((xcenter + w / 2.) * scale_w * ori_w, ori_w)

        bbox_list = np.transpose(np.stack([ymin, xmin, ymax, xmax]))

        return bbox_list

    def postprocess(anchors, box_encodings, class_predictions, ori_h, ori_w):
        detection_boxes = batch_decode(
            box_encodings, anchors, class_predictions, ori_h, ori_w)
        detection_scores_with_background = _sigmoid(class_predictions)
        detection_scores = detection_scores_with_background[:, :, 1:]
        detection_score = np.expand_dims(
            np.max(detection_scores[:, :, 0:], axis=-1), 2)
        detection_cls = np.expand_dims(
            np.argmax(detection_scores[:, :, 0:], axis=-1), 2)

        preds = np.concatenate(
            [detection_boxes, detection_score, detection_cls], axis=2)
        res_list = []
        for pred in preds:
            pred = pred[pred[:, 4] >= conf_threshold]
            all_boxes = [[] for ix in range(class_num)]
            for ix in range(pred.shape[0]):
                box = [int(pred[ix, iy]) for iy in range(4)]
                box.append(int(pred[ix, 5]))
                box.append(pred[ix, 4])
                all_boxes[box[4] - 1].append(box)

            res = apply_nms(all_boxes, iou_threshold)
            res_list.append(res)
        return res_list

    def get_result(result, img_h, img_w):
        try:
            anchors = result["anchors"]
            box_encodings = result["box_encodings"]
            class_predictions_with_background = result["class_predictions_with_background"]
        except KeyError as e:
            raise KeyError("retinanet model analysis require 'export_om_model' is True")
        detection_dict = postprocess(anchors, box_encodings,
                                     class_predictions_with_background,
                                     img_h, img_w)

        return detection_dict

    result = get_result(data, retina_ori_height, retina_ori_width)
    response = {
        'detection_classes': [],
        'detection_boxes': [],
        'detection_scores': []
    }
    for i in range(len(result[0])):
        ymin = result[0][i][0]
        xmin = result[0][i][1]
        ymax = result[0][i][2]
        xmax = result[0][i][3]
        score = result[0][i][5]
        label = result[0][i][4]

        if score < conf_threshold:
            continue
        response['detection_classes'].append(int(label))
        response['detection_boxes'].append([
            str(ymin), str(xmin),
            str(ymax), str(xmax)
        ])
        response['detection_scores'].append(str(score))
    return response

# ==============================================================================
# do not edit this code


default_image_classification_preprocess_func = preprocess_vgg_224
default_analysis_task_type = 'image_classification'
default_object_detection_preprocess_func = preprocess_default
default_model_name = 'resnet_v1_50'
default_object_detection_postprocess_func = postprocess_default
# ==============================================================================


class ModelAnalysis(object):

    def __init__(self):
        # model analysis parameters
        self.task_type = ''
        self.pred_list = []
        self.label_list = []
        self.name_list = []
        self.labels_to_names = {}
        self.names_to_labels = {}
        self.save_path = ''
        self.sample_list = []
        self.model_name = ''
        self.input_tensor_name = ''
        self.output_tensor_name = ''

        # infer parameters
        self.model_inputs = {}
        self.model_outputs = {}
        self.sess = None

    def _set_up(self, model_url, local_cache):
        """
        ===============================================================
        1.Download model in local,If them in obs.
        ===============================================================
        """
        self.__init__()
        if model_url.startswith(('obs://', 's3://')):
            mox.file.copy_parallel(
                model_url, os.path.join(
                    local_cache, 'model'))
            model_url = os.path.join(local_cache, 'model')

        with open(os.path.join(model_url, 'index'), 'r') as f:
            index_map = json.loads(f.read())

        global class_names
        global class_num
        class_names = index_map['labels_list'][:]
        class_num = len(class_names)

        """
        ===============================================================
        2.Loda Model
        ===============================================================
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        self.sess = tf.Session(graph=tf.Graph(), config=config)
        meta_graph_def = tf.saved_model.loader.load(
            self.sess, [tf.saved_model.tag_constants.SERVING], model_url)
        signature_defs = meta_graph_def.signature_def
        signature = []
        # only one signature allowed
        for signature_def in signature_defs:
            signature.append(signature_def)
        if len(signature) == 1:
            model_signature = signature[0]
        else:
            logging.warning(
                "signatures more than one, use serving_default signature")
            model_signature = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

        for signature_name in meta_graph_def.signature_def[model_signature].inputs:
            tensorinfo = meta_graph_def.signature_def[model_signature].inputs[signature_name]
            self.input_tensor_name = tensorinfo.name
            op = self.sess.graph.get_tensor_by_name(self.input_tensor_name)
            self.model_inputs[signature_name] = op

        for signature_name in meta_graph_def.signature_def[model_signature].outputs:
            tensorinfo = meta_graph_def.signature_def[model_signature].outputs[signature_name]
            self.output_tensor_name = tensorinfo.name
            op = self.sess.graph.get_tensor_by_name(self.output_tensor_name)
            self.model_outputs[signature_name] = op

    def _preprocess(self, data):
        return data

    def _inference(self, data):

        feed_dict = {}
        for k, v in data.items():
            if k not in self.model_inputs.keys():
                logging.error(
                    "input key %s is not in model inputs %s", k, list(
                        self.model_inputs.keys()))
                raise Exception(
                    "input key %s is not in model inputs %s" %
                    (k, list(
                        self.model_inputs.keys())))
            feed_dict[self.model_inputs[k]] = v

        result = self.sess.run(self.model_outputs, feed_dict=feed_dict)
        return result

    def _postprocess(self, data):
        return data

    def evaluate_model(self):
        """Evaluate model and generate config.json.

        ################################################################################################################

        Image Classification model evaluate

        Args::

            task_type :
                Image Classification only use ['image_classification','multi_label_classification']

            pred_list :
                type：one-dimensional list，The length is the number of categories
                express：The confidence level of the picture in each category
                example：[[0.87, 0.11, 0.02], [0.1, 0.7, 0.2], [0.03, 0.04, 0.93], [0.25, 0.65, 0.1], [0.3, 0.34, 0.36]]

            label_list :
                type：Integer value
                express：The category of the label
                example：[0, 1, 2, 1, 2]

        ################################################################################################################

        Object Detection model evaluate

        Args::

            task_type : Object Detection only use 'image_object_detection'

            pred_list :

                type：[[[]], [], []] Python list, There are three elements：
                    The first is a two-dimensional array or numpy ndarray,
                    shape is num(The number of box in a picture)*4(ymin, xmin, ymax, xmax);

                    The second is a one-dimensional array or numpy ndarray,
                    length is num(The number of box in a picture);

                    The third is a one-dimensional array or numpy ndarray,
                    length is num(The number of box in a picture)

                express：
                    [Prediction box coordinates，Prediction box category，Prediction box confidence]
                example：
                    [[[[142.26546  , 172.09337  , 182.41393  , 206.43747  ],
                    [149.60696  , 232.63474  , 185.081    , 262.0958   ],
                    [151.28708  , 305.58755  , 186.05899  , 335.83026  ]],
                    [1, 1, 1],
                    [0.999926  , 0.9999119 , 0.99985504]],
                    [[[184.18466 , 100.23248 , 231.96555 , 147.65791 ],
                    [ 43.406055, 252.89429 ,  84.62765 , 290.55862 ]],
                    [3, 3],
                    [0.99985814, 0.99972576]],
                    ...]

            label_list:

                type：[[[]], []] Python list There are two elements，
                    The first is a two-dimensional array or numpy ndarray,
                    shape is num(The number of box in a picture)*4(ymin, xmin, ymax, xmax);

                    The second is a one-dimensional array or numpy ndarray,
                    length is num(The number of box in a picture)

                express：
                    [groundtruth box coordinates，groundtruth box category]

                example：
                    [[[[182., 100., 229., 146.],
                    [ 44., 250.,  83., 290.]], [3, 3]],
                    [[[148., 303., 191., 336.],
                    [149., 231., 189., 262.],
                    [141., 171., 184., 206.],
                    [132., 344., 183., 387.],
                    [144., 399., 189., 430.]], [1., 1., 1., 2., 4.]],
                    ...]
        ################################################################################################################

        Common Args::

            name_list: The absolute path of the corresponding image

            label_map_dict : A dict with label index and name, like {"0": "dog", "1": "cat", "2": "horse"}

            save_path : path to save analyse result

        """
        from deep_moxing.model_analysis import api
        # 调用评估接口，进行指标计算
        if task_enum.is_task_imgcls(self.task_type):
            api.get_advanced_metric_info(
                task_type=self.task_type,
                sample_list=self.sample_list,
                gt_label_list=self.label_list,
                label_map_dict=self.labels_to_names,
                file_name_list=self.name_list,
                session=self.sess,
                model_name='' if not self.model_name else self.model_name,
                input_tensor_name=self.input_tensor_name,
                output_tensor_name=self.output_tensor_name,
                save_path='' if not self.save_path else self.save_path)

        res = api.analyse(
            task_type=self.task_type,
            pred_list=self.pred_list,
            label_list=self.label_list,
            name_list=self.name_list,
            label_map_dict=self.labels_to_names,
            op_list=[],
            save_path='' if not self.save_path else self.save_path)
        return res

    def get_result(
            self,
            task_type,
            data_url,
            model_url,
            save_path='',
            model_name=None,
            do_data_cleaning=None,
            data_type=None,
            local_cache='/cache'):
        """
        :param data_url: infer file path.
        :param model_url: model path, specific to model directory.
        :param save_path: save model analysis result
        :param local_cache: local cache path, only used when model_url is obs path.
        :param data_type: data_url type
        :return: infer output.
        """
        raise NotImplementedError('BaseClass Not Implemented get_result')


class ClassificationModelAnalysis(ModelAnalysis):

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                if image.mode != 'RGB':
                    logging.WARNING(
                        'Input image is not RGB mode, it will cost time to convert to RGB!'
                        'Input RGB mode will reduce infer time.')
                    image = image.convert('RGB')

                image = default_image_classification_preprocess_func(image)
                image = image[np.newaxis, :, :, :]
                preprocessed_data[k] = image
        return preprocessed_data

    def _postprocess(self, data):

        def softmax(x):
            EPS = np.finfo(float).eps
            x = np.array(x)
            orig_shape = x.shape

            if len(x.shape) > 1:
                # Matrix
                def exp_minmax(x):
                    return np.exp(x - np.max(x))

                def denom(x):
                    return 1.0 / (np.sum(x) + EPS)

                x = np.apply_along_axis(exp_minmax, 1, x)
                denominator = np.apply_along_axis(denom, 1, x)
                if len(denominator.shape) == 1:
                    denominator = denominator.reshape(
                        (denominator.shape[0], 1))
                x = x * denominator
            else:
                # Vector
                x_max = np.max(x)
                x = x - x_max
                numerator = np.exp(x)
                denominator = 1.0 / (np.sum(numerator) + EPS)
                x = numerator.dot(denominator)
            assert x.shape == orig_shape

            return x

        predictions_list = softmax(data['logits'][0])

        return predictions_list

    def get_result(
            self,
            task_type,
            data_url,
            model_url,
            save_path='',
            model_name=None,
            do_data_cleaning=None,
            data_type=None,
            local_cache='/cache'):

        self._set_up(model_url, local_cache)

        """
        ===============================================================
        3.Loda Data and predict
        ===============================================================
        """
        data_type = dataset_utils.get_data_type(data_url, data_type)

        if data_type == dataset_utils.MANIFEST_DATA_TYPE:
            data_path_map = None
            if do_data_cleaning:
                from deep_moxing.data_preprocess.preprocess_ops.image_preprocess import DATA_PATH_MAP
                data_path = os.path.join(
                    os.path.abspath(
                        os.path.dirname(data_url)),
                    DATA_PATH_MAP)
                if file_io.exists(data_path):
                    with open(data_path, 'r') as j:
                        data_path_map = json.loads(j.read())

            meta = manifest_metadata.ImageClassificationManifestSource(
                data_url)
            if meta.eval.num_samples <= 0:
                raise ValueError(
                    'Mode Analysis use eval type in manifest to infer,And now eval samples is 0')
            elif meta.eval.is_multilabel:
                raise ValueError(
                    'Mode Analysis only support single label in [classification eval data]')

            self.task_type = task_type
            self.save_path = save_path
            self.model_name = model_name
            for idx, item in enumerate(class_names):
                self.labels_to_names[str(idx)] = item
                self.names_to_labels[item] = idx

            global labels_to_names
            labels_to_names = self.labels_to_names
            for item in meta.eval.all_files_list:
                self.label_list.append(
                    self.names_to_labels[meta.eval.get_label(item)])
                with mox.file.File(item, 'rb') as f:
                    preprocessed_data = self._preprocess(
                        {'images': {'image_name': f}})
                self.sample_list.append(preprocessed_data['images'])
                infer_result = self._inference(preprocessed_data)
                output = self._postprocess(infer_result)
                self.pred_list.append(output)
                self.name_list.append(
                    item if data_path_map is None else data_path_map[item])

        else:
            raise ValueError(
                'Mode Analysis only support data type in [manifest]')

        """
        ===============================================================
        4.Model Analysis
        ===============================================================
        """
        return self.evaluate_model()


class DetectionModelAnalysis(ModelAnalysis):

    def get_result(
            self,
            task_type,
            data_url,
            model_url,
            save_path='',
            model_name=None,
            do_data_cleaning=None,
            data_type=None,
            local_cache='/cache'):

        self._set_up(model_url, local_cache)

        """
        ===============================================================
        3.Loda Data and predict
        ===============================================================
        """
        data_type = dataset_utils.get_data_type(data_url, data_type)

        if data_type == dataset_utils.MANIFEST_DATA_TYPE:
            data_path_map = None
            if do_data_cleaning:
                from deep_moxing.data_preprocess.preprocess_ops.image_preprocess import DATA_PATH_MAP
                data_path = os.path.join(
                    os.path.abspath(
                        os.path.dirname(data_url)),
                    DATA_PATH_MAP)
                if file_io.exists(data_path):
                    with open(data_path, 'r') as j:
                        data_path_map = json.loads(j.read())

            meta = manifest_metadata.ObjectDetectionManifestSource(data_url)
            if meta.eval.num_samples <= 0:
                raise ValueError(
                    'Mode Analysii use eval type in manifest to infer,And now eval samples is 0')

            self.task_type = 'image_object_detection'
            self.save_path = save_path
            self.model_name = model_name
            for idx, item in enumerate(class_names):
                self.labels_to_names[str(idx)] = item
                self.names_to_labels[item] = idx

            global labels_to_names
            labels_to_names = self.labels_to_names
            for item in meta.eval.all_files_list:
                annotation = meta.eval.get_label(item[1])
                bbox_list = []
                label_list = []
                for obj in annotation['annotation']['object']:
                    xmin = int(float(obj['bndbox']['xmin']))
                    ymin = int(float(obj['bndbox']['ymin']))
                    xmax = int(float(obj['bndbox']['xmax']))
                    ymax = int(float(obj['bndbox']['ymax']))
                    label_name = obj['name']
                    bbox_list.append([ymin, xmin, ymax, xmax])
                    label_list.append(self.names_to_labels[label_name])

                self.label_list.append(
                    [np.array(bbox_list), np.array(label_list)])
                with mox.file.File(item[0], 'rb') as f:
                    preprocessed_data = default_object_detection_preprocess_func(
                        {'images': {'image_name': f}})
                self.sample_list.append(preprocessed_data['images'])
                infer_result = self._inference(preprocessed_data)
                output = default_object_detection_postprocess_func(
                    infer_result)
                self.pred_list.append([np.array(output['detection_boxes']), np.array(
                    output['detection_classes']), np.array(output['detection_scores'])])
                self.name_list.append(
                    item[0] if data_path_map is None else data_path_map[item[0]])

        else:
            raise ValueError(
                'Mode Analysis only support data type in [manifest]')

        """
        ===============================================================
        4.Model Analysis
        ===============================================================
        """
        return self.evaluate_model()


class UndefinedModelAnalysis(ModelAnalysis):
    def get_result(
            self,
            task_type,
            data_url,
            model_url,
            save_path='',
            model_name=None,
            do_data_cleaning=None,
            data_type=None,
            local_cache='/cache'):
        raise ValueError(
            'the task_type now is unsupport, only support [image_classification, object_detection].')


class Factory(object):
    taskdict = {
        task_enum.IMAGE_CLASSIFICATION: ClassificationModelAnalysis(),
        task_enum.OBJECT_DETECTION: DetectionModelAnalysis()}

    def gettask(
            self,
            task_type,
            data_url,
            model_url,
            save_path='',
            model_name=None,
            do_data_cleaning=None,
            data_type=None,
            local_cache='/cache'):
        if task_type is None:
            task_type = default_analysis_task_type

        if model_name is None:
            model_name = default_model_name

        if task_type in self.taskdict:
            return self.taskdict[task_type].get_result(
                task_type,
                data_url,
                model_url,
                save_path,
                model_name,
                do_data_cleaning,
                data_type,
                local_cache)
        else:
            return UndefinedModelAnalysis().get_result(
                task_type,
                data_url,
                model_url,
                save_path,
                model_name,
                do_data_cleaning,
                data_type,
                local_cache)


if __name__ == '__main__':
    mox.file.shift('tf', 'mox')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', type=str, default=None)
    parser.add_argument('--train_url', type=str, default=None)
    parser.add_argument('--model_url', type=str, default=None)
    parser.add_argument('--task_type', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv)

    outputs = Factory().gettask(
        args.task_type,
        args.data_url,
        args.model_url,
        args.train_url,
        args.model_name)
    logging.info(outputs)
