# Copyright 2019 ModelArts Authors from Huawei Cloud. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import numpy as np
import os
from PIL import Image
from model_service.frozen_graph_model_service import FrozenGraphBaseService

EPS = np.finfo(float).eps
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def softmax(x):
  x = np.array(x)
  orig_shape = x.shape

  if len(x.shape) > 1:
    # Matrix
    exp_minmax = lambda x: np.exp(x - np.max(x))
    denom = lambda x: 1.0 / (np.sum(x) + EPS)
    x = np.apply_along_axis(exp_minmax, 1, x)
    denominator = np.apply_along_axis(denom, 1, x)
    if len(denominator.shape) == 1:
      denominator = denominator.reshape((denominator.shape[0], 1))
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


def keep_ratio_resize(im, base=256):
  short_side = min(float(im.size[0]), float(im.size[1]))
  resize_ratio = base / short_side
  resize_sides = int(round(resize_ratio * im.size[0])), int(round(resize_ratio * im.size[1]))
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


class ImgclsPBInfer(FrozenGraphBaseService):
  def __init__(self, model_name, model_path):
    super(ImgclsPBInfer, self).__init__(model_name, model_path)
    self.model_inputs = {"images": "images:0"}
    self.model_outputs = {"logits": "logits:0"}

    with open(os.path.join(self.model_path, 'index'), 'r') as f:
      index_map = json.loads(f.read())
    self.labels_list = index_map['labels_list']
    self.is_multilabel = index_map['is_multilabel']

  def _preprocess(self, data):
    preprocessed_data = {}
    for k, v in data.items():
      for file_name, file_content in v.items():
        image = Image.open(file_content)
        if image.mode != 'RGB':
          logging.warning('Input image is not RGB mode, it will cost time to convert to RGB!'
                          'Input RGB mode will reduce infer time.')
          image = image.convert('RGB')
        image = keep_ratio_resize(image, base=256)
        image = central_crop(image, base=224)
        image = np.asarray(image, dtype=np.float32)
        split_image = np.split(image, 3, axis=-1)
        mean = [_R_MEAN, _G_MEAN, _B_MEAN]
        nl_image = []
        for i in range(len(split_image)):
          nl_image.append(split_image[i] - mean[i])
        image = np.concatenate(nl_image, axis=-1)
        image = image[np.newaxis, :, :, :]
        preprocessed_data[k] = image

    return preprocessed_data

  def _postprocess(self, data):
    outputs = {}
    if self.is_multilabel:
      predictions_list = [1 / (1 + np.exp(-p)) for p in data['logits'][0]]
    else:
      predictions_list = softmax(data['logits'][0])
    predictions_list = ['%.3f' % p for p in predictions_list]
    scores = dict(zip(self.labels_list, predictions_list))
    scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(self.labels_list) > 5:
      scores = scores[:5]
    label_index = predictions_list.index(max(predictions_list))
    predicted_label = self.labels_list[label_index]
    outputs['predicted_label'] = predicted_label
    outputs['scores'] = scores

    return outputs
