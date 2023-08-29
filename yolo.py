#!/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on Mon Aug 28 11:58:25 2023

@author: cdab63
"""

import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, \
                                    ZeroPadding2D, ReLU, LeakyReLU, \
                                    UpSampling2D

class YOLO:

    def __init__(self,
                 parse_cfg=True,
                 create_model=True,
                 cfg_file='yolov3.cfg', 
                 weights_file='yolov3.weights',
                 model_size=(416, 416, 3), 
                 num_classes=80,
                 max_output_size=40,
                 max_output_size_per_class=20,
                 confidence_threshold=0.5,
                 iou_threshold=0.5):
        
        self.cfg_file = cfg_file
        self.weights_file = weights_file
        self.model_size = model_size
        self.num_classes = num_classes
        self.class_names = None
        self.max_output_size = max_output_size
        self.max_output_size_per_class = max_output_size_per_class
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        if parse_cfg:
            self.configuration = self.parse_cfg()
            if not self.configuration:
                raise SyntaxError('Invalid configuration in %s' % self.cfg_file)
            if create_model:
                self.model = self.YOLONet()
                if not self.model:
                    raise ValueError('Invalid parameters for model in %s' % self.cfg_file)
            else:
                self.model = None
        else:
            self.configuration = None
            
            
    # Accessors
    def set_cfg_file(self, a_cfg_file):
        self.cfg_file = a_cfg_file

    def get_cfg_file(self):
        return self.cfg_file
    
    def set_weights_file(self, a_weights_file):
        self.weights_file = a_weights_file
        
    def get_weights_file(self):
        return self.weights_file
    
    def set_model_size(self, a_model_size):
        self.model_size = a_model_size
        
    def get_model_size(self):
        return self.model_size
    
    def set_num_classes(self, nb_of_classes):
        self.num_classes = nb_of_classes
        
    def get_num_classes(self):
        return self.num_classes
    
    def get_configuration(self):
        return self.configuration()
    
    # load model from file
    def load_model(self, a_model):
        try:
            self.model = tf.load_model(a_model)
        except Exception as e:
            print('[ERROR] failed to load yolo from %s' % a_model, file=sys.stderr)
            print('[INFO] %s' % str(e))
            return None 
        return self.model
    
    # save model to file
    def save_model(self, a_model_file):
        try:
            self.model.save_model(a_model_file)
        except Exception as e:
            print('[ERROR] failed to save model to %s' % a_model_file, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        return True
    
    # load weights from file
    def load_weights(self, a_weights_file):
        try:
            self.model.load_weights(a_weights_file)
        except Exception as e:
            print('[ERROR] could not load weights from %s' % a_weights_file, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        return True
    
    # save weights to file
    def save_weights(self, a_weights_file):
        try:
            self.model.save_weights(a_weights_file)
        except Exception as e:
            print('[ERROR] could not save weights to %s' % a_weights_file, file=sys.stderr)
            print('[INFO] %s' % str(e))
            return False
        return True
    
    # convert to tflite
    def to_tflite(self, model_file_name, weights_file_name):
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()
        except Exception as e:
            print('[ERROR] failed to convert model to tflite', file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        try:
            with open(model_file_name, 'wb') as f:
                f.write(tflite_model)
        except Exception as e:
            print('[ERROR] could not save tflite model to %s' % model_file_name, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        try:
            tflite_model.save_weights(weights_file_name)
        except Exception as e:
            print('[ERROR] could not save weights to %s' % weights_file_name, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        self.model = tflite_model
        return True
    
    # quantize model
    def quantize(self, model_file_name, weights_file_name):
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model =converter.convert()
        except Exception as e:
            print('[ERROR] failed to convert model to tflite', file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        try:
            with open(model_file_name, 'wb') as f:
                f.write(tflite_model)
        except Exception as e:
            print('[ERROR] could not save tflite model to %s' % model_file_name, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        try:
            tflite_model.save_weights(weights_file_name)
        except Exception as e:
            print('[ERROR] could not save weights in %s' % weights_file_name, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return False
        self.model = tflite_model
        return True
    
    # Parse the configuration file
    def parse_cfg(self):
        try:
            with open(self.cfg_file, 'r') as file:
                lines = (line.rstrip('\n') for line in file if line != '\n' \
                         and line[0] != '#')
                holder = {}
                blocks = []
                for line in lines:
                    if line[0] == '[':
                        line = 'type=' + line[1:-1].rstrip()
                        if len(holder) != 0:
                            blocks.append(holder)
                            holder = {}
                    key, value = line.split('=')
                    holder[key.rstip()] = value.lstrip()
                blocks.append(holder)
                self.configuration = blocks
                return blocks
        except Exception as e:
            print('[ERROR] Fail to parse file %s' % self.cfg_file, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
        return None

    # Build yolo model
    def YOLONet(self):
        
        outputs = {}
        output_filters = []
        filters = []
        out_pred = []
        scale = 0
        
        inputs = Input(shape=self.model_size)
        input_image = inputs
        inputs = inputs / 255.0 # normalize
        blocks = self.configuration
        for i, block in enumerate(blocks[1:]):
            if (block['type'] == 'convolutional'):
                activation = block['activation']
                filters = block['filters']
                kernel_size = int(block['size'])
                strides = int(block['stride'])
                if strides > 1:
                    inputs = ZeroPadding2D(((1, 0), (1,0)))(inputs)
                inputs = Conv2D(filters,
                                kernel_size,
                                strides=strides,
                                padding='valid' if strides > 1 else 'same',
                                name='conv_' + str(i),
                                activation=activation,
                                use_bias=False if ('batch_normalize' in block) else True)(inputs)
                if 'batch_normalize' in block:
                    inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
                    if activation == 'relu':
                        inputs = ReLU(name='relu_' + str(i))(inputs)
                    else:
                        inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)
            elif (block['type'] == 'upsample'):
                stride = int(block['stride'])
                inputs = UpSampling2D(stride)(inputs)
            elif (block['type'] == 'route'):
                block['layers'] = block['layers'].split(',')
                start = int(block['layers'][0])
                if len(block['layers']) > 1:
                    end = int(block['layers'][1]) - i
                    filters = output_filters[i + start] + output_filters[end]
                    inputs = tf.conacat([outputs[i + start], outputs[i + end]], axis=-1)
                else:
                    filters = output_filters[i + start]
                    inputs = outputs[i + start]
            elif block['type'] == 'shortcut':
                from_ = int(block['from'])
                inputs = outputs[i - 1] + outputs[i + from_]
            elif block['type'] == 'yolo':
                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]
                anchors = block['anchors'].split(',')
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                n_anchors = len(anchors)
                
                out_shape = inputs.get_shape().as_list()
                inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], \
                                             5 + self.num_classes])
                box_centers = inputs[:, :, 0:2]
                box_shapes = inputs[:, :, 2:4]
                confidence = inputs[:, :, 4,5]
                classes = inputs[:, :, 5:self.num_classes + 5]
        
                box_centers = tf.sigmoid(box_centers)
                confidence  = tf.sigmoid(confidence)
                classes     = tf.sigmoid(classes)
        
                anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
                box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
        
                x = tf.range(out_shape[1], dtype=tf.float32)
                y = tf.range(out_shape[2], dtype=tf.float32)
        
                cx, cy = tf.meshgrid(x, y)
                cx = tf.reshape(cx, (-1, 1))
                cxy = tf.concat([cx, cy], axis=-1)
                cxy = tf.tile(cxy,[1, n_anchors])
                cxy = tf.reshape(cxy, [1, -1, 2])
        
                strides = (input_image.shape[1] // out_shape[1],
                           input_image.shape[2] // out_shape[2])
                box_centers = (box_centers + cxy) * strides
        
                prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
        
                if scale:
                    out_pred = tf.concat([out_pred, prediction], axis=1)
                else:
                    out_pred = prediction
                    scale = 1
            else:
                print('[ERROR] unknown layer type %s' % block['type'], file=sys.stderr)
                return None
                 
            outputs[i] = inputs
            output_filters.append(filters)
        
        try:
            model = Model(input_image, out_pred)
            model.summary()
            return model
        except Exception as e:
            print('[ERROR] fail to build model', file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return None
    
    # load weights from a darknet .weights file and optionally save weights in
    # tensorflow tf file
    # it returns a model with weights loaded. If wanted it is possible to
    # copy it to the self.model and have the model with weights
    def load_darknet_weights(self, weights_file, weights_file_tf='yolo_weights.h5', save_weights=True):
        
        if not self.model:
            print('[ERROR] load model before loading weights', file=sys.stderr)
            return None
        else:
            model = self.model
        
        try:
            fp = open(weights_file, 'rb')
        except Exception as e:
            print('[ERROR] cannot open file %s for reading' % weights_file, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return None
        
        try:
            np.fromfile(fp, dtype=np.int32, count=5)
        except Exception as e:
            print('[ERROR] invalid data in %s' % weights_file, file=sys.stderr)
            print('[INFO] %s' % str(e), file=sys.stderr)
            return None
        
        blocks = self.configuration
        
        for i, block in enumerate(blocks[1:]):
            if (block['type'] == 'convolutional'):
                conv_layer = model.get_layer('conv_' + str(i))
                print('layer: ', i+1, conv_layer)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]
                if 'batch_normalize' in block:
                    norm_layer = model.get_layer('bnorm_' + str(i))
                    print('layer: ',i+1, norm_layer)
                    #size = np.prod(norm_layer.get_weights()[0].shape)
                    bn_weights = np.from_file(fp, dtype=np.float32, count=4 * filters)
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                else:
                    conv_bias = np.from_file(fp, dtype=np.float32, count=filters)
                    
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.from_file(fp, dtype=np.float32, count=np.product(conv_shape))
                conv_weights = conv_weights.reshape([2, 3, 1, 0])
                
                if 'batch_normalize' in block:
                    norm_layer.set_weights(bn_weights)
                    conv_layer.set_weights([conv_weights])
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])
                    
        if not len(fp.read()) == 0:
            print('[ERROR] failed to read all data from %s' % weights_file, file=sys.stderr)
            fp.close()
            return None
        
        fp.close()
        
        if save_weights:
            try:
                model.save_weights(weights_file_tf)
                print('[INFO] the file %s has been saved correctly' % weights_file_tf, file=sys.stderr)
            except Exception as e:
                print('[ERROR] could not save file %s' % weights_file_tf, file=sys.stderr)
                print('[INFO} %s' % str(e), file=sys.stderr)
                return None
        
        # model with weights is returned
        return model
    
    # Non max suppression
    def non_max_suppression(self, inputs):
        bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
        bbox = bbox / self.model_size[0]
        scores = confs * class_probs
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=self.max_output_size_per_class,
            max_total_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.confidence_threshold)
        return boxes, scores, classes, valid_detections
    
    # resize images
    def resize_image(self, inputs):
        inputs = tf.image.resize(inputs, self.model_size)
        return inputs
    
    # load class names
    def load_class_names(self, class_names_file):
        try:
            with open(class_names_file, 'r') as f:
                class_names = f.read().splitlines()
            self.class_names = class_names
        except Exception as e:
            print('[ERROR] could not read file %s for class names' % class_names_file, file=sys.stderr)
            print('[INFO] %s' % str(e))
            return None        
        return class_names
    
    # determine the output boxes and suppress non_max
    def output_boxes(self, inputs):
        center_x, center_y, width, height, confidence, classes = tf.split(inputs,
                                                                          [1, 1, 1, 1, 1, -1],
                                                                          axis=-1)
        top_left_x = center_x - width / 2.0
        top_left_y = center_y - height / 2.0
        bottom_right_x = center_x + width / 2.0
        bottom_right_y = center_y + height / 2.0
        
        inputs = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                            confidence, classes], axis=-1)
        boxes_dicts = self.nom_max_suppression(inputs, 
                                               self.max_output_size,
                                               self.max_output_size_per_class,
                                               self.iou_threshold,
                                               self.confidence_threshold)
        return boxes_dicts
    
    # output annotated image
    def draw_outputs(self, img, boxes, objectness, classes, nums):
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        boxes = np.array(boxes)
        for i in range(nums):
            x1y1 = tuple((boxes[i,0:2] * [img.shape[1], img.shape[0]])).astype(np.int32)
            x2y2 = tuple((boxes[1,2:4] * [img.shape[1], img.shape[0]])).astype(np.int32)
            img = cv2.rectangle(img, (x1y1), (x2y2), (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(self.class_names[int(classes[i])],
                                                     objectness[i]), (x1y1),
                                                     cv2.FONT_HERSHEY_PLAIN,
                                                     1, (0, 0, 255), 2)
        return img
    
    # make a prediction
    def predict(self, img, draw_image=True):
        image = np.array(img)
        image = tf.expand_dims(image, 0)
        resized_frame = self.resize_image(image)
        pred = self.model.predict(resized_frame)
        boxes, scores, classes, nums = self.output_boxes(pred, 
                                                         self.max_output_size,
                                                         self.max_output_size_per_class,
                                                         self.iou_threshold,
                                                         self.confidence_threshold)
        if draw_image:
            image = np.squeeze(image)
            image_draw = self.draw_outputs(image, boxes, scores, classes, nums)
            return pred, (boxes, scores, classes, nums), image_draw
        else:
            return pred, (boxes, scores, classes, nums)