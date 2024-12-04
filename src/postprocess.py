# Original Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================

""" Modified.
Postprocessing for anchor-based detection."""


import functools

import anchors
import nms_np
import tensorflow as tf
import utils
from absl import logging
from utils_box import decode_uncert
from utils_extra import get_mcuncert

T = tf.Tensor  # A shortcut for typing check.
CLASS_OFFSET = 1

# TFLite-specific constants.
TFLITE_MAX_CLASSES_PER_DETECTION = 1
TFLITE_DETECTION_POSTPROCESS_FUNC = "TFLite_Detection_PostProcess"
TFLITE_USE_REGULAR_NMS = False


def to_list(inputs):
    if isinstance(inputs, dict):
        return [inputs[k] for k in sorted(inputs.keys())]
    if isinstance(inputs, list):
        return inputs
    if isinstance(inputs, tuple):
        return list(inputs)


def batch_map_fn(map_fn, inputs, *args):
    """Apply map_fn at batch dimension."""
    if isinstance(inputs[0], (list, tuple)):
        batch_size = len(inputs[0])
    else:
        batch_size = inputs[0].shape.as_list()[0]

    if not batch_size:
        # handle dynamic batch size: tf.vectorized_map is faster than tf.map_fn.
        return tf.vectorized_map(map_fn, inputs, *args)
    outputs = []
    for i in range(batch_size):
        outputs.append(map_fn([x[i] for x in inputs]))
    return [tf.stack(y) for y in zip(*outputs)]


def clip_boxes(boxes: T, image_size: int) -> T:
    """Clip boxes to fit the image size."""
    image_size = utils.parse_image_size(image_size) * 2
    return tf.clip_by_value(boxes, [0], image_size)


def merge_class_box_level_outputs(params, cls_outputs, box_outputs):
    """Concatenates class and box of all levels into one tensor."""
    cls_outputs_all, box_outputs_all = [], []
    batch_size = tf.shape(cls_outputs[0])[0]
    for level in range(0, params["max_level"] - params["min_level"] + 1):
        if params["data_format"] == "channels_first":
            cls_outputs[level] = tf.transpose(cls_outputs[level], [0, 2, 3, 1])
            box_outputs[level] = tf.transpose(box_outputs[level], [0, 2, 3, 1])
        cls_outputs_all.append(
            tf.reshape(cls_outputs[level], [batch_size, -1, params["num_classes"]])
        )
        box_outputs_all.append(tf.reshape(box_outputs[level], [batch_size, -1, 4]))
    return tf.concat(cls_outputs_all, 1), tf.concat(box_outputs_all, 1)


def topk_class_boxes(params, cls_outputs, box_outputs, uncerts=None):
    """Pick the topk class and box outputs."""
    batch_size = tf.shape(cls_outputs)[0]
    num_classes = params["num_classes"]

    max_nms_inputs = params["nms_configs"].get("max_nms_inputs", 0)
    if max_nms_inputs > 0:
        # Prune anchors and detections to only keep max_nms_inputs.
        # Due to some issues, top_k is currently slow in graph model.
        logging.info("use max_nms_inputs for pre-nms topk.")
        cls_outputs_reshape = tf.reshape(cls_outputs, [batch_size, -1])
        _, cls_topk_indices = tf.math.top_k(
            cls_outputs_reshape, k=max_nms_inputs, sorted=False
        )
        indices = cls_topk_indices // num_classes
        classes = cls_topk_indices % num_classes
        cls_indices = tf.stack([indices, classes], axis=2)

        cls_outputs_topk = tf.gather_nd(cls_outputs, cls_indices, batch_dims=1)
        box_outputs_topk = tf.gather_nd(
            box_outputs, tf.expand_dims(indices, 2), batch_dims=1
        )
        # also select topk uncertainty
        if uncerts is not None:
            for i in range(len(uncerts)):
                if uncerts[i] is not None:
                    if i != 0:  # for regression
                        uncerts[i] = tf.gather_nd(
                            uncerts[i], tf.expand_dims(indices, 2), batch_dims=1
                        )
                    else:
                        uncerts[i] = tf.gather_nd(uncerts[i], cls_indices, batch_dims=1)

    else:
        logging.info("use max_reduce for pre-nms topk.")
        # Keep all anchors, but for each anchor, just keep the max probablity for
        # each class.
        cls_outputs_idx = tf.math.argmax(cls_outputs, axis=-1, output_type=tf.int32)
        num_anchors = tf.shape(cls_outputs)[1]

        classes = cls_outputs_idx
        indices = tf.tile(
            tf.expand_dims(tf.range(num_anchors), axis=0), [batch_size, 1]
        )
        cls_outputs_topk = tf.reduce_max(cls_outputs, -1)
        box_outputs_topk = box_outputs

    # consider uncertainty
    if uncerts is not None:
        return cls_outputs_topk, box_outputs_topk, classes, indices, uncerts
    else:
        return cls_outputs_topk, box_outputs_topk, classes, indices


def pre_nms(params, cls_outputs, box_outputs, topk=True, uncerts=None):
    """Detection post processing before nms.

    It takes the multi-level class and box predictions from network, merge them
    into unified tensors, and compute boxes, scores, and classes.

    Args:
      params: a dict of parameters.
      cls_outputs: a list of tensors for classes, each tensor denotes a level of
        logits with shape [N, H, W, num_class * num_anchors].
      box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
        boxes with shape [N, H, W, 4 * num_anchors].
      topk: if True, select topk before nms (mainly to speed up nms).
      uncerts: if not None, the uncertainties are also filtered accordingly

    Returns:
      A tuple of (boxes, scores, classes),
      optionally with logits after classes and uncertainty after boxes
    """
    # Get boxes by apply bounding box regression to anchors.
    eval_anchors = anchors.Anchors(
        params["min_level"],
        params["max_level"],
        params["num_scales"],
        params["aspect_ratios"],
        params["anchor_scale"],
        params["image_size"],
    )
    # Transofrm uncertainties of all five levels into one tensor N,4
    if params["loss_attenuation"] and not (
        params["mc_boxheadrate"] or params["mc_dropoutrate"]
    ):
        uncerts[1] = merge_class_box_level_outputs(params, cls_outputs, uncerts[1])[1]
    if params["mc_classheadrate"] or params["mc_dropoutrate"]:
        uncerts[0] = merge_class_box_level_outputs(params, uncerts[0], box_outputs)[0]
    if params["mc_boxheadrate"] or params["mc_dropoutrate"]:
        batchsizeover1 = box_outputs[1].shape[0] != 1
        if params["loss_attenuation"]:
            al_unc = []
            for j in range(box_outputs[0].shape[0]):  # Merge for each feature map
                al_box = [uncerts[1][i][j] for i in range(len(box_outputs))]
                al_unc.append(
                    merge_class_box_level_outputs(params, cls_outputs, al_box)[1]
                )
            if batchsizeover1:
                uncerts[1] = tf.stack(al_unc, axis=0)
            else:
                uncerts[1] = tf.concat(al_unc, axis=0)

        boxes = []
        for j in range(box_outputs[0].shape[0]):
            mc_box = [box_outputs[i][j] for i in range(len(box_outputs))]
            temp_cls, temp_box = merge_class_box_level_outputs(
                params, cls_outputs, mc_box
            )
            boxes.append(temp_box)
        if batchsizeover1:
            box_outputs = tf.stack(boxes, axis=0)
        else:
            box_outputs = tf.concat(boxes, axis=0)
        cls_outputs = temp_cls
    else:
        cls_outputs, box_outputs = merge_class_box_level_outputs(
            params, cls_outputs, box_outputs
        )

    if params["enable_softmax"]:  # Copy predictions for logits extraction
        classes_multi = tf.identity(cls_outputs)
    if topk:
        # select topK purely based on scores before NMS, in order to speed up nms.
        if uncerts is not None:
            if params["mc_boxheadrate"] or params["mc_dropoutrate"]:
                if params["loss_attenuation"]:
                    al_unc = []
                    for i in range(params["mc_dropoutsamp"]):
                        if batchsizeover1:
                            al_unc.append(
                                topk_class_boxes(
                                    params,
                                    cls_outputs,
                                    box_outputs[i],
                                    [uncerts[0], uncerts[1][i], uncerts[2]],
                                )[-1]
                            )
                        else:
                            al_unc.append(
                                topk_class_boxes(
                                    params,
                                    cls_outputs,
                                    tf.expand_dims(box_outputs[i], axis=0),
                                    [
                                        uncerts[0],
                                        tf.expand_dims(uncerts[1][i], axis=0),
                                        uncerts[2],
                                    ],
                                )[-1]
                            )
                    al_unc = [al_unc[i][1] for i in range(len(al_unc))]
                    if batchsizeover1:
                        uncerts[1] = tf.stack(al_unc, axis=0)
                    else:
                        uncerts[1] = tf.concat(al_unc, axis=0)
                boxes = []
                for i in range(params["mc_dropoutsamp"]):
                    if batchsizeover1:
                        temp_box = box_outputs[i]
                        temp_uncerts = [uncerts[0], None, uncerts[2]]
                    else:
                        temp_box = tf.expand_dims(box_outputs[i], axis=0)
                        temp_uncerts = uncerts
                    temp_cls, temp_box, temp_classes, temp_indices, temp_uncerts = (
                        topk_class_boxes(params, cls_outputs, temp_box, temp_uncerts)
                    )
                    boxes.append(temp_box)

                if batchsizeover1:
                    box_outputs = tf.stack(boxes, axis=0)
                else:
                    box_outputs = tf.concat(boxes, axis=0)
                cls_outputs = temp_cls
                classes = temp_classes
                indices = temp_indices
                if batchsizeover1:
                    uncerts[0] = temp_uncerts[0]
                    uncerts[2] = temp_uncerts[2]
                else:
                    uncerts = temp_uncerts
            else:
                cls_outputs, box_outputs, classes, indices, uncerts = topk_class_boxes(
                    params, cls_outputs, box_outputs, uncerts
                )
        else:
            cls_outputs, box_outputs, classes, indices = topk_class_boxes(
                params, cls_outputs, box_outputs
            )
        anchor_boxes = tf.gather(eval_anchors.boxes, indices)
    else:
        anchor_boxes = eval_anchors.boxes
        classes = None

    scores = tf.math.sigmoid(cls_outputs)  # Convert logits to scores

    if params["loss_attenuation"] and not (
        params["mc_boxheadrate"] or params["mc_dropoutrate"]
    ):
        boxes, uncerts[1] = decode_uncert(
            box_outputs,
            uncerts[1],
            anchor_boxes,
            method=params["uncert_adjust_method"],
            n_samples=params["decode_nsamples"],
        )
    elif params["mc_boxheadrate"] or params["mc_dropoutrate"]:
        if params["loss_attenuation"]:
            boxes = []
            aluncdec = []
            for i in range(params["mc_dropoutsamp"]):
                temp_box, temp_aluncdec = decode_uncert(
                    box_outputs[i],
                    uncerts[1][i],
                    anchor_boxes,
                    method=params["uncert_adjust_method"],
                    n_samples=params["decode_nsamples"],
                )
                boxes.append(temp_box)
                aluncdec.append(temp_aluncdec)
            if batchsizeover1:
                alunc = tf.stack(aluncdec, axis=0)
                box_outputs = tf.stack(boxes, axis=0)
                uncerts[1] = tf.reduce_mean(alunc, axis=0)
            else:
                alunc = tf.concat(aluncdec, axis=0)
                box_outputs = tf.concat(boxes, axis=0)
                uncerts[1] = tf.expand_dims(tf.reduce_mean(alunc, axis=0), axis=0)
        else:
            boxes = []
            for i in range(params["mc_dropoutsamp"]):
                boxes.append(anchors.decode_box_outputs(box_outputs[i], anchor_boxes))
            if batchsizeover1:
                box_outputs = tf.stack(boxes, axis=0)
            else:
                box_outputs = tf.concat(boxes, axis=0)
        if batchsizeover1:
            boxes = tf.reduce_mean(box_outputs, axis=0)
            uncerts[2] = tf.math.reduce_std(box_outputs, axis=0)
        else:
            boxes = tf.expand_dims(tf.reduce_mean(box_outputs, axis=0), axis=0)
            uncerts[2] = tf.expand_dims(tf.math.reduce_std(box_outputs, axis=0), axis=0)

    else:
        boxes = anchors.decode_box_outputs(box_outputs, anchor_boxes)

    output = [boxes, uncerts, scores, classes]
    if params["enable_softmax"]:
        output.append(classes_multi)
    return output


def nms(
    params,
    boxes,
    scores,
    classes,
    padded,
    multiclass=None,
    uncerts1=None,
    uncerts2=None,
    uncerts3=None,
):
    """Non-maximum suppression.

    Args:
      params: a dict of parameters.
      boxes: a tensor with shape [N, 4], where N is the number of boxes. Box
        format is [y_min, x_min, y_max, x_max].
      scores: a tensor with shape [N].
      classes: a tensor with shape [N].
      padded: a bool vallue indicating whether the results are padded.
      uncerts1: list of tensors with shape [N, #classes] for class epistemic uncertainty
      uncerts2: list of tensors with shape [N, 4] for box aleatoric uncertainty
      uncerts3: list of tensors with shape [N, 4] for box epistemic uncertainty
      multiclass: a tensor with shape [N, #classes] representing the logits per detection

    Returns:
      A tuple (boxes, scores, classes, valid_lens), where valid_lens is a scalar
      denoting the valid length of boxes/scores/classes outputs.
      Additionally, aleatoric uncertainty and classification logits may be propagated
    """

    nms_configs = params["nms_configs"]
    method = nms_configs["method"]
    max_output_size = nms_configs["max_output_size"]

    if method == "hard" or not method:
        # Hard nms
        sigma = 0.0
        iou_thresh = nms_configs["iou_thresh"] or 0.5
        score_thresh = nms_configs["score_thresh"] or float("-inf")
    elif method == "gaussian":
        # Soft nms
        sigma = nms_configs["sigma"] or 0.5
        iou_thresh = 0.5
        score_thresh = nms_configs["score_thresh"] or 0.001
    else:
        raise ValueError("Inference has invalid nms method {}".format(method))

    # TF API's sigma is twice as the paper's value, so here we divide it by 2:
    # https://github.com/tensorflow/tensorflow/issues/40253.
    nms_top_idx, nms_scores, nms_valid_lens = tf.raw_ops.NonMaxSuppressionV5(
        boxes=boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_thresh,
        score_threshold=score_thresh,
        soft_nms_sigma=(sigma / 2),
        pad_to_max_output_size=padded,
    )

    nms_boxes = tf.gather(boxes, nms_top_idx)
    nms_classes = tf.cast(tf.gather(classes, nms_top_idx) + CLASS_OFFSET, boxes.dtype)
    # Select logits based on NMS indices
    if multiclass is not None:
        multiclass = tf.gather(multiclass, nms_top_idx)
    # Select uncertainty based on NMS indices
    if uncerts1 is not None:
        uncerts1 = tf.cast(tf.gather(uncerts1, nms_top_idx), boxes.dtype)
    if uncerts2 is not None:
        uncerts2 = tf.cast(tf.gather(uncerts2, nms_top_idx), boxes.dtype)
    if uncerts3 is not None:
        uncerts3 = tf.cast(tf.gather(uncerts3, nms_top_idx), boxes.dtype)

    output = [nms_boxes, nms_scores, nms_classes, nms_valid_lens]
    if multiclass is not None:  # Batch fn cant process None
        output.append(multiclass)
    if uncerts1 is not None:  # If one is none then its eval otherwise tf.zeros()
        output.extend([uncerts1, uncerts2, uncerts3])
    return output


def extract_uncertainties(params, cls_outputs, box_outputs):
    """Extracts uncertainties from class and box propagated by the model and performs pre NMS on the predictions

    Args:
      params: a dict of parameters.
      cls_outputs: a list of tensors for classes, each tensor denotes a level of
        logits with shape [N, H, W, num_class * num_anchors].
      box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
        boxes with shape [N, H, W, 4 * num_anchors]. Each box format is [center_y, center_x, height, width];
        May be 8 * num_anchors for uncertainty

    Returns:
        boxes, uncerts, scores, classes, classes_multi
    """
    cls_outputs = to_list(cls_outputs)
    box_outputs = to_list(box_outputs)
    uncerts = None
    if params["loss_attenuation"]:
        loss_att_boxes = []
        split_boxes = []
    if params["loss_attenuation"] or params["mc_dropout"]:
        uncerts = [None, None, None]  # Mcclass, albox, mcbox
        if params["mc_classheadrate"] or params["mc_dropoutrate"]:
            cls_outputs, uncerts[0] = get_mcuncert(cls_outputs)

        if params["loss_attenuation"]:
            for i in range(len(box_outputs)):
                split = int(box_outputs[0].shape[-1] / 2)
                if not (params["mc_boxheadrate"] or params["mc_dropoutrate"]):
                    split_boxes.append(box_outputs[i][:, :, :, :split])
                    loss_att_boxes.append(box_outputs[i][:, :, :, split:])

                elif params["loss_attenuation"] and (
                    params["mc_boxheadrate"] or params["mc_dropoutrate"]
                ):
                    split_boxes.append(box_outputs[i][:, :, :, :, :split])
                    loss_att_boxes.append(box_outputs[i][:, :, :, :, split:])

            box_outputs = split_boxes
            uncerts[1] = loss_att_boxes

    # Perform pre-NMS and NMS with logits
    if params["enable_softmax"]:
        return pre_nms(params, cls_outputs, box_outputs, uncerts=uncerts)
    else:
        pre_nms_output = pre_nms(params, cls_outputs, box_outputs, uncerts=uncerts)
        return pre_nms_output.append(None)


def postprocess_global(params, cls_outputs, box_outputs, image_scales=None):
    """Post processing with global NMS.

    A fast but less accurate version of NMS. The idea is to treat the scores for
    different classes in a unified way, and perform NMS globally for all classes.

    Args:
      params: a dict of parameters.
      cls_outputs: a list of tensors for classes, each tensor denotes a level of
        logits with shape [N, H, W, num_class * num_anchors].
      box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
        boxes with shape [N, H, W, 4 * num_anchors]. Each box format is [center_y, center_x, height, width];
        May be 8 * num_anchors for uncertainty
      image_scales: scaling factor or the final image and bounding boxes.

    Returns:
      A tuple of batch level (boxes, scores, classess, valid_len) after nms.
    """
    boxes, uncerts, scores, classes, classes_multi = extract_uncertainties(
        params, cls_outputs, box_outputs
    )  # None if no softmax
    if params["loss_attenuation"] or params["mc_dropout"]:
        nms_uncerts = uncerts.copy()
        for i in range(len(uncerts)):
            if uncerts[i] is None:
                nms_uncerts[i] = tf.zeros_like(boxes)
    else:
        nms_uncerts = [None, None, None]

    if classes_multi is not None and uncerts is not None:

        def single_batch_fn(element):
            return nms(
                params,
                element[0],
                element[1],
                element[2],
                True,
                multiclass=element[3],
                uncerts1=element[4],
                uncerts2=element[5],
                uncerts3=element[6],
            )

        (
            nms_boxes,
            nms_scores,
            nms_classes,
            nms_valid_len,
            nms_class_multi,
            nms_uncerts0,
            nms_uncerts1,
            nms_uncerts2,
        ) = batch_map_fn(
            single_batch_fn,
            [
                boxes,
                scores,
                classes,
                classes_multi,
                nms_uncerts[0],
                nms_uncerts[1],
                nms_uncerts[2],
            ],
        )

    elif classes_multi is None and uncerts is not None:

        def single_batch_fn(element):
            return nms(
                params,
                element[0],
                element[1],
                element[2],
                True,
                uncerts1=element[4],
                uncerts2=element[5],
                uncerts3=element[6],
            )

        (
            nms_boxes,
            nms_scores,
            nms_classes,
            nms_valid_len,
            nms_uncerts0,
            nms_uncerts1,
            nms_uncerts2,
        ) = batch_map_fn(
            single_batch_fn,
            [boxes, scores, classes, nms_uncerts[0], nms_uncerts[1], nms_uncerts[2]],
        )

    elif classes_multi is not None and uncerts is None:

        def single_batch_fn(element):
            return nms(
                params, element[0], element[1], element[2], True, multiclass=element[3]
            )

        nms_boxes, nms_scores, nms_classes, nms_valid_len, nms_class_multi = (
            batch_map_fn(single_batch_fn, [boxes, scores, classes, classes_multi])
        )

    else:

        def single_batch_fn(element):
            return nms(
                params,
                element[0],
                element[1],
                element[2],
                True,
            )

        nms_boxes, nms_scores, nms_classes, nms_valid_len = batch_map_fn(
            single_batch_fn, [boxes, scores, classes]
        )

    if uncerts is not None:
        if uncerts[0] is not None:
            uncerts[0] = nms_uncerts0
        if uncerts[1] is not None:
            uncerts[1] = nms_uncerts1
        if uncerts[2] is not None:
            uncerts[2] = nms_uncerts2

    nms_boxes = clip_boxes(nms_boxes, params["image_size"])
    if image_scales is not None:
        scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
        nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)
        # Scale uncertainty to original image resolution;
        # do not **2, since std not var
        if uncerts is not None:
            for i in range(len(uncerts)):
                if uncerts[i] is not None:
                    if i != 0:  # For regression only
                        uncerts[i] = uncerts[i] * tf.cast(scales, nms_boxes.dtype)
    output = [nms_boxes, nms_scores, nms_classes, nms_valid_len]
    if params["enable_softmax"]:
        output.append(nms_class_multi)
    if uncerts is not None:
        if uncerts[0] is not None:
            output[2] = tf.expand_dims(output[2], axis=-1)
            output[2] = tf.concat([output[2], uncerts[0]], -1)
        if uncerts[1] is not None:
            output[0] = tf.concat([output[0], uncerts[1]], -1)
        if uncerts[2] is not None:
            output[0] = tf.concat([output[0], uncerts[2]], -1)
    return tuple(output)


def per_class_nms(params, boxes, scores, classes, image_scales=None, logits=None):
    """Per-class nms, a utility for postprocess_per_class.

    Args:
      params: a dict of parameters.
      boxes: tensor with shape [N, K, 4], where N is batch_size, K is num_boxes.
        Box format is [y_min, x_min, y_max, x_max].
      scores: tensor with shape [N, K].
      classes: tensor with shape [N, K].
      image_scales: scaling factor or the final image and bounding boxes.
      logits: tensor with shape [N, K].

    Returns:
      A tuple of batch level (boxes, scores, classess, valid_len) after nms. May propagate the logits as well
    """

    def single_batch_fn(element):
        """A mapping function for a single batch"""
        boxes_i, scores_i, classes_i, logits = (
            element[0],
            element[1],
            element[2],
            element[3],
        )
        if logits is not None:
            nms_class_multi = []
        nms_boxes_cls, nms_scores_cls, nms_classes_cls = [], [], []
        nms_valid_len_cls = []
        for cls_num in range(params["num_classes"]):
            indices = tf.where(tf.equal(classes_i, cls_num))
            if indices.shape[0] == 0:
                continue
            classes_cls = tf.gather_nd(classes_i, indices)
            boxes_cls = tf.gather_nd(boxes_i, indices)
            scores_cls = tf.gather_nd(scores_i, indices)
            if logits is not None:
                if logits.shape[0] is not None:
                    if logits.shape[0] > 0:
                        logits = tf.gather_nd(logits, indices)
                nms_boxes, nms_scores, nms_classes, nms_valid_len, logits = nms(
                    params, boxes_cls, scores_cls, classes_cls, False, multiclass=logits
                )
                nms_class_multi.append(logits)
            else:
                nms_boxes, nms_scores, nms_classes, nms_valid_len = nms(
                    params, boxes_cls, scores_cls, classes_cls, False
                )

            nms_boxes_cls.append(nms_boxes)
            nms_scores_cls.append(nms_scores)
            nms_classes_cls.append(nms_classes)
            nms_valid_len_cls.append(nms_valid_len)
        # Pad zeros and select topk.
        max_output_size = params["nms_configs"].get("max_output_size", 100)
        nms_boxes_cls = tf.pad(
            tf.concat(nms_boxes_cls, 0), [[0, max_output_size], [0, 0]]
        )
        nms_scores_cls = tf.pad(tf.concat(nms_scores_cls, 0), [[0, max_output_size]])
        nms_classes_cls = tf.pad(tf.concat(nms_classes_cls, 0), [[0, max_output_size]])
        nms_valid_len_cls = tf.stack(nms_valid_len_cls)

        _, indices = tf.math.top_k(nms_scores_cls, k=max_output_size, sorted=True)
        output = [
            tf.gather(nms_boxes_cls, indices),
            tf.gather(nms_scores_cls, indices),
            tf.gather(nms_classes_cls, indices),
            tf.minimum(max_output_size, tf.reduce_sum(nms_valid_len_cls)),
        ]

        if logits is not None:
            nms_class_multi = tf.pad(
                tf.concat(nms_class_multi, 0), [[0, max_output_size], [0, 0]]
            )
            output.append(tf.gather(nms_class_multi, indices))
        return output

    if logits is not None:
        nms_boxes, nms_scores, nms_classes, nms_valid_len, nms_class_multi = (
            batch_map_fn(single_batch_fn, [boxes, scores, classes, logits])
        )
    else:
        nms_boxes, nms_scores, nms_classes, nms_valid_len = batch_map_fn(
            single_batch_fn, [boxes, scores, classes]
        )

    if image_scales is not None:
        scales = tf.expand_dims(tf.expand_dims(image_scales, -1), -1)
        nms_boxes = nms_boxes * tf.cast(scales, nms_boxes.dtype)

    if logits is not None:
        return nms_boxes, nms_scores, nms_classes, nms_valid_len, nms_class_multi
    else:
        return nms_boxes, nms_scores, nms_classes, nms_valid_len


def postprocess_per_class(params, cls_outputs, box_outputs, image_scales=None):
    """Post processing with per class NMS.

    An accurate but relatively slow version of NMS. The idea is to perform NMS for
    each class, and then combine them.

    Args:
      params: a dict of parameters.
      cls_outputs: a list of tensors for classes, each tensor denotes a level of
        logits with shape [N, H, W, num_class * num_anchors].
      box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
        boxes with shape [N, H, W, 4 * num_anchors]. Each box format is [y_min,
        x_min, y_max, x_man].
      image_scales: scaling factor or the final image and bounding boxes.

    Returns:
      A tuple of batch level (boxes, scores, classess, valid_len) after nms.
    """
    boxes, _, scores, classes, classes_multi = extract_uncertainties(
        params, cls_outputs, box_outputs
    )  # None if no softmax
    return per_class_nms(params, boxes, scores, classes, image_scales, classes_multi)


def generate_detections_from_nms_output(
    nms_boxes_bs,
    nms_classes_bs,
    nms_scores_bs,
    image_ids,
    original_image_widths=None,
    flip=False,
    nms_multi_class_bs=None,
):
    """Generating [id, x, y, w, h, score, class] from NMS outputs.
    May propagate the logits, one array for each class.
    """

    image_ids_bs = tf.cast(tf.expand_dims(image_ids, -1), nms_scores_bs.dtype)
    if flip:
        detections_bs = [
            image_ids_bs * tf.ones_like(nms_scores_bs),
            # The mirrored location of the left edge is the image width
            # minus the position of the right edge
            original_image_widths - nms_boxes_bs[:, :, 3],
            nms_boxes_bs[:, :, 0],
            # The mirrored location of the right edge is the image width
            # minus the position of the left edge
            original_image_widths - nms_boxes_bs[:, :, 1],
            nms_boxes_bs[:, :, 2],
            nms_scores_bs,
            nms_classes_bs,
        ]
    else:
        detections_bs = [
            image_ids_bs * tf.ones_like(nms_scores_bs),
            nms_boxes_bs[:, :, 1],
            nms_boxes_bs[:, :, 0],
            nms_boxes_bs[:, :, 3],
            nms_boxes_bs[:, :, 2],
            nms_scores_bs,
            nms_classes_bs,
        ]
    if nms_multi_class_bs is not None:
        for i in range(nms_multi_class_bs.shape[-1]):
            detections_bs.append(nms_multi_class_bs[:, :, i])

    return tf.stack(detections_bs, axis=-1, name="detections")


def generate_detections(
    params,
    cls_outputs,
    box_outputs,
    image_scales,
    image_ids,
    flip=False,
    per_class_nms=True,
):
    """A legacy interface for generating [id, x, y, w, h, score, class].
    May propagate the logits, one array for each class.
    """
    _, width = utils.parse_image_size(params["image_size"])

    original_image_widths = tf.expand_dims(image_scales, -1) * width

    if params["nms_configs"].get("pyfunc", True):
        detections_bs = []
        if params["enable_softnax"]:
            boxes, scores, classes, _ = pre_nms(params, cls_outputs, box_outputs)
        else:
            boxes, scores, classes = pre_nms(params, cls_outputs, box_outputs)
        for index in range(boxes.shape[0]):
            nms_configs = params["nms_configs"]
            detections = tf.numpy_function(
                functools.partial(nms_np.per_class_nms, nms_configs=nms_configs),
                [
                    boxes[index],
                    scores[index],
                    classes[index],
                    tf.slice(image_ids, [index], [1]),
                    tf.slice(image_scales, [index], [1]),
                    params["num_classes"],
                    nms_configs["max_output_size"],
                ],
                tf.float32,
            )

            if flip:
                detections = tf.stack(
                    [
                        detections[:, 0],
                        original_image_widths[index] - detections[:, 3],
                        detections[:, 2],
                        original_image_widths[index] - detections[:, 1],
                        detections[:, 4],
                        detections[:, 5],
                        detections[:, 6],
                    ],
                    axis=-1,
                )
            detections_bs.append(detections)
        return tf.stack(detections_bs, axis=0, name="detections")

    if per_class_nms:
        postprocess = postprocess_per_class
    else:
        postprocess = postprocess_global

    if params["enable_softmax"]:
        nms_boxes_bs, nms_scores_bs, nms_classes_bs, _, nms_class_multi_bs = (
            postprocess(params, cls_outputs, box_outputs, image_scales)
        )
        return generate_detections_from_nms_output(
            nms_boxes_bs,
            nms_classes_bs,
            nms_scores_bs,
            image_ids,
            original_image_widths,
            flip,
            nms_class_multi_bs,
        )
    else:
        nms_boxes_bs, nms_scores_bs, nms_classes_bs, _ = postprocess(
            params, cls_outputs, box_outputs, image_scales
        )
        return generate_detections_from_nms_output(
            nms_boxes_bs,
            nms_classes_bs,
            nms_scores_bs,
            image_ids,
            original_image_widths,
            flip,
        )


def transform_detections(detections):
    """A transforms detections in [id, x1, y1, x2, y2, score, class] form to [id, x, y, w, h, score, class]."""
    return tf.stack(  #
        [
            detections[:, :, 0],
            detections[:, :, 1],
            detections[:, :, 2],
            detections[:, :, 3] - detections[:, :, 1],
            detections[:, :, 4] - detections[:, :, 2],
            detections[:, :, 5],
            detections[:, :, 6],
        ],
        axis=-1,
    )
