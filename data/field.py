import torch
import numpy as np
import h5py
import pickle as pkl
import warnings
import json
from itertools import groupby
from speaksee.data import RawField


class COCOControlSequenceField(RawField):
    def __init__(self, postprocessing=None, detections_path=None, classes_path=None,
                 padding_idx=0, fix_length=None, all_boxes=True, pad_init=True, pad_eos=True, dtype=torch.float32,
                 max_detections=20, max_length=100, sorting=False):
        self.max_detections = max_detections
        self.max_length = max_length
        self.detections_path = detections_path
        self.padding_idx = padding_idx
        self.fix_length = fix_length
        self.all_boxes = all_boxes
        self.sorting = sorting
        self.eos_token = padding_idx if pad_eos else None
        self.dtype = dtype

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        super(COCOControlSequenceField, self).__init__(None, postprocessing)

    def get_detections_inside(self, det_boxes, query):
        cond1 = det_boxes[:, 0] >= det_boxes[query, 0]
        cond2 = det_boxes[:, 1] >= det_boxes[query, 1]
        cond3 = det_boxes[:, 2] <= det_boxes[query, 2]
        cond4 = det_boxes[:, 3] <= det_boxes[query, 3]
        cond = cond1 & cond2 & cond3 & cond4
        return np.nonzero(cond)[0]

    def _fill(self, cls_seq, det_features, det_boxes, selected_classes, most_probable_dets, max_len):
        det_sequences = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        for j, cls in enumerate(cls_seq[:max_len]):
            if cls == '_':
                det_sequences[j, :det_features.shape[0]] = most_probable_dets
            else:
                seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]
                if self.all_boxes:
                    det_ids = np.unique(np.concatenate([self.get_detections_inside(det_boxes, d) for d in seed_detections]))
                else:
                    det_ids = np.unique(seed_detections)
                det_sequences[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]

        if not self.sorting:
            last = len(cls_seq[:max_len])
            det_sequences[last:] = det_sequences[last-1]

        return det_sequences.astype(np.float32)

    def preprocess(self, x):
        image = x[0]
        det_classes = x[1]
        max_len = self.fix_length + (self.eos_token, self.eos_token).count(None) - 2

        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_boxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        most_probable_idxs = np.argsort(np.max(det_cls_probs, -1))[::-1][:self.max_detections]
        most_probable_dets = det_features[most_probable_idxs]

        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:])+1] for i in range(len(det_cls_probs))]

        cls_seq = []
        for i, cls in enumerate(det_classes):
            if cls is not None:
                cls_seq.append(cls)
            else:
                cls_ok = next((c for c in det_classes[i+1:] if c is not None), '_')
                cls_seq.append(cls_ok)

        cls_seq_gt = np.asarray([int(a != b) for (a, b) in zip(cls_seq[:-1], cls_seq[1:])] + [0, ])
        cls_seq_gt = cls_seq_gt[:max_len]
        cls_seq_gt = np.concatenate([cls_seq_gt, [self.eos_token, self.eos_token]])
        cls_seq_gt = np.concatenate([cls_seq_gt, [self.padding_idx]*max(0, self.fix_length - len(cls_seq_gt))])
        cls_seq_gt = cls_seq_gt.astype(np.float32)

        cls_seq_test = [x[0] for x in groupby(det_classes) if x[0] is not None]
        if self.sorting:
            cls_seq_test.sort()
            det_sequences_test = self._fill(cls_seq_test, det_features, det_boxes, selected_classes, most_probable_dets, max_len)
            return det_sequences_test
        else:
            det_sequences = self._fill(cls_seq, det_features, det_boxes, selected_classes, most_probable_dets, max_len)
            det_sequences_test = self._fill(cls_seq_test, det_features, det_boxes, selected_classes, most_probable_dets, max_len)

            cls_seq_test = ' '.join(cls_seq_test)

            return det_sequences, cls_seq_gt, det_sequences_test, cls_seq_test # , id_image


class COCOControlSetField(RawField):
    def __init__(self, postprocessing=None, detections_path=None, classes_path=None, img_shapes_path=None,
                 precomp_glove_path=None, fix_length=20, max_detections=20):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        super(COCOControlSetField, self).__init__(None, postprocessing)

    def preprocess(self, x):
        image = x[0]
        det_classes = x[1]

        id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_boxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_boxes = np.random.rand(10, 4)

        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))]
        width, height = self.img_shapes[str(id_image)]

        cls_seq = [x[0] for x in groupby(det_classes) if x[0] is not None]
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))

        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        cls_seq = cls_seq[:self.fix_length]
        cls_seq.sort()

        for j, cls in enumerate(cls_seq):
            cls_w = cls.split(',')[0].split(' ')[-1]
            if cls_w in self.vectors:
                det_sequences_word[j] = self.vectors[cls_w]
            seed_detections = [i for i, c in enumerate(selected_classes) if c == cls]
            det_ids = np.unique(seed_detections)
            det_sequences_visual_all[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]

            det_sequences_visual[j] = det_features[det_ids[0]]
            bbox = det_boxes[det_ids[0]]
            det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width
            det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height
            det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width
            det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height

        return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
               det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32)


class FlickrDetectionField(RawField):
    def __init__(self, preprocessing=None, postprocessing=None, detections_path=None):
        self.max_detections = 100
        self.detections_path = detections_path

        super(FlickrDetectionField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split('/')[-1].split('.')[0])
        try:
            precomp_data = h5py.File(self.detections_path, 'r')['%d_features' % image_id][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % image_id)
            precomp_data = np.random.rand(10, 2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return precomp_data.astype(np.float32)


class FlickrControlSequenceField(RawField):
    def __init__(self, postprocessing=None, detections_path=None,
                 padding_idx=0, fix_length=None, pad_init=True, pad_eos=True, dtype=torch.float32,
                 max_detections=20, max_length=100):
        self.detections_path = detections_path
        self.max_detections = max_detections
        self.max_length = max_length
        self.detections_path = detections_path
        self.padding_idx = padding_idx
        self.fix_length = fix_length
        self.eos_token = padding_idx if pad_eos else None
        self.dtype = dtype

        super(FlickrControlSequenceField, self).__init__(None, postprocessing)

    @staticmethod
    def _bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    def _fill(self, cls_seq, det_features, bbox_ids, most_probable_dets, max_len):
        det_sequences = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))
        for j, cls in enumerate(cls_seq[:max_len]):
            if cls == '_':
                det_sequences[j, :det_features.shape[0]] = most_probable_dets
            else:
                det_ids = bbox_ids[cls]
                det_sequences[j, :len(det_ids)] = np.take(det_features, det_ids, axis=0)[:self.max_detections]

        last = len(cls_seq[:max_len])
        det_sequences[last:] = det_sequences[last-1]
        return det_sequences.astype(np.float32)

    def preprocess(self, x, avoid_precomp=False):
        image = x[0]
        gt_bboxes = x[1]
        det_classes = x[2]
        max_len = self.fix_length + (self.eos_token, self.eos_token).count(None) - 2

        id_image = image.split('/')[-1].split('.')[0]

        try:
            f = h5py.File(self.detections_path, 'r')
            det_features = f['%s_features' % id_image][()]
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_bboxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_features = np.random.rand(10, 2048)
            det_cls_probs = np.random.rand(10, 2048)
            det_bboxes = np.random.rand(10, 4)

        det_classes = [d-1 if d > 0 else None for d in det_classes]

        most_probable_idxs = np.argsort(np.max(det_cls_probs, -1))[::-1][:self.max_detections]
        most_probable_dets = det_features[most_probable_idxs]

        cls_seq = []
        for i, cls in enumerate(det_classes):
            if cls is not None:
                cls_seq.append(cls)
            else:
                cls_ok = next((c for c in det_classes[i + 1:] if c is not None), '_')
                cls_seq.append(cls_ok)

        cls_seq_gt = np.asarray([int(a != b) for (a, b) in zip(cls_seq[:-1], cls_seq[1:])] + [0, ])
        cls_seq_gt = cls_seq_gt[:max_len]
        cls_seq_gt = np.concatenate([cls_seq_gt, [self.eos_token, self.eos_token]])
        cls_seq_gt = np.concatenate([cls_seq_gt, [self.padding_idx] * max(0, self.fix_length - len(cls_seq_gt))])
        cls_seq_gt = cls_seq_gt.astype(np.float32)

        cls_seq_test = [x[0] for x in groupby(det_classes) if x[0] is not None]

        bbox_ids = dict()
        for i, cls in enumerate(set(cls_seq_test)):
            id_boxes = []
            for k, bbox in enumerate(gt_bboxes[cls]):
                id_bbox = -1
                iou_max = 0
                for ii, det_bbox in enumerate(det_bboxes):
                    iou = self._bb_intersection_over_union(bbox, det_bbox)
                    if iou_max < iou:
                        id_bbox = ii
                        iou_max = iou
                id_boxes.append(id_bbox)
            bbox_ids[cls] = id_boxes

        det_sequences = self._fill(cls_seq, det_features, bbox_ids, most_probable_dets, max_len)
        det_sequences_test = self._fill(cls_seq_test, det_features, bbox_ids, most_probable_dets, max_len)

        cls_seq_test = [str(c) for c in cls_seq_test]
        cls_seq_test = ' '.join(cls_seq_test)

        return det_sequences, cls_seq_gt, det_sequences_test, cls_seq_test


class FlickrControlSetField(RawField):
    def __init__(self, postprocessing=None, detections_path=None, classes_path=None, img_shapes_path=None,
                 precomp_glove_path=None, fix_length=20, max_detections=20):
        self.fix_length = fix_length
        self.detections_path = detections_path
        self.max_detections = max_detections

        self.classes = ['__background__']
        with open(classes_path, 'r') as f:
            for object in f.readlines():
                self.classes.append(object.split(',')[0].lower().strip())

        with open(precomp_glove_path, 'rb') as fp:
            self.vectors = pkl.load(fp)

        with open(img_shapes_path, 'r') as fp:
            self.img_shapes = json.load(fp)

        super(FlickrControlSetField, self).__init__(None, postprocessing)

    @staticmethod
    def _bb_intersection_over_union(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    def preprocess(self, x):
        image = x[0]
        gt_bboxes = x[1]
        det_classes = x[2]

        id_image = image.split('/')[-1].split('.')[0]
        try:
            f = h5py.File(self.detections_path, 'r')
            det_cls_probs = f['%s_cls_prob' % id_image][()]
            det_features = f['%s_features' % id_image][()]
            det_bboxes = f['%s_boxes' % id_image][()]
        except KeyError:
            warnings.warn('Could not find detections for %d' % id_image)
            det_cls_probs = np.random.rand(10, 2048)
            det_features = np.random.rand(10, 2048)
            det_bboxes = np.random.rand(10, 4)

        det_classes = [d - 1 if d > 0 else None for d in det_classes]
        selected_classes = [self.classes[np.argmax(det_cls_probs[i][1:]) + 1] for i in range(len(det_cls_probs))]
        width, height = self.img_shapes[str(id_image)]

        cls_seq = [x[0] for x in groupby(det_classes) if x[0] is not None]
        det_sequences_visual_all = np.zeros((self.fix_length, self.max_detections, det_features.shape[-1]))

        det_sequences_visual = np.zeros((self.fix_length, det_features.shape[-1]))
        det_sequences_word = np.zeros((self.fix_length, 300))
        det_sequences_position = np.zeros((self.fix_length, 4))

        cls_seq = cls_seq[:self.fix_length]
        cls_seq.sort()

        for j, cls in enumerate(cls_seq):
            id_boxes = []
            for k, bbox in enumerate(gt_bboxes[cls]):
                id_bbox = -1
                iou_max = 0
                for ii, det_bbox in enumerate(det_bboxes):
                    iou = self._bb_intersection_over_union(bbox, det_bbox)
                    if iou_max < iou:
                        id_bbox = ii
                        iou_max = iou
                id_boxes.append(id_bbox)

            id_boxes.sort()

            cls_w = selected_classes[id_boxes[0]].split(',')[0].split(' ')[-1]
            if cls_w in self.vectors:
                det_sequences_word[j] = self.vectors[cls_w]

            det_sequences_visual_all[j, :len(id_boxes)] = np.take(det_features, id_boxes, axis=0)[:self.max_detections]
            det_sequences_visual[j] = det_features[id_boxes[0]]

            bbox = det_bboxes[id_boxes[0]]
            det_sequences_position[j, 0] = (bbox[2] - bbox[0] / 2) / width
            det_sequences_position[j, 1] = (bbox[3] - bbox[1] / 2) / height
            det_sequences_position[j, 2] = (bbox[2] - bbox[0]) / width
            det_sequences_position[j, 3] = (bbox[3] - bbox[1]) / height

        return det_sequences_word.astype(np.float32), det_sequences_visual.astype(np.float32), \
               det_sequences_position.astype(np.float32), det_sequences_visual_all.astype(np.float32)

