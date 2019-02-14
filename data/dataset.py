import os
import json
import numpy as np
import re
import xml.etree.ElementTree
from speaksee.data import field, Example, PairedDataset, COCO
from speaksee.utils import nostdout
from itertools import groupby
import pickle as pkl


class COCOEntities(PairedDataset):
    def __init__(self, image_field, det_field, text_field, img_root, ann_root, entities_file, id_root=None,
                 use_restval=True, filtering=False):
        roots = dict()
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['val'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['val']['img']),
            'cap': (roots['train']['cap'], roots['val']['cap'])
        }

        if id_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(id_root, 'coco_train_ids.npy'))
            ids['val'] = np.load(os.path.join(id_root, 'coco_dev_ids.npy'))
            ids['test'] = np.load(os.path.join(id_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (
                ids['train'],
                np.load(os.path.join(id_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        if not filtering:
            dataset_path = 'saved_data/coco/coco_entities_precomp.pkl'
        else:
            dataset_path = 'saved_data/coco/coco_entities_filtered_precomp.pkl'
        if not os.path.isfile(dataset_path):
            with nostdout():
                train_examples, val_examples, test_examples = COCO.get_samples(roots, ids)

            self.train_examples, self.val_examples, self.test_examples = self.get_samples([train_examples, val_examples, test_examples], entities_file, filtering)
            pkl.dump((self.train_examples, self.val_examples, self.test_examples), open(dataset_path, 'wb'), -1)
        else:
            self.train_examples, self.val_examples, self.test_examples = pkl.load(open(dataset_path, 'rb'))

        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCOEntities, self).__init__(examples, {'image': image_field, 'detection': det_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, samples, entities_file, filtering=False):
        train_examples = []
        val_examples = []
        test_examples = []

        with open(entities_file, 'r') as fp:
            visual_chunks = json.load(fp)

        for id_split, samples_split in enumerate(samples):
            for s in samples_split:
                id_image = str(int(s.image.split('/')[-1].split('_')[-1].split('.')[0]))
                caption = s.text.lower().replace('\t', ' ').replace('\n', '')

                words = caption.strip().split(' ')
                caption_fixed = []
                for w in words:
                    if w not in field.TextField.punctuations and w != '':
                        caption_fixed.append(w)

                det_classes = [None for _ in caption_fixed]
                caption_fixed = ' '.join(caption_fixed)

                for p in field.TextField.punctuations:
                    caption_fixed = caption_fixed.replace(p, '')

                if id_image in visual_chunks:
                    if caption in visual_chunks[id_image]:
                        chunks = visual_chunks[id_image][caption]
                        for chunk in chunks:
                            words = chunk[0].split(' ')
                            chunk_fixed = []
                            for w in words:
                                if w not in field.TextField.punctuations and w != '':
                                    chunk_fixed.append(w)
                            chunk_fixed = ' '.join(chunk_fixed)
                            for p in field.TextField.punctuations:
                                chunk_fixed = chunk_fixed.replace(p, '')

                            sub_str = ' '.join(['_' for _ in chunk_fixed.split(' ')])
                            sub_cap = caption_fixed.replace(chunk_fixed, sub_str).split(' ')
                            for i, w in enumerate(sub_cap):
                                if w == '_':
                                    det_classes[i] = chunk[1]

                        example = Example.fromdict({'image': s.image,
                                                    'detection': (s.image, tuple(det_classes)),
                                                    'text': caption_fixed})

                        det_classes_set = [x[0] for x in groupby(det_classes) if x[0] is not None]
                        chunks_filtered = list(set([c[1] for c in chunks]))
                        if len(det_classes_set) < len(chunks_filtered):
                            pass
                        else:
                            if id_split == 0:
                                train_examples.append(example)
                            elif id_split == 1:
                                if filtering:
                                    if '_' not in example.detection[1]:
                                        val_examples.append(example)
                                else:
                                    val_examples.append(example)
                            elif id_split == 2:
                                if filtering:
                                    if '_' not in example.detection[1]:
                                        test_examples.append(example)
                                else:
                                    test_examples.append(example)

        return train_examples, val_examples, test_examples


class FlickrEntities(PairedDataset):
    def __init__(self, image_field, text_field, det_field, img_root, ann_file, entities_root,
                 precomp_file='saved_data/flickr/flickr_entities_precomp.pkl'):
        if os.path.isfile(precomp_file):
            with open(precomp_file, 'rb') as pkl_file:
                self.train_examples, self.val_examples, self.test_examples = pkl.load(pkl_file)
        else:
            self.train_examples, self.val_examples, self.test_examples = self.get_samples(ann_file, img_root, entities_root)
            with open(precomp_file, 'wb') as pkl_file:
                pkl.dump((self.train_samples, self.val_samples, self.test_samples), pkl_file, protocol=pkl.HIGHEST_PROTOCOL)

        examples = self.train_examples + self.val_examples + self.test_examples
        super(FlickrEntities, self).__init__(examples, {'image': image_field, 'detection': det_field, 'text': text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    def get_samples(self, ann_file, img_root, entities_root):
        def _get_sample(d):
            filename = d['filename']
            split = d['split']
            xml_root = xml.etree.ElementTree.parse(os.path.join(entities_root, 'Annotations',
                                                                filename.replace('.jpg', '.xml'))).getroot()
            det_dict = dict()
            id_counter = 1
            for obj in xml_root.findall('object'):
                obj_names = [o.text for o in obj.findall('name')]
                if obj.find('bndbox'):
                    bbox = tuple(int(o.text) for o in obj.find('bndbox'))
                    for obj_name in obj_names:
                        if obj_name not in det_dict:
                            det_dict[obj_name] = {'id': id_counter, 'bdnbox': [bbox]}
                            id_counter += 1
                        else:
                            det_dict[obj_name]['bdnbox'].append(bbox)

            bdnboxes = [[] for _ in range(id_counter - 1)]
            for it in det_dict.values():
                bdnboxes[it['id'] - 1] = tuple(it['bdnbox'])
            bdnboxes = tuple(bdnboxes)

            captions = [l.strip() for l in open(os.path.join(entities_root, 'Sentences',
                                                             filename.replace('.jpg', '.txt')), encoding="utf-8").readlines()]
            outputs = []
            for c in captions:
                matches = prog.findall(c)
                caption = []
                det_ids = []

                for match in matches:
                    for i, grp in enumerate(match):
                        if i in (0, 2):
                            if grp != '':
                                words = grp.strip().split(' ')
                                for w in words:
                                    if w not in field.TextField.punctuations and w != '':
                                        caption.append(w)
                                        det_ids.append(0)
                        elif i == 1:
                            words = grp[1:-1].strip().split(' ')
                            obj_name = words[0].split('#')[-1].split('/')[0]
                            words = words[1:]
                            for w in words:
                                if w not in field.TextField.punctuations and w != '':
                                    caption.append(w)
                                    if obj_name in det_dict:
                                        det_ids.append(det_dict[obj_name]['id'])
                                    else:
                                        det_ids.append(0)

                caption = ' '.join(caption)
                if caption != '' and np.sum(np.asarray(det_ids)) > 0:
                    example = Example.fromdict({'image': os.path.join(img_root, filename),
                                                'detection': (os.path.join(img_root, filename), bdnboxes, det_ids),
                                                'text': caption})
                    outputs.append([example, split])

            return outputs

        train_samples = []
        val_samples = []
        test_samples = []

        prog = re.compile(r'([^\[\]]*)(\[[^\[\]]+\])([^\[\]]*)')
        dataset = json.load(open(ann_file, 'r'))['images']

        samples = []
        for d in dataset:
            samples.extend(_get_sample(d))

        for example, split in samples:
            if split == 'train':
                train_samples.append(example)
            elif split == 'val':
                val_samples.append(example)
            elif split == 'test':
                test_samples.append(example)

        return train_samples, val_samples, test_samples
