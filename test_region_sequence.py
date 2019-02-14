from speaksee.data import TextField, ImageDetectionsField
from data import COCOControlSequenceField
from data import FlickrDetectionField, FlickrControlSequenceField
from data.dataset import COCOEntities, FlickrEntities
from models import ControllableCaptioningModel
from models import ControllableCaptioningModel_NoVisualSentinel, ControllableCaptioningModel_SingleSentinel
from speaksee.data import DataLoader, DictionaryDataset, RawField
from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer
from utils import NWNounAligner
from config import *
import torch
import random
import numpy as np
import itertools
import argparse
import os
from tqdm import tqdm

random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='ours', type=str,
                    help='model name: ours | ours_without_visual_sentinel | ours_with_single_sentinel')
parser.add_argument('--dataset', default='coco', type=str, help='dataset: coco | flickr')
parser.add_argument('--sample_rl', action='store_true', help='test the model with cider optimization')
parser.add_argument('--sample_rl_nw', action='store_true', help='test the model with cider + nw optimization')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--nb_workers', default=0, type=int, help='number of workers')
opt_test = parser.parse_args()
print(opt_test)

assert(opt_test.exp_name in ['ours', 'ours_without_visual_sentinel', 'ours_with_single_sentinel'])

if not opt_test.sample_rl and not opt_test.sample_rl_nw:
    exp_name ='%s_%s' % (opt_test.exp_name, opt_test.dataset)
    print('Loading \"%s\" model trained with cross-entropy loss.' % opt_test.exp_name)
if opt_test.sample_rl:
    exp_name = '%s_%s_%s' % (opt_test.exp_name, opt_test.dataset, 'rl')
    print('Loading \"%s\" model trained with CIDEr optimization.' % opt_test.exp_name)
if opt_test.sample_rl_nw:
    exp_name = '%s_%s_%s' % (opt_test.exp_name, opt_test.dataset, 'rl_nw')
    print('Loading \"%s\" model trained with CIDEr + NW optimization.' % opt_test.exp_name)
saved_data = torch.load('saved_models/%s/%s.pth' % (opt_test.exp_name, exp_name))
opt = saved_data['opt']

if opt_test.dataset == 'coco':
    image_field = ImageDetectionsField(detections_path=os.path.join(coco_root, 'coco_detections.hdf5'), load_in_tmp=False)

    det_field = COCOControlSequenceField(detections_path=os.path.join(coco_root, 'coco_detections.hdf5'),
                                         classes_path=os.path.join(coco_root, 'object_class_list.txt'),
                                         pad_init=False, padding_idx=-1, all_boxes=False, fix_length=20)

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True, fix_length=20)

    dataset = COCOEntities(image_field, det_field, text_field,
                           img_root='',
                           ann_root=os.path.join(coco_root, 'annotations'),
                           entities_file=os.path.join(coco_root, 'coco_entities.json'),
                           id_root=os.path.join(coco_root, 'annotations'))

    test_dataset = COCOEntities(image_field, det_field, RawField(),
                                img_root='',
                                ann_root=os.path.join(coco_root, 'annotations'),
                                entities_file=os.path.join(coco_root, 'coco_entities.json'),
                                id_root=os.path.join(coco_root, 'annotations'),
                                filtering=True)

    nw_aligner = NWNounAligner(pre_comp_file=os.path.join(coco_root, '%s_noun_glove.pkl' % opt_test.dataset), normalized=True)

elif opt_test.dataset == 'flickr':
    image_field = FlickrDetectionField(detections_path=os.path.join(flickr_root, 'flickr30k_detections.hdf5'))

    det_field = FlickrControlSequenceField(detections_path=os.path.join(flickr_root, 'flickr30k_detections.hdf5'),
                                           pad_init=False, padding_idx=-1, fix_length=20)

    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True, fix_length=20)

    dataset = FlickrEntities(image_field, text_field, det_field,
                             img_root='',
                             ann_file=os.path.join(flickr_root, 'flickr30k_annotations.json'),
                             entities_root=flickr_entities_root)

    test_dataset = FlickrEntities(image_field, RawField(), det_field,
                                  img_root='',
                                  ann_file=os.path.join(flickr_root, 'flickr30k_annotations.json'),
                                  entities_root=flickr_entities_root)

    nw_aligner = NWNounAligner(pre_comp_file=os.path.join(flickr_root, '%s_noun_glove.pkl' % opt_test.dataset), normalized=True)

else:
    raise NotImplementedError

train_dataset, val_dataset, _ = dataset.splits
text_field.build_vocab(train_dataset, val_dataset, min_freq=5)

if opt_test.exp_name == 'ours':
    model = ControllableCaptioningModel(20, len(text_field.vocab), text_field.vocab.stoi['<bos>'],
                                        h2_first_lstm=opt.h2_first_lstm, img_second_lstm=opt.img_second_lstm).to(device)
elif opt_test.exp_name == 'ours_without_visual_sentinel':
    model = ControllableCaptioningModel_NoVisualSentinel(20, len(text_field.vocab), text_field.vocab.stoi['<bos>'],
                                                         h2_first_lstm=opt.h2_first_lstm,
                                                         img_second_lstm=opt.img_second_lstm).to(device)
elif opt_test.exp_name == 'ours_with_single_sentinel':
    model = ControllableCaptioningModel_SingleSentinel(20, len(text_field.vocab), text_field.vocab.stoi['<bos>'],
                                                       h2_first_lstm=opt.h2_first_lstm,
                                                       img_second_lstm=opt.img_second_lstm).to(device)
else:
    raise NotImplementedError

_, _, test_dataset = test_dataset.splits
test_dataset = DictionaryDataset(test_dataset.examples, test_dataset.fields, 'image')
dataloader_test = DataLoader(test_dataset, batch_size=opt_test.batch_size, num_workers=opt_test.nb_workers)

model.eval()
model.load_state_dict(saved_data['state_dict'])

predictions = []
gt_captions = []
max_len = 20

with tqdm(desc='Test', unit='it', total=len(iter(dataloader_test))) as pbar:
    for it, (keys, values) in enumerate(iter(dataloader_test)):
        detections = keys
        _, _, ctrl_det_seqs_test, _, captions = values
        for i in range(detections.size(0)):
            ctrl_det_seqs_test_i = ctrl_det_seqs_test[i].numpy()
            ctrl_det_seqs_sum = np.sum(np.abs(ctrl_det_seqs_test_i), axis=-1)
            _, unique_indexes, unique_inverse = np.unique(ctrl_det_seqs_sum, axis=0, return_index=True, return_inverse=True)
            ctrl_det_seqs_test_unique = ctrl_det_seqs_test_i[unique_indexes]
            this_captions = [[captions[i][ii] for ii in range(len(unique_inverse)) if unique_inverse[ii] == jj] for jj in range(ctrl_det_seqs_test_unique.shape[0])]

            detections_i, ctrl_det_seqs_test_unique = detections[i].to(device), torch.tensor(ctrl_det_seqs_test_unique).float().to(device)
            detections_i = detections_i.unsqueeze(0).expand(ctrl_det_seqs_test_unique.size(0), detections_i.size(0), detections_i.size(1))

            out, _ = model.beam_search((detections_i, ctrl_det_seqs_test_unique), eos_idxs=[text_field.vocab.stoi['<eos>'], -1], beam_size=5)

            out = out[0].data.cpu().numpy()
            for o, caps in zip(out, this_captions):
                predictions.append(np.expand_dims(o, axis=0))
                gt_captions.append(caps)

        pbar.update()

predictions = np.concatenate(predictions, axis=0)
gen = {}
gts = {}
scores_nw = []

print("Computing sequence contrallabity results.")
for i, cap in enumerate(predictions):
    pred_cap = text_field.decode(cap, join_words=False)
    pred_cap = ' '.join([k for k, g in itertools.groupby(pred_cap)])

    score_nw = 0.
    for c in gt_captions[i]:
        score = nw_aligner.score(c, pred_cap)
        score_nw += score
    scores_nw.append(score_nw / len(gt_captions[i]))

    gts[i] = gt_captions[i]
    gen[i] = [pred_cap]


gts_t = PTBTokenizer.tokenize(gts)
gen_t = PTBTokenizer.tokenize(gen)

val_bleu, _ = Bleu(n=4).compute_score(gts_t, gen_t)
method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
for metric, score in zip(method, val_bleu):
    print(metric, score)

val_meteor, _ = Meteor().compute_score(gts_t, gen_t)
print('METEOR', val_meteor)

val_rouge, _ = Rouge().compute_score(gts_t, gen_t)
print('ROUGE_L', val_rouge)

val_cider, _ = Cider().compute_score(gts_t, gen_t)
print('CIDEr', val_cider)

val_spice, _ = Spice().compute_score(gts_t, gen_t)
print('SPICE', val_spice)

print('NW Alignment Score', np.mean(scores_nw))
