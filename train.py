from speaksee.data import TextField, ImageDetectionsField
from data import COCOControlSequenceField
from data.dataset import COCOEntities, PairedDataset
from models import ControllableCaptioningModel
from speaksee.data import DataLoader, RawField
from speaksee import evaluation
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import NLLLoss
from utils import NWNounAligner
from config import *
import torch
import random
import argparse
import itertools
import numpy as np
import os
from tqdm import tqdm

random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='show_control_and_tell', type=str, help='experiment name')
parser.add_argument('--nb_workers', default=0, type=int, help='number of workers')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--step_size', default=3, type=int, help='learning rate schedule step size')
parser.add_argument('--gamma', default=0.8, type=float, help='learning rate schedule gamma')
parser.add_argument('--h2_first_lstm', default=1, type=int, help='h2 as input to the first lstm')
parser.add_argument('--img_second_lstm', default=0, type=int, help='img vector as input to the second lstm')
parser.add_argument('--sample_rl', action='store_true', help='reinforcement learning with cider optimization')
parser.add_argument('--sample_rl_nw', action='store_true', help='reinforcement learning with cider + nw optimization')
opt = parser.parse_args()
print(opt)

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

train_dataset, val_dataset, _ = dataset.splits
text_field.build_vocab(train_dataset, val_dataset, min_freq=5)

test_dataset = COCOEntities(image_field, det_field, RawField(),
                            img_root='',
                            ann_root=os.path.join(coco_root, 'annotations'),
                            entities_file=os.path.join(coco_root, 'coco_entities.json'),
                            id_root=os.path.join(coco_root, 'annotations'),
                            filtering=True)

_, val_dataset, _ = test_dataset.splits

if opt.sample_rl or opt.sample_rl_nw:
    train_dataset.fields['text'] = RawField()
    train_dataset_raw = PairedDataset(train_dataset.examples, {'image': image_field, 'detection': det_field, 'text': RawField()})
    ref_caps_train = list(train_dataset_raw.text)
    cider_train = evaluation.Cider(evaluation.PTBTokenizer.tokenize(ref_caps_train))

dataloader_train = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nb_workers)

val_dataset.fields['text'] = RawField()
dataloader_val = DataLoader(val_dataset, batch_size=16, num_workers=opt.nb_workers)

model = ControllableCaptioningModel(20, len(text_field.vocab), text_field.vocab.stoi['<bos>'],
                                    h2_first_lstm=opt.h2_first_lstm, img_second_lstm=opt.img_second_lstm).to(device)

optim = Adam(model.parameters(), lr=opt.lr)
scheduler = StepLR(optim, step_size=opt.step_size, gamma=opt.gamma)
loss_fn = NLLLoss()
loss_fn_gate = NLLLoss(ignore_index=-1)

start_epoch = 0
best_cider = .0
patience = 0
if opt.sample_rl or opt.sample_rl_nw:
    saved_data = torch.load('saved_models/%s_best.pth' % opt.exp_name)
    print("Loading from epoch %d, with validation CIDER %.02f" % (saved_data['epoch'], saved_data['val_cider']))
    start_epoch = saved_data['epoch'] + 1
    model.load_state_dict(saved_data['state_dict'])
    best_cider = saved_data['best_cider']

if opt.sample_rl_nw:
    nw_aligner = NWNounAligner(pre_comp_file=os.path.join(coco_root, 'coco_noun_glove.pkl'), normalized=True)


for e in range(start_epoch, start_epoch+100):
    if not opt.sample_rl and not opt.sample_rl_nw:
        # Training with cross-entropy
        model.train()
        running_loss = .0
        running_loss_gate = .0
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(iter(dataloader_train))) as pbar:
            for it, (detections, ctrl_det_seqs, ctrl_det_gts, _, _, captions) in enumerate(iter(dataloader_train)):
                detections, ctrl_det_seqs = detections.to(device), ctrl_det_seqs.to(device)
                ctrl_det_gts, captions = ctrl_det_gts.to(device), captions.to(device)

                out, gate = model((detections, ), (captions, ctrl_det_seqs))

                optim.zero_grad()
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss_cap = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                loss_gate = loss_fn_gate(gate.view(-1, 2), ctrl_det_gts.view(-1).long())
                loss = loss_cap + 4*loss_gate

                loss.backward()
                optim.step()

                running_loss += loss_cap.item()
                running_loss_gate += loss_gate.item()
                pbar.set_postfix(loss_cap=running_loss / (it+1), loss_gate=running_loss_gate / (it+1))
                pbar.update()

        scheduler.step()
    else:
        # Baseline captions
        model.eval()
        baselines = []
        with tqdm(desc='Epoch %d - baseline' % e, unit='it', total=len(iter(dataloader_train))) as pbar:
            with torch.no_grad():
                for it, (detections, ctrl_det_seqs, ctrl_det_gts, ctrl_det_seqs_test, _, caps_gt) in enumerate(iter(dataloader_train)):
                    detections, ctrl_det_seqs_test = detections.to(device), ctrl_det_seqs_test.to(device)
                    outs_baseline, _ = model.test(detections, ctrl_det_seqs_test)

                    caps_baseline = text_field.decode(outs_baseline.cpu().numpy(), join_words=False)

                    bas = []
                    for i, bas_i in enumerate(caps_baseline):
                        bas_i = ' '.join([k for k, g in itertools.groupby(bas_i)])
                        bas.append(bas_i)
                    baselines.append(bas)
                    pbar.update()

        # Training with self-critical
        model.train()
        running_loss = .0
        running_loss_gate = .0
        running_loss_nw = .0
        running_reward = .0
        running_reward_nw = .0
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(iter(dataloader_train))) as pbar:
            for it, (detections, ctrl_det_seqs, ctrl_det_gts, ctrl_det_seqs_test, _, caps_gt) in enumerate(iter(dataloader_train)):
                detections, ctrl_det_seqs = detections.to(device), ctrl_det_seqs.to(device)
                ctrl_det_gts, ctrl_det_seqs_test = ctrl_det_gts.to(device), ctrl_det_seqs_test.to(device)
                outs, log_probs = model.sample_rl(detections, ctrl_det_seqs_test)
                optim.zero_grad()

                caps_gen = text_field.decode(outs[0].detach().cpu().numpy(), join_words=False)

                gts = []
                gen = []
                scores = []
                scores_baseline = []
                if not opt.sample_rl_nw:
                    for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                        gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                        gts.append([gts_i, ])
                        gen.append(gen_i)
                else:
                    for i, (gts_i, bas_i, gen_i) in enumerate(zip(caps_gt, baselines[it], caps_gen)):
                        gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                        gts.append([gts_i, ])
                        gen.append(gen_i)
                        scores.append((1 + nw_aligner.score(gts_i, gen_i)) / 2.)
                        scores_baseline.append((1 + nw_aligner.score(gts_i, bas_i)) / 2.)

                caps_gt = evaluation.PTBTokenizer.tokenize(gts)
                caps_gen = evaluation.PTBTokenizer.tokenize(gen)
                caps_baseline = evaluation.PTBTokenizer.tokenize(baselines[it])

                reward = cider_train.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                reward_baseline = cider_train.compute_score(caps_gt, caps_baseline)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(device)
                reward_baseline = torch.from_numpy(reward_baseline).to(device)

                if not opt.sample_rl_nw:
                    loss_cap = -(torch.mean(log_probs[0], -1) + torch.mean(log_probs[1], -1)) * (reward - reward_baseline)
                    loss_cap = loss_cap.mean()
                    loss = loss_cap
                    loss.backward()
                    optim.step()

                    running_loss += loss_cap.item()
                    running_reward += torch.mean(reward - reward_baseline).item()
                    pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1))
                    pbar.update()
                else:
                    reward_nw = torch.cat(scores).to(device)
                    reward_nw_baseline = torch.cat(scores_baseline).to(device)

                    loss_cap = -(torch.mean(log_probs[0], -1) + torch.mean(log_probs[1], -1)) * (reward - reward_baseline)
                    loss_cap = loss_cap.mean()
                    loss_nw = -(torch.mean(log_probs[0], -1) + torch.mean(log_probs[1], -1)) * (reward_nw - reward_nw_baseline)
                    loss_nw = loss_nw.mean()
                    loss = loss_cap + 4*loss_nw
                    loss.backward()
                    optim.step()

                    running_loss += loss_cap.item()
                    running_loss_nw += loss_nw.item()
                    running_reward += torch.mean(reward - reward_baseline).item()
                    running_reward_nw += torch.mean(reward_nw - reward_nw_baseline).item()
                    pbar.set_postfix(loss=running_loss / (it + 1), loss_nw=running_loss_nw / (it + 1),
                                     reward=running_reward / (it + 1), reward_nw=running_reward_nw / (it + 1))
                    pbar.update()

    # Validation with CIDEr
    gen = []
    gts = []
    max_len = 20
    model.eval()
    with tqdm(desc='Test', unit='it', total=len(iter(dataloader_val))) as pbar:
        with torch.no_grad():
            for it, (detections, ctrl_det_seqs, ctrl_det_gts, ctrl_det_seqs_test, _, captions) in enumerate(iter(dataloader_val)):
                detections, ctrl_det_seqs = detections.to(device), ctrl_det_seqs.to(device)
                ctrl_det_seqs_test = ctrl_det_seqs_test.to(device)
                out, gate = model.test(detections, ctrl_det_seqs_test)

                caps_gen = text_field.decode(out, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(captions, caps_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen.append(gen_i)
                    gts.append([gts_i, ])
                pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    val_bleu, _ = evaluation.Bleu(n=4).compute_score(gts, gen)
    method = ['Blue_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    for metric, score in zip(method, val_bleu):
        print(metric, score)

    val_meteor, _ = evaluation.Meteor().compute_score(gts, gen)
    print('METEOR', val_meteor)

    val_rouge, _ = evaluation.Rouge().compute_score(gts, gen)
    print('ROUGE_L', val_rouge)

    val_cider, _ = evaluation.Cider().compute_score(gts, gen)
    print('CIDEr', val_cider)

    saved_data = {
        'epoch': e,
        'opt': opt,
        'val_cider': val_cider,
        'patience': patience,
        'best_cider': best_cider,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    if not os.path.exists('saved_models/'):
        os.makedirs('saved_models/')

    if val_cider >= best_cider:
        best_cider = val_cider
        best_srt = 'best_rl' if opt.sample_rl else 'best'
        best_srt = 'best_rl_nw' if opt.sample_rl_nw else best_srt
        patience = 0
        saved_data['best_cider'] = best_cider
        saved_data['patience'] = patience
        torch.save(saved_data, 'saved_models/%s_%s.pth' % (opt.exp_name, best_srt))
    else:
        patience += 1
        saved_data['patience'] = patience
    torch.save(saved_data, 'saved_models/%s_last.pth' % opt.exp_name)

    if patience == 5:
        print('patience ended.')
        break
