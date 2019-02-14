import torch
import torch.nn.functional as F
import pickle as pkl
import munkres


class NounIoU(object):
    def __init__(self, pre_comp_file):
        self.pre_comp_file = pre_comp_file
        self.munkres = munkres.Munkres()

        with open(self.pre_comp_file, 'rb') as fp:
            self.vectors = pkl.load(fp)

    def prep_seq(self, seq):
        seq = seq.split(' ')
        seq = [w for w in seq if w in self.vectors]
        return seq

    def score(self, seq_gt, seq_pred):
        seq_gt = self.prep_seq(seq_gt)
        seq_pred = self.prep_seq(seq_pred)
        m, n = len(seq_gt), len(seq_pred)  # length of two sequences

        if m == 0:
            return 1.
        if n == 0:
            return 0.

        similarities = torch.zeros((m, n))
        for i in range(m):
            for j in range(n):
                a = self.vectors[seq_gt[i]]
                b = self.vectors[seq_pred[j]]
                a = torch.from_numpy(a)
                b = torch.from_numpy(b)
                similarities[i, j] = torch.mean(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))).unsqueeze(-1)

        similarities = (similarities + 1) / 2
        similarities = similarities.numpy()
        ass = self.munkres.compute(munkres.make_cost_matrix(similarities))

        intersection_score = .0
        for a in ass:
            intersection_score += similarities[a]
        iou_score = intersection_score / (m + n - intersection_score)

        return iou_score
