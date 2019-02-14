from speaksee import GloVe
import torch
import torch.nn.functional as F
import pickle as pkl


class NWNounAligner(object):
    def __init__(self, match_award=1, mismatch_penalty=-1, gap_penalty=-1, pre_comp_file=None, normalized=False):
        self.match_award = match_award
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.pre_comp_file = pre_comp_file
        self.normalized = normalized
        if self.pre_comp_file is not None:
            with open(self.pre_comp_file, 'rb') as fp:
                self.vectors = pkl.load(fp)
        else:
            self.vectors = GloVe()

    @staticmethod
    def zeros(shape):
        retval = []
        for x in range(shape[0]):
            retval.append([])
            for y in range(shape[1]):
                retval[-1].append(0)
        return retval

    def prep_seq(self, seq):
        seq = seq.split(' ')
        seq = [w for w in seq if w in self.vectors]
        return seq

    def match_score(self, alpha, beta):
        if alpha == beta:
            return self.match_award
        elif alpha == '-' or beta == '-':
            return self.gap_penalty
        else:
            a = self.vectors[alpha]
            b = self.vectors[beta]
            if self.pre_comp_file is not None:
                a = torch.from_numpy(a)
                b = torch.from_numpy(b)

            return torch.mean(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))).unsqueeze(-1)

    def finalize(self, align1, align2):
        align1 = align1[::-1]  # reverse sequence 1
        align2 = align2[::-1]  # reverse sequence 2

        # calcuate identity, score and aligned sequeces
        symbol = []
        score = 0
        identity = 0
        for i in range(0, len(align1)):
            # if two AAs are the same, then output the letter
            if align1[i] == align2[i]:
                symbol.append(align1[i])
                identity = identity + 1
                score += self.match_score(align1[i], align2[i])

            # if they are not identical and none of them is gap
            elif align1[i] != align2[i] and align1[i] != '-' and align2[i] != '-':
                score += self.match_score(align1[i], align2[i])
                symbol.append(' ')

            # if one of them is a gap, output a space
            elif align1[i] == '-' or align2[i] == '-':
                symbol.append(' ')
                score += self.gap_penalty

        return score

    def score(self, seq1, seq2):
        seq1 = self.prep_seq(seq1)
        seq2 = self.prep_seq(seq2)
        m, n = len(seq1), len(seq2)  # length of two sequences

        # Generate DP table and traceback path pointer matrix
        score = self.zeros((m + 1, n + 1))  # the DP table

        # Calculate DP table
        for i in range(0, m + 1):
            score[i][0] = self.gap_penalty * i
        for j in range(0, n + 1):
            score[0][j] = self.gap_penalty * j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score[i - 1][j - 1] + self.match_score(seq1[i - 1], seq2[j - 1])
                delete = score[i - 1][j] + self.gap_penalty
                insert = score[i][j - 1] + self.gap_penalty
                score[i][j] = max(match, delete, insert)

        # Traceback and compute the alignment
        align1, align2 = [], []
        i, j = m, n  # start from the bottom right cell
        while i > 0 and j > 0:  # end toching the top or the left edge
            score_current = score[i][j]
            score_diagonal = score[i - 1][j - 1]
            score_up = score[i][j - 1]
            score_left = score[i - 1][j]

            if score_current == score_diagonal + self.match_score(seq1[i - 1], seq2[j - 1]):
                align1.append(seq1[i - 1])
                align2.append(seq2[j - 1])
                i -= 1
                j -= 1
            elif score_current == score_left + self.gap_penalty:
                align1.append(seq1[i - 1])
                align2.append('-')
                i -= 1
            elif score_current == score_up + self.gap_penalty:
                align1.append('-')
                align2.append(seq2[j - 1])
                j -= 1

        # Finish tracing up to the top left cell
        while i > 0:
            align1.append(seq1[i - 1])
            align2.append('-')
            i -= 1
        while j > 0:
            align1.append('-')
            align2.append(seq2[j - 1])
            j -= 1

        nw_score = torch.zeros((1,)) + self.finalize(align1, align2)
        if self.normalized:
            nb_nouns = max(m, n)
            if nb_nouns > 0:
                return nw_score / nb_nouns
            else:
                return nw_score
        else:
            return nw_score
