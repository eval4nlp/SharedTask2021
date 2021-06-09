import argparse
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr


def read_sentence_data(gold_sent_fh, model_sent_fh):
    gold_scores = [float(line.strip()) for line in gold_sent_fh]
    model_scores = [float(line.strip()) for line in model_sent_fh]
    assert len(gold_scores) == len(model_scores)
    return gold_scores, model_scores


def read_word_data(gold_explanations_fh, model_explanations_fh):
    gold_explanations = [list(map(int, line.split())) for line in gold_explanations_fh]
    model_explanations = [list(map(float, line.split())) for line in model_explanations_fh]
    assert len(gold_explanations) == len(model_explanations)
    for i in range(len(gold_explanations)):
        assert len(gold_explanations[i]) == len(model_explanations[i])
        assert len(gold_explanations[i]) > 0
    return gold_explanations, model_explanations


def validate_word_level_data(gold_explanations, model_explanations):
    valid_gold, valid_model = [], []
    for gold_expl, model_expl in zip(gold_explanations, model_explanations):
        if sum(gold_expl) == 0 or sum(gold_expl) == len(gold_expl):
            continue
        else:
            valid_gold.append(gold_expl)
            valid_model.append(model_expl)
    return valid_gold, valid_model


def compute_auc_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += roc_auc_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_ap_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += average_precision_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_rec_topk(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        idxs = np.argsort(model_explanations[i])[::-1][:sum(gold_explanations[i])]
        res += len([idx for idx in idxs if gold_explanations[i][idx] == 1])/sum(gold_explanations[i])
    return res / len(gold_explanations)


def evaluate_word_level(gold_explanations, model_explanations):
    gold_explanations, model_explanations = validate_word_level_data(gold_explanations, model_explanations)
    auc_score = compute_auc_score(gold_explanations, model_explanations)
    ap_score = compute_ap_score(gold_explanations, model_explanations)
    rec_topk = compute_rec_topk(gold_explanations, model_explanations)
    print('AUC score: {:.3f}'.format(auc_score))
    print('AP score: {:.3f}'.format(ap_score))
    print('Recall at top-K: {:.3f}'.format(rec_topk))


def evaluate_sentence_level(gold_scores, model_scores):
    corr = pearsonr(gold_scores, model_scores)[0]
    print('Pearson correlation: {:.3f}'.format(corr))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_explanations_fname', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model_explanations_fname', type=argparse.FileType('r'), required=True)
    parser.add_argument('--gold_sentence_scores_fname', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model_sentence_scores_fname', type=argparse.FileType('r'), required=True)
    args = parser.parse_args()
    gold_explanations, model_explanations = read_word_data(args.gold_explanations_fname, args.model_explanations_fname)
    gold_scores, model_scores = read_sentence_data(args.gold_sentence_scores_fname, args.model_sentence_scores_fname)
    evaluate_word_level(gold_explanations, model_explanations)
    evaluate_sentence_level(gold_scores, model_scores)


if __name__ == '__main__':
    main()
