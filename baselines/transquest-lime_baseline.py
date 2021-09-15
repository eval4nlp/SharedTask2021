import argparse
import os

import numpy as np
import torch

from lime.lime_text import LimeTextExplainer


from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel


def format_explanations(explanations):
    explanations = explanations.as_map()[1]
    ordered_explanations = np.zeros(len(explanations))
    for idx, v in explanations:
        ordered_explanations[idx] = v * -1.
    return ordered_explanations


def explain_instance_target(model, explainer, text_a, text_b):
    def predict_proba(texts):
        text_src = [text_a] * len(texts)
        to_predict = list(zip(text_src, texts))
        to_predict = list(map(list, to_predict))
        preds, _ = model.predict(to_predict)
        return np.vstack((preds, preds)).T

    prediction, raw_outputs = model.predict([[text_a, text_b]])
    explanations = explainer.explain_instance(text_b, predict_proba, num_features=len(text_b.split()), labels=(1,))
    explanations = format_explanations(explanations)
    return prediction, explanations


def explain_instance_source(model, explainer, text_a, text_b):
    def predict_proba(texts):
        text_tgt = [text_b] * len(texts)
        to_predict = list(zip(texts, text_tgt))
        to_predict = list(map(list, to_predict))
        preds, _ = model.predict(to_predict)
        return np.vstack((preds, preds)).T

    prediction, raw_outputs = model.predict([[text_a, text_b]])
    explanations = explainer.explain_instance(text_a, predict_proba, num_features=len(text_a.split()), labels=(1,))
    explanations = format_explanations(explanations)
    return prediction, explanations


def explain_testset(src_path, tgt_path, pretrained_model_name, task_model_name, output_pref, explain_source=False):
    src_lines = [s.strip() for s in open(src_path).readlines()]
    tgt_lines = [s.strip() for s in open(tgt_path).readlines()]
    assert len(src_lines) == len(tgt_lines)
    model = MonoTransQuestModel(pretrained_model_name, task_model_name, num_labels=1, use_cuda=torch.cuda.is_available())
    explainer = LimeTextExplainer(class_names=['score', 'score'], bow=False, split_expression=' ')
    explain_fn = explain_instance_source if explain_source else explain_instance_target

    output_predictions = open(f"{output_pref}.predictions", "w")
    output_explanations = open(f"{output_pref}.explanations", "w")

    for src, tgt in zip(src_lines, tgt_lines):
        prediction, explanations = explain_fn(model, explainer, src, tgt)
        output_predictions.write(f"{prediction}\n")
        output_explanations.write(" ".join(map(str, explanations)) + "\n")
    output_explanations.close()
    output_predictions.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--tgt_path", type=str, required=True)
    parser.add_argument("--output_pref", required=True)
    parser.add_argument("--pretrained_model_name", required=True)
    parser.add_argument("--task_model_name", required=True)
    parser.add_argument("--explain_source", action="store_true", required=False)
    parser.add_argument("--cache_path", default=None, required=False)
    args = parser.parse_args()

    if args.cache_path is not None:
        os.environ["TRANSFORMERS_CACHE"] = args.cache_path

    explain_testset(
        args.src_path, args.tgt_path, args.pretrained_model_name, args.task_model_name, args.output_pref,
        explain_source=args.explain_source
    )


if __name__ == "__main__":
    main()