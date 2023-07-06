import os
import argparse
import json
import re
import string
import sys
import random 
from typing import List
from collections import Counter, OrderedDict, defaultdict
from datasets import load_dataset, list_metrics, load_metric
# from nlgeval import compute_metrics, compute_individual_metrics
import language_evaluation
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk import word_tokenize
# from bert_score import BERTScorer

re_art = re.compile(r'\b(a|an|the)\b')	
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')	

def normalize_answer(s):	
    """	
    Lower text and remove punctuation, articles and extra whitespace.	
    """	
    s = s.lower()	
    s = re_punc.sub(' ', s)	
    s = re_art.sub(' ', s)	
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '	
    s = ' '.join(s.split())	
    return s.split(' ')

class DialogEvaluator:
    def __init__(self, metric_name=None, tokenizer=None, eval_selection=False, eval_gen=True):
        metric_names = metric_name.split("&")
        metric_fns = {"gen": [], "cls": []}
        for name in metric_names:
            if name.strip() == "f1":
                metric_fn = self.get_unigram_F1
                keys = ["f1"]
                metric_type = "gen"
            elif name.strip() == "bleu":
                metric_fn = self.compute_corpus_bleu
                keys = ["bleu1", "bleu2", "bleu3", "bleu4"]
                metric_type = "gen"
            elif name.strip() == "rouge":
                # self.rouge_evaluator = load_metric('rouge')
                self.rouge_evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=1, tokenization_fn=normalize_answer)
                metric_fn = self.compute_rouge
                keys = ["rouge1", "rouge2"]
                metric_type = "gen"
            elif name.strip() == "dist":
                metric_fn = self.calc_corpus_distinct
                keys = ["distinct-1", "distinct-2"]
                metric_type = "gen"
            elif name.strip() == "bert":
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                metric_fn =self.compute_bert_score
                keys = ["bert-f1"]
                metric_type = "gen"
            elif name.strip() == "ppl":
                continue
            elif name.strip() == "acc":
                # for evaluating knowledge selection task
                self.accuracy_metric = load_metric("accuracy")
                metric_fn = self.compute_accuracy
                keys = ["accurracy"]
                metric_type = "cls"
            elif name.strip() == "recall":
                # for evaluating knowledge selection task
                metric_fn = self.compute_recall
                keys = ["recall@1", "recall@2", "recall@5", "recall@10"]
                metric_type = "cls"
            else:
                raise NotImplementedError
            metric_fns[metric_type].append((metric_fn, keys))

        self.metric_fns = metric_fns
        self.tokenizer = tokenizer
        self.eval_selection = eval_selection
        self.eval_gen = eval_gen

    def __call__(self, eval_predictions):
        predictions, labels = eval_predictions
        cls_predictions, cls_labels = None, None
        if isinstance(predictions, tuple):
            if self.eval_selection:
                cls_predictions = predictions[1]
                cls_labels = labels[1]

            predictions = predictions[0]
            labels = labels[0] if isinstance(labels, tuple) else labels
        
        results = self.compute(predictions, labels, post_proc=True) if self.eval_gen else {}
        if self.eval_selection:
            assert cls_predictions is not None
            cls_results = self.compute_cls(cls_predictions, cls_labels)
            results.update(cls_results)

        return results

    def compute(self, pred, gold, post_proc=False, prefix=""):
        if post_proc:
            # decode both pred and gold if ther are tensors
            if torch.is_tensor(pred):
                assert self.tokenizer is not None
                pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)

            if torch.is_tensor(gold):
                assert self.tokenizer is not None
                gold = np.where(gold != -100, gold, self.tokenizer.pad_token_id)
                gold = self.tokenizer.batch_decode(gold, skip_special_tokens=True)

            # double check to remove \n
            pred = [p.strip() for p in pred]
            gold = [g.strip() for g in gold]

        results = {}
        for metric_fn, keys in self.metric_fns["gen"]:
            metric_results = metric_fn(pred, gold)
            for k, v in zip(keys, metric_results):
                results[prefix+k] = round(v * 100, 2)
        return results
    
    def compute_cls(self, pred, gold, prefix=""):
        results = {}
        for metric_fn, keys in self.metric_fns["cls"]:
            metric_results = metric_fn(pred, gold)
            
            for k, v in zip(keys, metric_results):
                results[prefix+k] = round(v * 100, 2)
        return results
    
    def _preproc_preds_golds(self, pred, gold=None):
        cands = []
        golds = []
        help_tokenize = lambda x: word_tokenize(x.lower())
        for idx, p in enumerate(pred):
            cands.append(help_tokenize(p.lower()))
            if gold is not None:
                golds.append(help_tokenize(gold[idx].lower()))
        return cands, golds

    def _get_ngrams(self, text, n):
        """
        Returns all ngrams that are in the text.
        Note: this function does NOT lowercase text. If you want to lowercase, you should
        do so before calling this function.
        Inputs:
        text: string, space-separated
        n: int
        Returns:
        list of strings (each is a ngram, space-separated)
        """
        tokens = text.split()
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]  # list of str

    """
    Compute unigram-F1 score
    """

    def _prec_recall_f1_score(self, pred_items, gold_items):
        """
        PARLAI
        Computes precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def get_unigram_F1(self, pred, gold):
        f1, precision, recall = [], [], []
        for p,g in zip(pred,gold):
            p = normalize_answer(p)
            g = normalize_answer(g)
            f1_i, precision_i, recall_i = self._prec_recall_f1_score(p, g)

            f1.append(f1_i)
            precision.append(precision_i)
            recall.append(recall_i)
        return np.mean(f1), np.mean(precision), np.mean(recall)

    def compute_EM(self, pred, gold):
        EM = []
        for p,g in zip(pred,gold):
            p = normalize_answer(p)
            g = normalize_answer(g)
            EM.append(1 if p == g else 0)
        return np.mean(EM)

    def compute_corpus_bleu(self, pred, gold):
        # hypothesis = [normalize_answer(hyp) for hyp in hypothesis]
        # references = [[normalize_answer(ref)] for ref in references]
        hypothesis, references = self._preproc_preds_golds(pred, gold)

        references = [[ref] for ref in references]
        sf = SmoothingFunction(epsilon=1e-12).method1
        b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=sf)
        b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=sf)
        b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=sf)
        b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=sf)
        return b1, b2, b3, b4


    def compute_rouge(self, pred, gold):
        pred, gold = self._preproc_preds_golds(pred, gold)
        predictions = [' '.join(c) for c in pred]
        answers = [' '.join(g) for g in gold]
        scores = self.rouge_evaluator.run_evaluation(predictions, answers)
        return scores["rouge1"], scores["rouge2"]


    def _calc_ngram_dict(self, tokens:List[str], ngram:int, dict_ref=None):
        ngram_dict = defaultdict(int) if dict_ref is None else dict_ref
        total = len(tokens)
        for i in range(0, total - ngram + 1):
            item = tuple(tokens[i:i + ngram])
            ngram_dict[item] += 1
        return ngram_dict

    def _calc_distinct_ngram(self, cands, ngram):
        ngram_total = 0.00001
        ngram_distinct_count = 0.00001
        pred_dict = defaultdict(int)
        for cand_tokens in cands:
            self._calc_ngram_dict(cand_tokens, ngram, pred_dict)
        for key, freq in pred_dict.items():
            ngram_total += freq
            ngram_distinct_count += 1
        return ngram_distinct_count / ngram_total

    def _calc_sent_distinct_ngram(self, cand, ngram):
        ngram_total = 0.0000000001
        ngram_distinct_count = 0.0
        ngram_dict = defaultdict(int)
        for i in range(0, len(cand) - ngram + 1):
            item = tuple(cand[i:i + ngram])
            ngram_dict[item] += 1
        for _, freq in ngram_dict.items():
            ngram_total += freq
            ngram_distinct_count += 1
        return ngram_distinct_count / ngram_total

    def calc_corpus_distinct(self, cands, golds=None):
        cands, _ = self._preproc_preds_golds(cands)
        distinct1 = self._calc_distinct_ngram(cands, 1)
        distinct2 = self._calc_distinct_ngram(cands, 2)
        return distinct1, distinct2

    def compute_bert_score(self, cands, golds):
        P, R, F1 = self.bert_scorer.score(cands, golds)
        F1 = np.mean(F1.tolist())
        return (F1,)
    
    """Classification evaluation metrics from here"""

    def compute_accuracy(self, preds, golds):
        preds = np.argmax(preds, axis=1)
        results = self.accuracy_metric.compute(references=golds, predictions=preds)
        return (results["accuracy"],)
    
    def compute_recall(self, preds, golds):
        orders = np.argsort(preds).tolist()
        r1, r2, r5, r10 = 0, 0, 0, 0
        for order, gold in zip(orders, golds):
            for i, rank in enumerate(order[::-1]):
                if i >= 10: break 
                if gold == rank:
                    if i < 1:  r1 += 1
                    if i < 2:  r2 += 1
                    if i < 5:  r5 += 1
                    if i < 10: r10 += 1
        N = len(orders)
        return r1/N, r2/N, r5/N, r10/N


if __name__ == "__main__":
    evaluator = DialogEvaluator(metric_name="bleu&rouge&f1&dist")

    # pred_path = "/home/xuyan/dialog-kn/EbmDial/save/langevin_s_5_ss_01_bart-base_gp_1_e15_top5_pad/epoch_1/test_unseen_generations.txt"
    # gold_path = "/home/xuyan/dialog-kn/EbmDial/save/langevin_s_5_ss_01_bart-base_gp_1_e15_top5_pad/epoch_1/test_unseen_golds.txt"

    # with open(pred_path, "r") as f:
    #     pred = f.readlines()
    # with open(gold_path, "r") as f:
    #     gold = f.readlines()
    
    # results = evaluator.compute(pred, gold, post_proc=True)
    # print(results)

    
    # KnowledGPT 
    eval_path = "/home/xuyan/dialog-kn/knowledgpt/wizard_of_wikipedia/log/test_oracle/test_unseen-decoded-iter-0.txt"

    with open(eval_path, "r") as f:
        lines = f.readlines()

    pred = []
    gold = []
    for line in lines:
        p = line.split("|||")[0]
        g = line.split("|||")[1]

        pred.append(p)
        gold.append(g)

    results = evaluator.compute(pred, gold, post_proc=True)
    print("Number of predictions:", len(pred))
    print(results)


    # # BridgeKS
    # eval_path = "/home/xuyan/dialog-kn/BridgeKS/my.example.PIPM.seen"

    # with open(eval_path, "r") as f:
    #     lines = f.readlines()

    # pred = []
    # gold = []
    # for line in lines:
    #     line = json.loads(line)
    #     p = line["rsp_pd"]
    #     g = line["rsp_gt"]

    #     pred.append(p)
    #     gold.append(g)

    # results = evaluator.compute(pred, gold, post_proc=True)
    # print(results)


    # # MIKe
    # eval_path = "/home/ziwei/EbmDial/MIKe/MIKe_WoW/test_unseen_generated.txt"
    # gold_path = "/home/ziwei/EbmDial/MIKe/MIKe_WoW/test_unseen_ans.txt"

    # with open(eval_path, "r") as f:
    #     plines = f.readlines()
    # with open(gold_path, "r") as f:
    #     glines = f.readlines()
    
    # kns, pred, gold = [], [], []
    # for pline, gline in zip(plines, glines):
    #     kn, p = pline.strip().split("\t")
    #     g = gline.strip()

    #     kns.append(kn)
    #     pred.append(p)
    #     gold.append(g)
    
    # results = evaluator.compute(pred, gold, post_proc=True)
    # print(results)


    # # KAT
    # eval_path = "/home/xuyan/dialog-kn/KAT-TSLF/results/generations/predictions_wizard_kat_b1_test_unseen_.txt"
    # gold_path = "/home/xuyan/dialog-kn/EbmDial/save/langevin_s_5_ss_01_bart-base_gp_1_e15_top5_pad/epoch_1/test_unseen_golds.txt"

    # with open(eval_path, "r") as f:
    #     plines = f.readlines()
    # with open(gold_path, "r") as f:
    #     glines = f.readlines()
    
    # kns, pred, gold = [], [], []
    # for pline, gline in zip(plines, glines):
    #     p = pline.strip()
    #     g = gline.strip()

    #     # kns.append(kn)
    #     pred.append(p)
    #     gold.append(g)
    
    # results = evaluator.compute(pred, gold, post_proc=True)
    # print(results)
