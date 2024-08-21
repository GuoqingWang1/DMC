import os
import json
from tqdm import tqdm
from Prompt import myPrompt
from LLM import LLM
from myDatasets import myDatasets
import re
from discretize import Discretizer
from meta_d_prime import Meta_d_prime
# from plotter import Plotter
import time
import datetime
import numpy as np
from interval import Interval
from collections import Counter
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss

def comp_meta_d(file_path, confidence_extract_methods, prompting, beta, p, num_bins=4, discre_type="equal_width", padCells=1):
    # reading parsed data
    answers  = []
    labels = []
    probs = []
    with open(file_path, 'r')  as f:
        for line in f:
            line = json.loads(line)
            if confidence_extract_methods == "verb_confidence":
                if prompting in ["vanilla", "self_probing", "multi_steps", "cot"]:
                    if line["answer"] is not None and line["prob"] is not None:
                        if line["prob"] in Interval(0.0, 1.0):
                            assert line["prob"] in Interval(0.0, 1.0), f"Incorrect prob, file_path: {file_path}, prob={line['prob']}"
                            answers.append(line["answer"])
                            labels.append(line["label"])
                            probs.append(line["prob"])
                elif prompting == "topk":
                    if all(x is not None for x in line["answers"]) and all(x is not None for x in line["probs"]):
                        incorrect_prob = False
                        for prob in line["probs"]:
                            if (prob in Interval(0.0, 1.0)) == False:
                                incorrect_prob = True
                        if incorrect_prob == False:
                            ans_list = line["answers"]
                            prob_list = line["probs"]
                            if prob_list[0] or prob_list[1]:
                                assert (prob_list[0] in Interval(0.0, 1.0)) and (prob_list[1] in Interval(0.0, 1.0)), f"Incorrect prob, file_path: {file_path}, prob={prob_list}"
                                prob_list = [-1.0 if x is None else x for x in prob_list]
                                max_prob_idx = prob_list.index(max(prob_list))
                                answers.append(ans_list[max_prob_idx])
                                labels.append(line["label"])
                                probs.append(prob_list[max_prob_idx])
            elif confidence_extract_methods in ["consis_confidence", "consis_disturb_confidence", "consis_misleading_confidence"]:
                if prompting == "zero_shot":
                    if all(x is not None for x in line["answers"]):
                        counter = Counter(line["answers"])
                        total = len(line["answers"])
                        freqs = {key: value / total for key, value in counter.items()}
                        most_freq_ans = max(freqs, key=freqs.get)
                        most_freq_score = freqs[most_freq_ans]
                        answers.append(most_freq_ans)
                        probs.append(most_freq_score)
                        labels.append(line["label"])
            elif confidence_extract_methods == "verbis_confidence":
                if prompting == "vanilla":
                    is_nonsense = False
                    incorrect_prob = False
                    for item in line["answers"]:
                        if all(x is not None for x in item) == False:
                            is_nonsense = True
                            break
                        if item[1] is not None and (item[1] in Interval(0.0, 1.0)) == False:
                            incorrect_prob = True
                            break

                    if is_nonsense == False and incorrect_prob == False:
                        unique_answers = list(set([x[0] for x in line["answers"]]))
                        conf_score = []
                        for unique_answer in unique_answers:
                            num = 0
                            conf_sum = 0
                            for answer in line["answers"]:
                                if unique_answer == answer[0]:
                                    num = num + 1
                                    conf_sum = conf_sum + answer[1]
                                    assert answer[1] in Interval(0.0, 1.0), f"Incorrect prob, file_path: {file_path}, prob={answer[1]}"
                            
                            conf_score.append(conf_sum / num)
                        assert len(unique_answers) == len(conf_score), f"compute error: verbis_confidence vanilla"
                        if len(unique_answers) == 1:
                            answers.append(unique_answers[0])
                            probs.append(conf_score[0])
                            labels.append(line["label"])
                        else:
                            max_prob_idx = conf_score.index(max(conf_score))
                            answers.append(unique_answers[max_prob_idx])
                            probs.append(conf_score[max_prob_idx])
                            labels.append(line["label"])
                elif prompting == "topk":
                    is_nonsense = False
                    incorrect_prob = False
                    for Item in line["answers"]:
                        for item in Item:
                            if all(x is not None for x in item) == False:
                                is_nonsense = True
                                break
                        for p in Item[1]:
                            if p is not None and (p in Interval(0.0, 1.0)) == False:
                                incorrect_prob = True
                                break

                    if is_nonsense == False and incorrect_prob == False:
                        top1_answers = []
                        for answer in line["answers"]:
                            max_prob_idx = answer[1].index(max(answer[1]))
                            top1_answers.append([answer[0][max_prob_idx], answer[1][max_prob_idx]])
                        unique_answers = list(set([x[0] for x in top1_answers]))
                        conf_score = []
                        for unique_answer in unique_answers:
                            num = 0
                            conf_sum = 0
                            for answer in top1_answers:
                                if unique_answer == answer[0]:
                                    num = num + 1
                                    conf_sum = conf_sum + answer[1]
                                    assert answer[1] in Interval(0.0, 1.0), f"Incorrect prob, file_path: {file_path}, prob={answer[1]}"
                            
                            conf_score.append(conf_sum / num)
                        assert len(unique_answers) == len(conf_score), f"compute error: verbis_confidence vanilla"
                        if len(unique_answers) == 1:
                            answers.append(unique_answers[0])
                            probs.append(conf_score[0])
                            labels.append(line["label"])
                        else:
                            max_prob_idx = conf_score.index(max(conf_score))
                            answers.append(unique_answers[max_prob_idx])
                            probs.append(conf_score[max_prob_idx])
                            labels.append(line["label"])

    # discretize prob
    discretizer = Discretizer()
    discre_probs = discretizer.apply(num_bins=num_bins, data=probs, discre_type=discre_type)

    # compute accuracy rate
    accuracy_rate = np.sum(np.array(answers) == np.array(labels)) / len(answers)

    # compute meta-d-prime
    meta_d_prime_computer = Meta_d_prime()
    nR_S1, nR_S2 = meta_d_prime_computer.trials2counts(stimID=labels, response=answers, rating=discre_probs, nRatings=num_bins, padCells=padCells)
    print(nR_S1)
    print(nR_S2)
    fit = meta_d_prime_computer.fit_meta_d_MLE(nR_S1=nR_S1, nR_S2=nR_S2, beta=beta, p=p)
    print(file_path)
    return fit, accuracy_rate, nR_S1, nR_S2, answers, labels, discre_probs

if __name__ == "__main__":

    datasets = ["AGIEval_sat-math", "bbh_boolean_expressions",
            "bbh_date_understanding", "commonsense_qa",
            "mmlu_business_ethics", "mmlu_global_facts",
            "mmlu_high_school_mathematics", "mmlu_professional_medicine"]

    confidence_extract_methods = {
        "verb_confidence": ["vanilla", "topk", "cot"],
        "consis_confidence": ["zero_shot"],
        "consis_disturb_confidence": ["zero_shot"],
        "consis_misleading_confidence": ["zero_shot"]
        #"verbis_confidence": ["vanilla", "topk"]
    }      