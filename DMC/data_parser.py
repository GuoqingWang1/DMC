import json
import re

datasets = ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_boolean_expressions",
            "bbh_date_understanding", "bbh_sports_understanding", "commonsense_qa",
            "mmlu_abstract_algebra", "mmlu_business_ethics", "mmlu_global_facts",
            "mmlu_high_school_mathematics", "mmlu_professional_law", "mmlu_professional_medicine"]

confidence_extract_methods = {
    "verb_confidence": ["vanilla", "topk", "self_probing", "multi_steps", "cot"],
    "consis_confidence": ["zero_shot", "cot"],
    "consis_disturb_confidence": ["zero_shot", "cot"],
    "consis_misleading_confidence": ["zero_shot", "cot"],
    "verbis_confidence": ["vanilla", "topk"]
}

def data_parser(file_path, save_path, confidence_extract_methods, prompting, dataset):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    data_parsed = []

    Option1_match_pattern = r'Option1:\s*(.*?)\n'
    Option2_match_pattern = r'Option2:\s*(.*)'

    if confidence_extract_methods == "verb_confidence":
        if prompting in ["vanilla", "self_probing"]:
            if dataset in ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_date_understanding",
                            "commonsense_qa", "mmlu_abstract_algebra", "mmlu_business_ethics",
                            "mmlu_global_facts", "mmlu_high_school_mathematics", "mmlu_professional_law",
                            "mmlu_professional_medicine"]:
                answer_pattern = r"Option\s*[12]"
                prob_pattern = r"Probability:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}

                    if prompting == "vanilla":
                        resp = item["response"] 
                    elif prompting == "self_probing":
                        resp1 = item["response1"]
                        resp2 = item["response2"]
                        if re.fullmatch(r'-?\d+(\.\d+)?', resp2):
                            resp = resp1 + '\n' + 'Probability: ' + resp2
                        else:
                            resp = resp1 + '\n' + resp2
                    
                    question = item["question"]
                    Option1 = re.search(Option1_match_pattern, question, re.DOTALL).group(1).strip()
                    Option2 = re.search(Option2_match_pattern, question, re.DOTALL).group(1).strip()

                    label = item["label"]
                    idx = item["idx"]
                    answer_matching = re.search(answer_pattern, resp)
                    prob_matching = re.search(prob_pattern, resp)
                    if answer_matching and prob_matching:
                        answer = 0 if answer_matching.group(0).replace(" ", "") == "Option1" else 1
                        prob = float(prob_matching.group(1))
                    else:
                        if Option1 in resp:
                            resp = re.sub(re.escape(Option1), "Option1", resp, flags=re.IGNORECASE)
                        if Option2 in resp:
                            resp = re.sub(re.escape(Option2), "Option2", resp, flags=re.IGNORECASE)
                        answer_matching = re.search(answer_pattern, resp)
                        prob_matching = re.search(prob_pattern, resp)
                        if answer_matching and prob_matching:
                            answer = 0 if answer_matching.group(0).replace(" ", "") == "Option1" else 1
                            prob = float(prob_matching.group(1))
                        else:
                            answer = prob = None
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)

            elif dataset == "bbh_boolean_expressions":
                answer_pattern = r'(?:Answer:\s*(.*?)(?=\n))?([^\n]*)'
                prob_pattern = r"Probability:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in  data:
                    item_parsed = {}    

                    if prompting == "vanilla":
                        resp = item["response"] 
                    elif prompting == "self_probing":
                        resp1 = item["response1"]
                        resp2 = item["response2"]
                        if re.fullmatch(r'-?\d+(\.\d+)?', resp2):
                            resp = resp1 + '\n' + 'Probability: ' + resp2
                        else:
                            resp = resp1 + '\n' + resp2

                    idx = item["idx"]
                    answer_matching = re.search(answer_pattern, resp)
                    prob_matching = re.search(prob_pattern, resp)
                    if answer_matching and prob_matching:
                        answer = answer_matching.group(1) if answer_matching.group(1) is not None else answer_matching.group(2)
                        if answer: answer = answer.strip()

                        if answer.endswith("."):
                            answer = answer[:-1]
                        
                        if answer.startswith("(") and answer.endswith(")"):
                            answer = answer[1:-1].strip()

                        if answer.endswith("."):
                            answer = answer[:-1]

                        if answer.lower() == "true" or answer.lower() == "false": 
                            answer = True if answer.lower() == "true" else False
                        else:
                            answer = None
                        
                        prob = float(prob_matching.group(1))
                    else:
                        answer = None
                        prob = None
                    label = True if item["label"] == "True" else False
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)

            elif dataset == "bbh_sports_understanding":
                pattern = r'Answer: (\w+)\nProbability: (\d+\.\d+)'
                for item in data:
                    item_parsed = {}
                    
                    if prompting == "vanilla":
                        resp = item["response"] 
                    elif prompting == "self_probing":
                        resp1 = item["response1"]
                        resp2 = item["response2"]
                        if re.fullmatch(r'-?\d+(\.\d+)?', resp2):
                            resp = "Answer: " + resp1 + '\n' + 'Probability: ' + resp2
                        else:
                            resp = "Answer: " + resp1 + '\n' + resp2

                    label = item["label"]
                    idx = item["idx"]
                    matching = re.search(pattern, resp)
                    if matching:
                        answer = matching.group(1).lower().strip()
                        prob = float(matching.group(2))
                    else:
                        answer = None
                        prob = None
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)
        
        elif prompting == "topk":
            if dataset in ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_date_understanding",
                            "commonsense_qa", "mmlu_abstract_algebra", "mmlu_business_ethics",
                            "mmlu_global_facts", "mmlu_high_school_mathematics", "mmlu_professional_law",
                            "mmlu_professional_medicine"]:
                answer_pattern = r"Option\s*[12]"
                P1_pattern = r"P1\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                P2_pattern = r"P2\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                add_P_pattern_Option1 = r"Option1\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                add_P_pattern_Option2 = r"Option2\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    label = item["label"]
                    idx = item["idx"]

                    question = item["question"]
                    Option1 = re.search(Option1_match_pattern, question, re.DOTALL).group(1).strip()
                    Option2 = re.search(Option2_match_pattern, question, re.DOTALL).group(1).strip()

                    answers = []
                    probs = []
                    answers_matching = re.findall(answer_pattern, resp)
                    P1_matching = re.search(P1_pattern, resp)
                    P2_matching = re.search(P2_pattern, resp)
        
                    if answers_matching and (len(answers_matching) == 2 or (len(answers_matching) == 4 and answers_matching[0] == answers_matching[2] and answers_matching[1] == answers_matching[3])) and P1_matching and P2_matching:
                        A1 = 0 if answers_matching[0].replace(" ", "") == "Option1" else 1
                        A2 = 0 if answers_matching[1].replace(" ", "") == "Option1" else 1
                        P1 = float(P1_matching.group(1))
                        P2 = float(P2_matching.group(1))
                    else:
                        if Option1 in resp:
                            resp = re.sub(re.escape(Option1), "Option1", resp, flags=re.IGNORECASE)
                        if Option2 in resp:
                            resp = re.sub(re.escape(Option2), "Option2", resp, flags=re.IGNORECASE)
                        answers_matching = re.findall(answer_pattern, resp)
                        P1_matching = re.search(P1_pattern, resp)
                        P2_matching = re.search(P2_pattern, resp)
                        
                        if answers_matching and (len(answers_matching) == 2 or (len(answers_matching) == 4 and answers_matching[0] == answers_matching[2] and answers_matching[1] == answers_matching[3])) and P1_matching and P2_matching:
                            A1 = 0 if answers_matching[0].replace(" ", "") == "Option1" else 1
                            A2 = 0 if answers_matching[1].replace(" ", "") == "Option1" else 1
                            P1 = float(P1_matching.group(1))
                            P2 = float(P2_matching.group(1))
                        else:    
                            Option1_P_matching = re.search(add_P_pattern_Option1, item["response"], re.IGNORECASE)
                            Option2_P_matching = re.search(add_P_pattern_Option2, item["response"], re.IGNORECASE)
                            if Option1_P_matching and Option2_P_matching:
                                P1 = float(Option1_P_matching.group(1))
                                P2 = float(Option2_P_matching.group(1))
                                A1 = 0
                                A2 = 1
                            else:
                                A1 = A2 = P1 = P2 = None
                    answers.append(A1)
                    answers.append(A2)
                    probs.append(P1)
                    probs.append(P2)
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    item_parsed["probs"] = probs
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_boolean_expressions":
                A1_pattern = r'A1:\s*(.*?)(?=\n)([^\n]*)'
                A2_pattern = r'A2:\s*(.*?)(?=\n)([^\n]*)'
                P1_pattern = r"P1:.*?([0-9]+(?:\.[0-9]+)?)"
                P2_pattern = r"P2:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    label = item["label"]
                    idx = item["idx"]
                    answers = []
                    probs = []
                    A1_matching = re.search(A1_pattern, resp)
                    A2_matching = re.search(A2_pattern, resp)
                    P1_matching = re.search(P1_pattern, resp)
                    P2_matching = re.search(P2_pattern, resp)  
                    if A1_matching and A2_matching and P1_matching and P2_matching:
                        A1 = A1_matching.group(1)
                        if A1: A1 = A1.strip()
                        A2 = A2_matching.group(1)
                        if A2: A2 = A2.strip()

                        if A1.endswith("."):
                            A1 = A1[:-1]
                        
                        if A1.startswith("(") and A1.endswith(")"):
                            A1 = A1[1:-1].strip()

                        if A1.endswith("."):
                            A1 = A1[:-1]

                        if A2.endswith("."):
                            A2 = A2[:-1]
                        
                        if A2.startswith("(") and A2.endswith(")"):
                            A2 = A2[1:-1].strip()

                        if A2.endswith("."):
                            A2 = A2[:-1]

                        if A1.lower() == "true" or A1.lower() == "false":
                            A1 = True if A1.lower() == "true" else False
                        else:
                            A1 = None

                        if A2.lower() == "true" or A2.lower() == "false":
                            A2 = True if A2.lower() == "true" else False
                        else:
                            A2 = None    

                        P1 = float(P1_matching.group(1))
                        P2 = float(P2_matching.group(1))
                    else:
                        A1 = A2 = P1 = P2 = None
                    answers.append(A1)
                    answers.append(A2)
                    probs.append(P1)
                    probs.append(P2)
                    label = True if item["label"] == "True" else False
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    item_parsed["probs"] = probs
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_sports_understanding":
                A1_pattern = r'A1: (\w+)\n'
                A2_pattern = r'A2: (\w+)\n'
                P1_pattern = r"P1:.*?([0-9]+(?:\.[0-9]+)?)"
                P2_pattern = r"P2:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    label = item["label"]
                    idx = item["idx"]
                    answers = []
                    probs = []
                    A1_matching = re.search(A1_pattern, resp)
                    A2_matching = re.search(A2_pattern, resp)
                    P1_matching = re.search(P1_pattern, resp)
                    P2_matching = re.search(P2_pattern, resp)
                    if A1_matching and A2_matching and P1_matching and P2_matching:
                        A1 = A1_matching.group(1).lower().strip()
                        A2 = A2_matching.group(1).lower().strip()
                        P1 = float(P1_matching.group(1))
                        P2 = float(P2_matching.group(1))
                    else:
                        A1 = A2 = P1 = P2 = None
                    answers.append(A1)
                    answers.append(A2)
                    probs.append(P1)
                    probs.append(P2)
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    item_parsed["probs"] = probs
                    data_parsed.append(item_parsed)

        elif prompting == "multi_steps":
            if dataset in ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_date_understanding",
                            "commonsense_qa", "mmlu_abstract_algebra", "mmlu_business_ethics",
                            "mmlu_global_facts", "mmlu_high_school_mathematics", "mmlu_professional_law",
                            "mmlu_professional_medicine"]:
                pattern = r'Final Answer and Overall Confidence(?: \(.*?\))? ?: (Option\s*[12]), ([0-9]*\.?[0-9]+)'
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    label = item["label"]
                    idx = item["idx"]

                    question = item["question"]
                    Option1 = re.search(Option1_match_pattern, question, re.DOTALL).group(1).strip()
                    Option2 = re.search(Option2_match_pattern, question, re.DOTALL).group(1).strip()

                    matching = re.search(pattern, resp)
                    if matching:
                        answer = matching.group(1) if matching.group(1) else None
                        prob = matching.group(2) if matching.group(2) else None
                        if answer: answer = 0 if answer.replace(" ", "") == "Option1" else 1
                        if prob: prob = float(prob)
                    else:
                        if Option1 in resp:
                            resp = re.sub(re.escape(Option1), "Option1", resp, flags=re.IGNORECASE)
                        if Option2 in resp:
                            resp = re.sub(re.escape(Option2), "Option2", resp, flags=re.IGNORECASE)
                        matching = re.search(pattern, resp)
                        if matching:
                            answer = matching.group(1) if matching.group(1) else None
                            prob = matching.group(2) if matching.group(2) else None
                            if answer: answer = 0 if answer.replace(" ", "") == "Option1" else 1
                            if prob: prob = float(prob)
                        else:
                            answer = prob = None

                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_boolean_expressions":
                pattern = r'Final Answer and Overall Confidence(?: \(.*?\))? ?: (.*?), ([0-9]*\.?[0-9]+)'
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    label = item["label"]
                    idx = item["idx"]
                    matching = re.search(pattern, resp)
                    if matching:
                        answer = matching.group(1) if matching.group(1) else None
                        prob = matching.group(2) if matching.group(2) else None
                        if answer: answer = answer.strip()

                        if answer.endswith("."):
                            answer = answer[:-1]
                        
                        if answer.startswith("(") and answer.endswith(")"):
                            answer = answer[1:-1].strip()

                        if answer.endswith("."):
                            answer = answer[:-1]

                        if prob: prob = float(prob)

                        if answer.lower() == "true" or answer.lower() == "false":
                            answer = True if answer.lower() == "true" else False
                        else:
                            answer = None
                    else:
                        answer = None
                        prob = None
                    label = True if item["label"] == "True" else False
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_sports_understanding":
                pattern = r"Final Answer and Overall Confidence(?: \(.*?\))? ?: (\w+), ([0-9]*\.?[0-9]+)"
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    label = item["label"]
                    idx = item["idx"]
                    matching = re.search(pattern, resp)
                    if matching:
                        answer = matching.group(1) if matching.group(1) else None
                        prob = matching.group(2) if matching.group(2) else None
                        if answer: answer = answer.lower().strip()
                        if prob: prob = float(prob)
                    else:
                        answer = prob = None
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)    

        elif prompting == "cot":
            if dataset in ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_date_understanding",
                            "commonsense_qa", "mmlu_abstract_algebra", "mmlu_business_ethics",
                            "mmlu_global_facts", "mmlu_high_school_mathematics", "mmlu_professional_law",
                            "mmlu_professional_medicine"]:
                answer_pattern = r'Answer:\s*(Option\s*[12])'
                prob_pattern = r"Probability:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    idx = item["idx"]
                    label = item["label"]

                    question = item["question"]
                    Option1 = re.search(Option1_match_pattern, question, re.DOTALL).group(1).strip()
                    Option2 = re.search(Option2_match_pattern, question, re.DOTALL).group(1).strip()

                    answer_matching = re.search(answer_pattern, resp)
                    prob_matching = re.search(prob_pattern, resp)
                    if answer_matching and prob_matching:
                        answer = 0 if answer_matching.group(1).replace(" ", "") == "Option1" else 1
                        prob = float(prob_matching.group(1))
                    else:
                        if Option1 in resp:
                            resp = re.sub(re.escape(Option1), "Option1", resp, flags=re.IGNORECASE)
                        if Option2 in resp:
                            resp = re.sub(re.escape(Option2), "Option2", resp, flags=re.IGNORECASE)
                        answer_matching = re.search(answer_pattern, resp)
                        prob_matching = re.search(prob_pattern, resp)
                        if answer_matching and prob_matching:
                            answer = 0 if answer_matching.group(1).replace(" ", "") == "Option1" else 1
                            prob = float(prob_matching.group(1))
                        else:
                            answer = prob = None
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_boolean_expressions":
                pattern = r'Answer: (.*?)\nProbability: ([0-9.]+)'
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    idx = item["idx"]
                    label = item["label"]
                    matching = re.search(pattern, resp)
                    if matching:
                        answer = matching.group(1) if matching.group(1) is not None else None
                        prob = matching.group(2) if matching.group(2) is not None else None
                        if answer: answer = answer.strip()

                        if answer.endswith("."):
                            answer = answer[:-1]
                        
                        if answer.startswith("(") and answer.endswith(")"):
                            answer = answer[1:-1].strip()

                        if answer.endswith("."):
                            answer = answer[:-1]

                        if prob: prob = float(prob)
                        
                        if answer.lower() == "true" or answer.lower() == "false":
                            answer = True if answer.lower() == "true" else False
                        else:
                            answer = None
                    label = True if item["label"] == "True" else False
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_sports_understanding":
                pattern = r'Answer: (\w+)\s*\nProbability: (\d+\.\d+)'
                for item in data:
                    item_parsed = {}
                    resp = item["response"]
                    idx = item["idx"]
                    label = item["label"]
                    matching = re.search(pattern, resp)
                    if matching:
                        answer = matching.group(1).lower().strip()
                        prob = float(matching.group(2))
                    else:
                        answer = None
                        prob = None
                    item_parsed["idx"] = idx
                    item_parsed["answer"] = answer
                    item_parsed["label"] = label
                    item_parsed["prob"] = prob
                    data_parsed.append(item_parsed)

    elif confidence_extract_methods in ["consis_confidence", "consis_disturb_confidence", "consis_misleading_confidence"]:
        if prompting == "zero_shot":
            if dataset in ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_date_understanding",
                            "commonsense_qa", "mmlu_abstract_algebra", "mmlu_business_ethics",
                            "mmlu_global_facts", "mmlu_high_school_mathematics", "mmlu_professional_law",
                            "mmlu_professional_medicine"]:
                pattern = r"Option\s*[12]"
                for item in data:
                    item_parsed = {}
                    answers = []
                    resps = item["responses"]
                    label = item["label"]
                    idx = item["idx"]

                    question = item["question"]

                    for i in range(10):
                        resp = resps[f"sample{i+1}"]

                        Option1 = re.search(Option1_match_pattern, question, re.DOTALL).group(1).strip()
                        Option2 = re.search(Option2_match_pattern, question, re.DOTALL).group(1).strip()

                        matching = re.search(pattern, resp)
                        if matching:
                            answer = 0 if matching.group(0).replace(" ", "") == "Option1" else 1
                        else:
                            if Option1 in resp:
                                resp = re.sub(re.escape(Option1), "Option1", resp, flags=re.IGNORECASE)
                            if Option2 in resp:
                                resp = re.sub(re.escape(Option2), "Option2", resp, flags=re.IGNORECASE)
                            matching = re.search(pattern, resp)
                            if matching:
                                answer = 0 if matching.group(0).replace(" ", "") == "Option1" else 1
                            else:
                                answer = None
                        answers.append(answer)
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_boolean_expressions":
                for item in data:
                    item_parsed = {}
                    answers = []
                    resps = item["responses"]
                    label = item["label"]
                    idx = item["idx"]
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]
                        colon_index = resp.find(":")
                        if colon_index != -1:
                            resp = resp[colon_index + 1:].strip()
                        else:
                            resp = resp.strip()
                        if resp.endswith("."):
                            resp = resp[:-1]
                        
                        if resp.startswith("(") and resp.endswith(")"):
                            resp = resp[1:-1].strip()


                        if resp.endswith("."):
                            resp = resp[:-1]

                        if resp.lower() == "true" or resp.lower() == "false":
                            answer = True if resp.lower() == "true" else False
                        else:
                            answer = None
                        answers.append(answer)
                    label = True if item["label"] == "True" else False
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_sports_understanding":
                pattern = r'\b(?:Yes|No)\b'
                for item in data:
                    item_parsed = {}
                    answers = []
                    resps = item["responses"]
                    label = item["label"]
                    idx = item["idx"]
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]
                        matching = re.findall(pattern, resp, re.IGNORECASE)
                        if matching:
                            answer = matching[0].lower().strip()
                        else:
                            answer = None
                        answers.append(answer)
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)

    elif confidence_extract_methods == "verbis_confidence":
        if prompting == "vanilla":
            if dataset in ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_date_understanding",
                            "commonsense_qa", "mmlu_abstract_algebra", "mmlu_business_ethics",
                            "mmlu_global_facts", "mmlu_high_school_mathematics", "mmlu_professional_law",
                            "mmlu_professional_medicine"]:
                answer_pattern = r"Option\s*[12]"
                prob_pattern = r"Probability:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}
                    resps = item["responses"]
                    idx = item["idx"]
                    label = item["label"]
                    question = item["question"]
                    answers = []
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]
                        Option1 = re.search(Option1_match_pattern, question, re.DOTALL).group(1).strip()
                        Option2 = re.search(Option2_match_pattern, question, re.DOTALL).group(1).strip()

                        answer_matching = re.search(answer_pattern, resp)
                        prob_matching = re.search(prob_pattern, resp)
                        if answer_matching and prob_matching:
                            answer = 0 if answer_matching.group(0).replace(" ", "") == "Option1" else 1
                            prob = float(prob_matching.group(1))
                        else:
                            if Option1 in resp:
                                resp = re.sub(re.escape(Option1), "Option1", resp, flags=re.IGNORECASE)
                            if Option2 in resp:
                                resp = re.sub(re.escape(Option2), "Option2", resp, flags=re.IGNORECASE)
                            answer_matching = re.search(answer_pattern, resp)
                            prob_matching = re.search(prob_pattern, resp)
                            if answer_matching and prob_matching:
                                answer = 0 if answer_matching.group(0).replace(" ", "") == "Option1" else 1
                                prob = float(prob_matching.group(1))
                            else:
                                answer = None
                                prob = None
                        answers.append([answer, prob])
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_boolean_expressions":
                answer_pattern = r'(?:Answer:\s*(.*?)(?=\n))?([^\n]*)'
                prob_pattern = r"Probability:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in  data:
                    item_parsed = {}
                    resps = item["responses"]
                    idx = item["idx"]
                    label = item["label"]
                    answers = []
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]
                        answer_matching = re.search(answer_pattern, resp)
                        prob_matching = re.search(prob_pattern, resp)
                        if answer_matching and prob_matching:
                            answer = answer_matching.group(1) if answer_matching.group(1) is not None else answer_matching.group(2)
                            if answer: answer = answer.strip()

                            if answer.endswith("."):
                                answer = answer[:-1]
                            
                            if answer.startswith("(") and answer.endswith(")"):
                                answer = answer[1:-1].strip()

                            if answer.endswith("."):
                                answer = answer[:-1]

                            if answer.lower() == "true" or answer.lower() == "false": 
                                answer = True if answer.lower() == "true" else False
                            else:
                                answer = None
                        
                            prob = float(prob_matching.group(1))
                        else:
                            answer = None
                            prob = None
                        answers.append([answer, prob])
                    label = True if item["label"] == "True" else False
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_sports_understanding":
                pattern = r'Answer: (\w+)\s*\nProbability: (\d+\.\d+)'
                for item in data:
                    item_parsed = {}
                    resps = item["responses"]
                    idx = item["idx"]
                    label = item["label"]
                    answers = []
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]
                        matching = re.search(pattern, resp)
                        if matching:
                            answer = matching.group(1).lower().strip()
                            prob = float(matching.group(2))
                        else:
                            answer = None
                            prob = None
                        answers.append([answer, prob])
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)

        elif prompting == "topk":
            if dataset in ["AGIEval_gaokao-mathqa", "AGIEval_sat-math", "bbh_date_understanding",
                            "commonsense_qa", "mmlu_abstract_algebra", "mmlu_business_ethics",
                            "mmlu_global_facts", "mmlu_high_school_mathematics", "mmlu_professional_law",
                            "mmlu_professional_medicine"]:
                answer_pattern = r"Option\s*[12]"
                P1_pattern = r"P1\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                P2_pattern = r"P2\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                add_P_pattern_Option1 = r"Option1\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                add_P_pattern_Option2 = r"Option2\s*:?.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}
                    resps = item["responses"]
                    idx = item["idx"]
                    label = item["label"]
                    question = item["question"]
                    answers = []
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]

                        Option1 = re.search(Option1_match_pattern, question, re.DOTALL).group(1).strip()
                        Option2 = re.search(Option2_match_pattern, question, re.DOTALL).group(1).strip()

                        answers_matching = re.findall(answer_pattern, resp)
                        P1_matching = re.search(P1_pattern, resp)
                        P2_matching = re.search(P2_pattern, resp)
                        
                        if answers_matching and (len(answers_matching) == 2 or (len(answers_matching) == 4 and answers_matching[0] == answers_matching[2] and answers_matching[1] == answers_matching[3])) and P1_matching and P2_matching:
                            A1 = 0 if answers_matching[0].replace(" ", "") == "Option1" else 1
                            A2 = 0 if answers_matching[1].replace(" ", "") == "Option1" else 1
                            P1 = float(P1_matching.group(1))
                            P2 = float(P2_matching.group(1))
                        else:
                            if Option1 in resp:
                                resp = re.sub(re.escape(Option1), "Option1", resp, flags=re.IGNORECASE)
                            if Option2 in resp:
                                resp = re.sub(re.escape(Option2), "Option2", resp, flags=re.IGNORECASE)
                            answers_matching = re.findall(answer_pattern, resp)
                            P1_matching = re.search(P1_pattern, resp)
                            P2_matching = re.search(P2_pattern, resp)
                            
                            if answers_matching and (len(answers_matching) == 2 or (len(answers_matching) == 4 and answers_matching[0] == answers_matching[2] and answers_matching[1] == answers_matching[3])) and P1_matching and P2_matching:
                                A1 = 0 if answers_matching[0].replace(" ", "") == "Option1" else 1
                                A2 = 0 if answers_matching[1].replace(" ", "") == "Option1" else 1
                                P1 = float(P1_matching.group(1))
                                P2 = float(P2_matching.group(1))
                            else:
                                Option1_P_matching = re.search(add_P_pattern_Option1, resps[f"sample{i+1}"], re.IGNORECASE)
                                Option2_P_matching = re.search(add_P_pattern_Option2, resps[f"sample{i+1}"], re.IGNORECASE)
                                if Option1_P_matching and Option2_P_matching:
                                    P1 = float(Option1_P_matching.group(1))
                                    P2 = float(Option2_P_matching.group(1))
                                    A1 = 0
                                    A2 = 1
                                else:
                                    A1 = A2 = P1 = P2 = None
                        answers.append([[A1, A2], [P1, P2]])
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_boolean_expressions":
                A1_pattern = r'A1:\s*(.*?)(?=\n)([^\n]*)'
                A2_pattern = r'A2:\s*(.*?)(?=\n)([^\n]*)'
                P1_pattern = r"P1:.*?([0-9]+(?:\.[0-9]+)?)"
                P2_pattern = r"P2:.*?([0-9]+(?:\.[0-9]+)?)"
                for item in data:
                    item_parsed = {}
                    resps = item["responses"]
                    idx = item["idx"]
                    label = item["label"]
                    answers = []
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]
                        A1_matching = re.search(A1_pattern, resp)
                        A2_matching = re.search(A2_pattern, resp)
                        P1_matching = re.search(P1_pattern, resp)
                        P2_matching = re.search(P2_pattern, resp)  
                        if (A1_matching and P1_matching) or (A2_matching and P2_matching):
                            if A1_matching and P1_matching:
                                A1 = A1_matching.group(1)
                                if A1: A1 = A1.strip()

                                if A1.endswith("."):
                                    A1 = A1[:-1]
                                
                                if A1.startswith("(") and A1.endswith(")"):
                                    A1 = A1[1:-1].strip()

                                if A1.endswith("."):
                                    A1 = A1[:-1]
                                
                                if A1.lower() == "true" or A1.lower() == "false":
                                    A1 = True if A1.lower() == "true" else False
                                else:
                                    A1 = None
                                P1 = float(P1_matching.group(1))

                            
                            if A2_matching and P2_matching:
                                A2 = A2_matching.group(1)
                                if A2: A2 = A2.strip()

                                if A2.endswith("."):
                                    A2 = A2[:-1]
                                
                                if A2.startswith("(") and A2.endswith(")"):
                                    A2 = A2[1:-1].strip()

                                if A2.endswith("."):
                                    A2 = A2[:-1]

                                if A2.lower() == "true" or A2.lower() == "false":
                                    A2 = True if A2.lower() == "true" else False
                                else:
                                    A2 = None    
                                
                                P2 = float(P2_matching.group(1))
                        else:
                            A1 = A2 = P1 = P2 = None
                        answers.append([[A1, A2], [P1, P2]])
                    label = True if item["label"] == "True" else False
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)
            elif dataset == "bbh_sports_understanding":
                A1_pattern = r'A1: (\w+)\s*\n'
                A2_pattern = r'A2: (\w+)\s*\n'
                P1_pattern = r"P1:.*?([0-9]+(?:\.[0-9]+)?)"
                P2_pattern = r"P2:.*?([0-9]+(?:\.[0-9]+)?)"
        
                for item in data:
                    item_parsed = {}
                    resps = item["responses"]
                    idx = item["idx"]
                    label = item["label"]
                    answers = []
                    for i in range(10):
                        resp = resps[f"sample{i+1}"]
                        A1_matching = re.search(A1_pattern, resp)
                        A2_matching = re.search(A2_pattern, resp)
                        P1_matching = re.search(P1_pattern, resp)
                        P2_matching = re.search(P2_pattern, resp)
                        if (A1_matching  and P1_matching) or (A2_matching and P2_matching):
                            if A1_matching: 
                                A1 = A1_matching.group(1).lower().strip()
                            else:
                                A1 = None
                            if A2_matching: 
                                A2 = A2_matching.group(1).lower().strip()
                            else: 
                                A2 = None
                            if P1_matching: 
                                P1 = float(P1_matching.group(1))
                            else:
                                P1 = None
                            if P2_matching: 
                                P2 = float(P2_matching.group(1))
                            else:
                                P2 = None
                        else:
                            A1 = A2 = P1 = P2 = None
                        answers.append([[A1, A2], [P1, P2]])
                    item_parsed["idx"] = idx
                    item_parsed["answers"] = answers
                    item_parsed["label"] = label
                    data_parsed.append(item_parsed)

    with open(save_path, 'w') as f:
        for item in data_parsed:
            json_line = json.dumps(item)
            f.write(json_line+'\n')




