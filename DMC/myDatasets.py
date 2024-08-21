from datasets import load_dataset, load_dataset_builder
import json
import random
import re

class myDatasets(object):
    def __init__(self, dataset_name, subject_name=None):
        self.dataset_name = dataset_name
        self.subject_name = subject_name

    def __read_bbh(self):
        data = load_dataset("lukaemon/bbh", self.subject_name)["test"]
        new_data = []

        if self.subject_name == "boolean_expressions":
            new_data = data
        elif self.subject_name == "date_understanding":
            for item in data:
                options = re.findall(r"\(([A-F])\)", item["input"])
                question = re.search(r".*?\?", item["input"]).group(0)
                answer_choice = re.search(r"\((\w)\)", item["target"]).group(1)
                answer = re.search(rf"\({answer_choice}\)\s(\d{{2}}/\d{{2}}/\d{{4}})", item["input"]).group(1)

                wrong_answers = []
                for latter in options:
                    if latter != answer_choice:
                        wrong_answers.append(re.search(rf"\({latter}\)\s(\d{{2}}/\d{{2}}/\d{{4}})", item["input"]).group(1))

                for wrong_answer in wrong_answers:
                    new_question = f"{question}\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {answer}\nOption2: {wrong_answer}"
                    new_question_inverse = f"{question}\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {wrong_answer}\nOption2: {answer}"
                    new_data.append({"input": new_question, "target": 0})
                    new_data.append({"input": new_question_inverse, "target": 1})
        elif self.subject_name == "sports_understanding":
            for item in data:
                new_question = f"{item['input']}(Please respond with yes or no.)"
                new_data.append({"input": new_question, "target": item["target"]})

        data_example = new_data[0]
        question_key = "input"
        label_key = "target"
        return new_data, data_example, question_key, label_key
    
    def __read_mmlu(self):
        data = load_dataset("cais/mmlu", self.subject_name)["test"]
        new_data = []
        for item in data:
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]

            answer = choices[answer_idx]
            
            for i in range(len(choices)):
                if i != answer_idx:
                    wrong_answer = choices[i]
                    new_question = f"{question}\n\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {answer}\nOption2: {wrong_answer}"
                    new_question_inverse = f"{question}\n\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {wrong_answer}\nOption2: {answer}"
                    new_data.append({"question": new_question, "label": 0})
                    new_data.append({"question": new_question_inverse, "label": 1})

        data_example = new_data[0]
        return new_data, data_example, "question", "label"
    
    def __read_commonsense_qa(self):
        data = load_dataset("tau/commonsense_qa")["validation"]
        new_data = []
        for item in data:
            question = item["question"]
            choices = item["choices"]["text"]
            answer_idx = ord(item["answerKey"]) - ord("A")
            answer = choices[answer_idx]
            del choices[answer_idx]
            wrong_answers = choices
            for wrong_answer in  wrong_answers:
                new_question = f"{question}\n\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {answer}\nOption2: {wrong_answer}"
                new_question_inverse = f"{question}\n\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {wrong_answer}\nOption2: {answer}"
                new_data.append({"question": new_question, "label": 0})
                new_data.append({"question": new_question_inverse, "label": 1})
        data_example = new_data[0]
        return new_data, data_example, "question", "label"
    
    def __read_AGIEval(self):
        data = []
        with open(f"./datasets/{self.dataset_name}/{self.subject_name}.jsonl") as f:
            for line in f:
                data.append(json.loads(line))
        new_data = []

        for item in data:
            question = item["question"]
            options = item["options"]
            if len(item["label"]) > 1:
                continue
            answer_idx = ord(item["label"].strip()) - ord("A")
            answer = re.sub(r"\([ABCD]\)", "", options[answer_idx]).strip()
            del options[answer_idx]
            wrong_answers = options
            for wrong_answer in  wrong_answers:
                _wrong_answer = re.sub(r"\([ABCD]\)", "", wrong_answer).strip()
                new_question = f"{question}\n\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {answer}\nOption2: {_wrong_answer}"
                new_question_inverse = f"{question}\n\nPlease choose the correct answer to the above question from the following two options.(Please output 'Option1' or 'Option2' instead of the answer itself.):\nOption1: {_wrong_answer}\nOption2: {answer}"
                new_data.append({"question": new_question, "label": 0})
                new_data.append({"question": new_question_inverse, "label": 1})

        data_example = new_data[0]
        return new_data, data_example, "question", "label"
    
    def read_dataset(self):
        dataset2read = {
            "bbh": self.__read_bbh,
            "mmlu": self.__read_mmlu,
            "commonsense_qa": self.__read_commonsense_qa,
            "AGIEval": self.__read_AGIEval
        }

        return dataset2read[self.dataset_name]()