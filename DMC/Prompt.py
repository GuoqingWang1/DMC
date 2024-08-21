

class myPrompt(object):
    def __init__(self):
        self.base_templates = {"zero_shot": "Provide your best answer for the following question. Give ONLY the answer, no other words or explanation.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                               "cot": "Provide your best answer for the following question, analyze step by step. Give the thought process and the answer.\n\nFor example:\n\nExplanation:<one sentence step-by-step explanation of your thought process>\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}"}

        
        self.verb_templates_1S = {"topk": "Provide your {k} best answers and the probability that each is correct (0.0 to 1.0) for the following question. Give ONLY the answers and probabilities, no other words or explanation.\n\n For example:\n\nA1: <first most likely answer, as short as possible; not a complete sentence, just the answer!>\nP1: <the probability between 0.0 and 1.0 that A1 is correct, without any extra commentary whatsoever; just the probability!>\n...\n A{k}: <{k}-th most likely answer, as short as possible; not a complete sentence, just the answer!>\nP{k}: <the probability between 0.0 and 1.0 that A{k} is correct, without any extra commentary whatsoever; just the probability!> \n\nThe question is: {THE_QUESTION}",
                                  "vanilla": "Provide your best answer and the probability that it is correct (0.0 to 1.0) for the following question. Give ONLY the answer and probability, no other words or explanation.\n\n For example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\n Probability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: ${THE_QUESTION}",
                                  "cot": "Provide your best answer and the probability that it is correct (0.0 to 1.0) for the following question, analyze step by step. Give the thought process, the answer and the probability. For example:\n\nExplanation:<one sentence step-by-step explanation of your thought process>\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\n Probability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>\n\nThe question is: ${THE_QUESTION}",
                                  "multi_steps": "Read the question, break down the problem into K steps, think step by step, give your confidence in each step, and then derive your final answer and your confidence. Note: The confidence indicates how likely you think your answer is correct.\nUse the following format to answer:\n\nStep 1: [first reasoning step], Confidence: [the probability between 0.0 and 1.0 that Step 1 is correct, without any extra commentary whatsoever; just the probability!]\n...\nStep K: [Kth reasoning step], Confidence: [the probability between 0.0 and 1.0 that Step K is correct, without any extra commentary whatsoever; just the probability!]\nFinal Answer and Overall Confidence (0.0 to 1.0): [most likely Final Answer, as short as possible; not a complete sentence, just the answer!], [the probability between 0.0 and 1.0 that Final Answer is correct, without any extra commentary whatsoever; just the probability!]\n\nThe question is: ${THE_QUESTION}"}
        
        self.verb_templates_2S = {"topk": ["Provide your {k} best answers for the following question. Give ONLY the answers, no other words or explanation.\n\n For example:\n\nA1: <first most likely answer, as short as possible; not a complete sentence, just the answer!>\n...\nA{k}: <{k}-th most likely answer, as short as possible; not a complete sentence, just the answer!>\n\nThe question is: {THE_QUESTION}",
                                  "Provide the probability that each of your answers is correct. Give ONLY the probabilities, no other words or explanation.\n\nFor example:\n\nP1: <the probability between 0.0 and 1.0 that A1 is correct, without any extra commentary whatsoever; just the probability!>\n...\nP{k}: <the probability between 0.0 and 1.0 that A{k} is correct, without any extra commentary whatsoever; just the probability!>"],

                                  "vanilla": ["Provide your best answer for the following question, no other words or explanation. For example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\n\nThe question is: ${THE_QUESTION}",
                                  "Provide the probability that your answers is correct. Give ONLY the probabilities, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that Answer is correct, without any extra commentary whatsoever; just the probability!>"],

                                  "cot": ["Provide your best answer for the following question. Before giving your answer, provide a step-by-step explanation of your thought process. Then on a new line give the answer with no other words or explanation.\n\nFor example:\n\nExplanation: <one sentence step-by-step explanation of your thought process>\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\n\nThe question is: ${THE_QUESTION}",
                                  "Provide the probability that your answer is correct. Give ONLY the probability, no other words or explanation.\n\nFor example:\n\nProbability: <the probability between 0.0 and 1.0 that your answer is correct, without any extra commentary whatsoever; just the probability!>"]}
        
        self.base_templates_disturb = {
                                            "zero_shot": [
                                                        "Provide your best answer for the following question. Give ONLY the answer, no other words or explanation.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Please provide your most accurate answer to the question below. Respond with ONLY the answer, no additional words or explanations.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Give your best answer to the following question. Include ONLY the answer—no explanations or extra words.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Please deliver your top answer for the question below. Use ONLY the answer—no explanations or extra words.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Submit your most likely answer to the following question. ONLY the answer is needed—no additional words or explanation.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Kindly provide your best answer to the question below. Include ONLY the answer, with no explanations or extra words.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Please offer your most accurate answer to the question below. ONLY the answer is required—no extra words or explanations.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Provide your most likely answer to the question below. Answer ONLY—no other words or explanations.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "Please submit your best answer to the following question. ONLY the answer is needed—no additional words or explanation.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                                                        "You need to provide the best answer to the following question. Give ONLY the answer, no additional words or explanations.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}"
                                                        ],
                                    }
        
        self.misleading_hint = {"Weak_Claim": ["I think the answer should be",
                                               "I read online the answer is",
                                               "I vaguely remember the answer is"],
                                "Strong_Claim": ["I am 80% sure the answer is",
                                                 "I am pretty sure the answer is",   
                                                 "Undoubtedly the answer is"],
                                "Externel Source": ["Wikipedia claims that the answer is",
                                                    "Latest research shows that the answer is",
                                                    "The textbook shows that answer is"]}

        self.base_templates_misleading = {
            "zero_shot": [
                "Provide your best answer for the following question. Give ONLY the answer, no other words or explanation.\n\nFor example:\n\nAnswer: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: I think the answer should be: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: I read online the answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: I vaguely remember the answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: I am 80% sure the answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: I am pretty sure the answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: Undoubtedly the answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: Wikipedia claims that the answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: Latest research shows that the answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}",
                "Provide your best answer for the following question. Give the answer in the following format: The textbook shows that answer is: <most likely answer, as short as possible; not a complete sentence, just the answer!>\nThe question is: {THE_QUESTION}"
            ],
        }

    def format_verb_prompt(self, question, type="vanilla", is2stage=False, k=None): 
        verb_types = ["topk", "vanilla", "multi_steps", "cot"]
        
        assert type in verb_types, "The prompt type is not available!"

        if type == "topk":
            assert k is not None, "Please provide the parameter k!"
        
        if type == "multi_steps":
            assert not is2stage, "multi_steps doesn't have 2 stage!"

        if not is2stage:
            if type == "topk":
                prompt = self.verb_templates_1S[type].format(THE_QUESTION=question, k=k)
            else:
                prompt = self.verb_templates_1S[type].format(THE_QUESTION=question)

            return prompt
        else:
            if type == "topk":
                prompt1 = self.verb_templates_2S[type][0].format(THE_QUESTION=question, k=k)
                prompt2 = self.verb_templates_2S[type][1].format(THE_QUESTION=question, k=k)
            else:
                prompt1 = self.verb_templates_2S[type][0].format(THE_QUESTION=question)
                prompt2 = self.verb_templates_2S[type][1].format(THE_QUESTION=question)

            return [prompt1, prompt2]
    
    def format_base_prompt(self, question, type):
        if type == "zero_shot":
            return self.base_templates["zero_shot"].format(THE_QUESTION=question)
        elif type == "cot":
            return self.base_templates["cot"].format(THE_QUESTION=question)

    def format_base_prompt_disturb(self, question, type):
        
        base_temp_list_formatted = []
        if type == "zero_shot":
            base_temp_list = self.base_templates_disturb["zero_shot"]
        elif type == "cot":
            base_temp_list = self.base_templates_disturb["cot"]

        for base_temp in base_temp_list:
            base_temp_list_formatted.append(base_temp.format(THE_QUESTION=question))
        return base_temp_list_formatted
        
    def format_base_prompt_misleading(self, question, type):
        base_temp_list_formatted = []
        if type == "zero_shot":
            base_temp_list = self.base_templates_misleading["zero_shot"]
        elif type == "cot":
            base_temp_list = self.base_templates_misleading["cot"]

        for base_temp in base_temp_list:
            base_temp_list_formatted.append(base_temp.format(THE_QUESTION=question))
        return base_temp_list_formatted

    
if __name__ == "__main__":
    prompt = myPrompt().format_base_prompt_misleading(question="hi", type="cot")
    for item in prompt:
        print(item)