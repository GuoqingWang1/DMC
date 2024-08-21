from openai import OpenAI
import httpx
import os
import qianfan
import json
import requests
import time


class LLM(object):
    def __init__(self, model_class, model_name, key, orgnization=None, base_url=None):
        self.model_class = model_class
        self.model_name = model_name
        self.key = key
        self.orgnization = orgnization
        self.base_url = base_url
    
    def __query_GPT(self, prompt, dialogue_turns):
        client = OpenAI(api_key=self.key, 
                            organization=self.orgnization, 
                            base_url=self.base_url, 
                            )

        if dialogue_turns == 1:
            response = client.chat.completions.create(
                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
            )
            # print(response)
            return [response.choices[0].message.content, response.model]
        elif dialogue_turns == 2:
            # print(type(prompt))
            # print("prompt:", prompt)
            prompt1 = prompt[0]
            prompt2 = prompt[1]

            response1 = client.chat.completions.create(
            messages=[
                    {
                        "role": "user",
                        "content": prompt1
                    }
                ],
                model=self.model_name
            )

            response2 = client.chat.completions.create(
                messages=[
                        {"role": "user", "content": prompt2},
                        {"role": "assistant", "content": response1.choices[0].message.content},
                        {"role": "user", "content": prompt2}
                ],
                model=self.model_name
            )

            return [response1.choices[0].message.content, response2.choices[0].message.content, response1.model]
    
    def __query_Llama2(self, prompt, dialogue_turns):
        url = url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_70b?access_token=" + self.key
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        for iter in range(10):
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                return [json.loads(response.text)['result'], "Llama2-70B-chat"]
            except requests.exceptions.RequestException as e:
                if iter < 9:
                    time.sleep(2)
                else:
                    return None
    
    def query(self, prompt, dialogue_turns=1):
        if self.model_class == "GPT":
            return self.__query_GPT(prompt, dialogue_turns)
        elif self.model_class == "Llama2":
            return self.__query_Llama2(prompt, dialogue_turns)
if __name__ == '__main__':
    prompt = ["Mary's father has 5 daughters but no sons—Nana, Nene, Nini, Nono. What is the fifth daughter's name probably?",
              "Do you remember thinking at any point that 'Nunu' could be the answer?"]
    llm = LLM("GPT", "GPT-4", "sk-o21HlUxTlQRLcSed610dE680E13443A49fF91136510aC8Bd", "https://www.jcapikey.com/v1")
    res = llm.query(prompt, 2)
    print(res[0])
    print(res[1])