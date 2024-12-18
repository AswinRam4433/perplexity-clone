# Synchronous Example
from mistralai import Mistral
import yaml
import os
import logging

def MistralHealthCheck(run=False):
    with open(r"D:\Perplexity\config.yaml", "r") as file:
        config = yaml.safe_load(file)

    with Mistral(
        api_key=config['API']['key'],
    ) as s:
        res = s.chat.complete(model="mistral-small-latest", messages=[
            {
                "content": "Who is the best French painter? Answer in one short sentence.",
                "role": "user",
            },
        ])

        if res is not None:
            print(res)
            logging.info("Mistral Health Check Passed")
            return 0
        else:
            logging.error("Mistral Health Check Failed")
            return -1



if __name__=="__main__":
    assert MistralHealthCheck()==0