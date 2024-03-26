##### FULL DOCUMENTATION https://huggingface.github.io/text-generation-inference/#/ #####

import re
import sys
import os
import json

import boto3
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel

class GenerateParameters(dict):
    def __init__(self, 
                best_of: int = None,
                decoder_input_details: bool = True,
                details: bool = True,
                do_sample: bool = False,
                max_new_tokens: int = 20,
                repetition_penalty: float = None,
                return_full_text: bool = None,
                seed: int = None,
                stop: list = [], #max 4
                temperature: float = 0.5,
                top_k: int = None,
                top_n_tokens: int = None,
                top_p: int = None,
                truncate: int = None,
                typical_p: float = None,
                watermark: bool = False, 
                 ):
        super().__init__()
        self.update({
            "best_of": best_of,
            "decoder_input_details": decoder_input_details,
            "details": details,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "return_full_text": return_full_text,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "top_k": top_k,
            "top_n_tokens": top_n_tokens,
            "top_p": top_p,
            "truncate": truncate,
            "typical_p": typical_p,
            "watermark": watermark,  
        })
    
class GenerateRequest(dict):
    def __init__(self, inputs: str, parameters: GenerateParameters = None):
        super().__init__()
        self.update({"inputs": inputs, "parameters": parameters})

class LogProbs(BaseModel):
    log_prob: float
    normalized_log_prob: float
    token_log_probs: list[dict[str, float]] | None

class TGI:
    
    def __init__(self, endpoint_name, region_name="us-east-1"):
        self._runtime = boto3.client("sagemaker-runtime", region_name=region_name)
        self.endpoint_name = endpoint_name
        self.region_name = region_name

    def sm_query(self, payload):
        response = self._runtime.invoke_endpoint(
            EndpointName = self.endpoint_name,
            ContentType = "application/json",
            Body = json.dumps(payload),
        )

        return json.loads(response["Body"].read().decode("utf8"))
    

    def create_from_objects(self, reqs: list[GenerateRequest]) -> list[str]:
        
        with ThreadPoolExecutor(max_workers=len(reqs)) as executor:
            responses = list(executor.map(self.sm_query, reqs))
            
        raw_responses = [res[0]['generated_text'] for res in responses]
        return raw_responses
    
    def is_greedy_generation(self, reqs: list, candidates: list) -> list[bool]:
        
        req_greedy = [GenerateRequest(req['reference'], GenerateParameters(max_new_tokens=len(length[0]['details']['prefill']), return_full_text=False, temperature=None, top_k=1)) for req, length in zip(reqs, candidates)]
        
        with ThreadPoolExecutor(max_workers=len(reqs)) as executor:
            responses = list(executor.map(self.sm_query, req_greedy))
        
        is_greedy_list = ["".join([res['text'] for res in response[0]['details']['tokens']]).lstrip() == "".join([can['text'] for can in candidate[0]['details']['prefill']]).lstrip() for response, candidate in zip(responses, candidates)]  
        return is_greedy_list
    
    def select_from_objects(self, reqs: list) -> list:
        parameter = GenerateParameters(best_of=1, do_sample=False, max_new_tokens=1, decoder_input_details=True)
        candidates = [GenerateRequest(cand['candidates'][0], parameter) for cand in reqs]
        requests = [GenerateRequest(req1['reference'] + req1['candidates'][0], parameter) if re.search(r"^\s", req1['candidates'][0]) else GenerateRequest(req1['reference'] + " " + req1['candidates'][0], parameter) for req1 in reqs]
        assert len(candidates) == len(requests) == len(reqs)
                
        with ThreadPoolExecutor(max_workers=len(reqs)) as executor: #threads
            candidates_tokens = list(executor.map(self.sm_query, candidates)) #to get the canditates tokens
            responses = list(executor.map(self.sm_query, requests))
        
        if reqs[0]['is_greedy']:
            greedy_list = self.is_greedy_generation(reqs, candidates_tokens)
        else:
            greedy_list = [False for _ in range(len(reqs))]

        selected = []
        for response, l, is_greedy in zip(responses, candidates_tokens, greedy_list):
            length = len(l[0]['details']['prefill'])
            candidate_result = response[0]['details']['prefill'][-length:] #get the tokens details for caditates
            tokens_log_prob = [{item['text']: item['logprob']} for item in candidate_result] 
            
            candidate_log_prob = sum([lp['logprob'] for lp in candidate_result]) #getting the sum of tokens log_probs for candidate
            assert isinstance(candidate_log_prob, float), candidate_log_prob
            candidate_log_prob_norm = candidate_log_prob / len(candidate_result) #candidate_log_prob / #_of_canditate_tokens
            assert isinstance(candidate_log_prob_norm, float), candidate_log_prob_norm
            
            log_probs=LogProbs(log_prob=candidate_log_prob, normalized_log_prob=candidate_log_prob_norm,token_log_probs=tokens_log_prob)
            
            sel = (log_probs, is_greedy)
            selected.append(sel) if sel is not None else selected.append(None)
                    
        assert len(selected) == len(reqs)
        return selected
    
if __name__ == "__main__":
    endpoint_name = "40B-py-over-ML-stack-ess-textbook-2024-02-05-08-00-11"
    region_name = "us-west-2"
    tgi = TGI(endpoint_name, region_name)
    stop_words = ["\nclass", "\ndef", "\n#", "\n```"]
    params = GenerateParameters( max_new_tokens=150, 
                            temperature=0.2, 
                            stop =stop_words, 
                            top_p = 0.95,
                            #return_log_probs = True,
                            )
    req = GenerateRequest(inputs="def advantages_equal(musician_1, musician_2): \"\"\"This exercise will determine the intersection of advantages between two musicians and check if they are equal.\"\"\"", parameters=params)
    
    response = tgi.sm_query(req)
    print(json.dumps(response, indent=2))