import os
import re
import random
import logging
from tqdm import tqdm
import subprocess
import time

from typing import List

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, tokenizer_path,device ):
    kwargs = {}
    logging.info(f"loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    model = model.half().to(device)
    logging.info(f"loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


class InCoder:
    def __init__(self, model_path, tokenizer_path, device):
        self.model, self.tokenizer = load_model(model_path, tokenizer_path, device)
        self.device = device
        self.BOS = "<|endoftext|>"
        self.EOM = "<|endofmask|>"

    def make_sentinel(self,i):
        return f"<|mask:{i}|>"
    

    # --- generate(): 조기 종료(early stop)로 속도 개선 ---
    # 배경
    # - 이전 구현은 max_length까지 계속 토큰을 생성해, EOM("<|endofmask|>")이 나온 뒤에도
    #   불필요한 토큰을 더 생성하는 일이 잦았음.
    # - 본 변경은 Hugging Face의 StoppingCriteria를 사용해, 매 스텝마다 출력의 "토큰 접미사"가
    #   EOM 토큰 시퀀스와 일치하면 즉시 중단하도록 함 → 생성 토큰 수 감소 → 평균 2~3배 속도 향상 가능
    #   (향상폭은 EOM이 얼마나 빨리 샘플되는지에 비례).
    #
    # 핵심 보존 사항(기존 호출부/후처리와의 호환성)
    # - 반환 문자열 형식 동일: 전체를 디코드한 뒤 BOS("<|endoftext|>")만 제거.
    # - 샘플링 설정(top_p=0.95, temperature)과 max_length 정책은 그대로 유지(안전망 역할).
    #
    # 구현 메모
    # - EOM은 다중 토큰일 수 있으므로, 문자열 포함이 아니라 "토큰 접미사 완전 일치"로 판정.
    # - 배치 크기 1을 가정(현재 코드 경로가 batch=1).
    # - Python 3.8 호환을 위해 list[list[int]] 대신 typing.List[List[int]] 사용.
    # - EOM이 분포상 잘 샘플되지 않거나 문맥상 늦게 등장하면 조기 종료가 일어나지 않아 느릴 수 있음.
    #   이런 경우 temperature를 낮추거나 top_p를 줄이는 것이 도움이 될 수 있음.
    #
    # 확장 팁(선택)
    # - 필요 시 EOM 외의 종료 마커(예: "```", 두 줄 개행 등)를 추가적인 stop 시퀀스로 함께 넘길 수 있음.
    #   예) other = self.tokenizer.encode("```", add_special_tokens=False)
    #       stopping_criteria = StoppingCriteriaList(
    #           [_StopOnSubsequence([eom_ids, other], device=self.device)]
    #       )

    def generate(self, input, max_to_generate=128, temperature=0.2):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StopOnSubsequence(StoppingCriteria):
            # Python 3.8 호환: typing.List 사용
            def __init__(self, stop_sequences: List[List[int]], device):
                super().__init__()
                # 비교 비용을 줄이기 위해 미리 (GPU) 텐서로 변환해 둠
                self.stop_sequences = [torch.tensor(s, device=device, dtype=torch.long) for s in stop_sequences]

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                # batch=1 가정. 마지막 L개 토큰이 stop 시퀀스와 정확히 일치하면 True
                seq = input_ids[0]
                for stop_ids in self.stop_sequences:
                    L = stop_ids.numel()
                    if L == 0 or seq.numel() < L:
                        continue
                    if torch.equal(seq[-L:], stop_ids):
                        return True
                return False

        # (1) 입력 토크나이즈 → 디바이스
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(self.device)

        # (2) 기존 정책 유지: max_length = 프롬프트 길이 + max_to_generate
        max_length = max_to_generate + input_ids.flatten().size(0)
        if max_length > 2048:
            logging.warning("warning: max_length {} is greater than the context window {}".format(max_length, 2048))

        # (3) EOM 토큰 시퀀스를 스톱 조건으로 등록(다중 토큰 고려)
        eom_ids = self.tokenizer.encode(self.EOM, add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([_StopOnSubsequence([eom_ids], device=self.device)])

        # (4) 생성: EOM이 나오면 즉시 중단, 그렇지 않으면 max_length까지 진행
        with torch.no_grad():
            # # performance(time) observation (starting point)
            # start_time = time.perf_counter()
            
            output = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                max_length=max_length,               # 안전망 유지
                stopping_criteria=stopping_criteria, # 조기 종료의 핵심
            )
            torch.cuda.empty_cache()

            # # performance(time) observation (end point)
            # output_time = time.perf_counter() - start_time
            # print(f"Generation time: {output_time:.3f} seconds")

        # (5) 디코딩: 기존과 동일하게 전체 디코딩 후 BOS 제거
        detok_hypo_str = self.tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
        if detok_hypo_str.startswith(self.BOS):
            detok_hypo_str = detok_hypo_str[len(self.BOS):]
        return detok_hypo_str


    def infill(self,parts,max_to_generate=128,temperature=0.2,extra_sentinel=True,max_retries=1):
        assert isinstance(parts, list)
        retries_attempted = 0
        done = False

        while (not done) and (retries_attempted < max_retries):
            retries_attempted += 1

            
            
            ## (1) build the prompt
            if len(parts) == 1:
                prompt = parts[0]
            else:
                prompt = ""
                # encode parts separated by sentinel
                for sentinel_ix, part in enumerate(parts):
                    prompt += part
                    if extra_sentinel or (sentinel_ix < len(parts) - 1):
                        # print("sentinel_ix:",sentinel_ix)
                        # prompt += self.make_sentinel(sentinel_ix)
                        prompt+=f"<|mask:{sentinel_ix}|>"
            
            infills = []
            complete = []

            done = True

            ## (2) generate infills
            for sentinel_ix, part in enumerate(parts[:-1]):
                complete.append(part)
                # prompt += self.make_sentinel(sentinel_ix)
                prompt+=f"<|mask:{sentinel_ix}|>"
                # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
                completion = self.generate(prompt, max_to_generate, temperature)
                completion = completion[len(prompt):]
                if self.EOM not in completion:
                    
                    completion += self.EOM
                    done = False
                completion = completion[:completion.index(self.EOM) + len(self.EOM)]
                infilled = completion[:-len(self.EOM)]
                infills.append(infilled)
                complete.append(infilled)
                prompt += completion
            complete.append(parts[-1])
            text = ''.join(complete)

        return {
            'text': text, # str, the completed document (with infills inserted)
            'parts': parts, # List[str], length N. Same as passed to the method
            'infills': infills, # List[str], length N-1. The list of infills generated
            'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
        } 
    
    def code_infilling(self,maskedCode,max_to_generate=128,temperature=0.2):
        parts = maskedCode.split("<insert>")
        result = self.infill(parts, max_to_generate=max_to_generate, temperature=temperature)
        # print("completed code:")
        # print(result["text"])
        return result["text"]



