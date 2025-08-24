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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import transformers

# incoder model reference: https://github.com/dpfried/incoder
# batch-usage upstream: https://github.com/dpfried/incoder/blob/main/example_batched_usage.py

# --- [권장] float16 + padding 환경에서 NaN 방지를 위한 causal mask 몽키패치 (HF 최신 시그니처 호환) ---
def _safe_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    past_key_values_length: int = 0,
    device: torch.device = None,
    **_: dict,  # 향후 HF 내부 인자 변화에 대비한 흡수 장치
):
    bsz, tgt_len = input_ids_shape
    if device is None:
        device = torch.device("cpu")

    # 마스크를 처음부터 target device+dtype로 생성
    # half 정밀도에서도 충분히 큰 음수로 억제: -1e4 (incoder/upstream 관행 유지)
    mask = torch.full((tgt_len, tgt_len), fill_value=-1e4, dtype=dtype, device=device)
    ar = torch.arange(tgt_len, device=device)
    mask.masked_fill_(ar < (ar + 1).view(tgt_len, 1), 0)

    if past_key_values_length > 0:
        prev = torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device)
        mask = torch.cat([prev, mask], dim=-1)

    # [bsz, 1, tgt_len, tgt_len + past_kv]로 브로드캐스트
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


try:
    transformers.models.xglm.modeling_xglm._make_causal_mask = _safe_make_causal_mask
except Exception:
    # 다른 백엔드/버전에서도 깨지지 않도록 안전하게 무시
    pass


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import logging

def load_model(model_path, tokenizer_path, device):
    kwargs = {}
    logging.info(f"loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).half().to(device)

    logging.info(f"loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
    )
    tokenizer.padding_side = "left"

    # --- 체크: bos_token 값 확인 ---
    if tokenizer.bos_token != "<|endoftext|>":
        raise ValueError(
            f"Unexpected bos_token: {tokenizer.bos_token!r}. "
            f"InCoder requires '<|endoftext|>' as bos_token."
        )
    logging.info(f"bos_token check passed: {tokenizer.bos_token!r}")

    # 토큰 사전 크기 동기화
    model.resize_token_embeddings(len(tokenizer))

    # 모델 config 동기화
    ids = dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    for k, v in ids.items():
        setattr(model.config, k, v)

    # generation_config 재생성
    try:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    except Exception:
        model.generation_config = GenerationConfig(
            bos_token_id=ids["bos_token_id"],
            eos_token_id=ids["eos_token_id"],
            pad_token_id=ids["pad_token_id"],
        )

    return model, tokenizer


class InCoder:
    def __init__(self, model_path, tokenizer_path, device):
        self.model, self.tokenizer = load_model(model_path, tokenizer_path, device)
        self.device = device
        self.BOS = "<|endoftext|>"
        self.EOM = "<|endofmask|>"

    def _eom_token_id(self):
        """
        EOM(\"<|endofmask|>\")가 단일 토큰이면 해당 ID(int)를 돌려주고,
        다중 토큰이면 None을 돌려준다.
        배치 안전 조기종료는 단일 토큰일 때만 HF의 eos_token_id로 처리한다.
        """
        try:
            ids = self.tokenizer.encode(self.EOM, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None


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
        eom_id = self._eom_token_id()

        # (1) 입력 토크나이즈 → 디바이스 (token_type_ids 미생성)

        enc = self.tokenizer(input, return_tensors="pt", return_token_type_ids=False, add_special_tokens=False)
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device) if hasattr(enc, "attention_mask") else None


        # (2) 기존 정책 유지: max_length = 프롬프트 길이 + max_to_generate
        max_length = max_to_generate + input_ids.flatten().size(0)
        if max_length > 2048:
            logging.warning("warning: max_length {} is greater than the context window {}".format(max_length, 2048))

        # (3) 조기 종료 설정
        #  - 단일 토큰이면 eos_token_id 사용(배치에서도 안전한 방식과 일치)
        #  - 다중 토큰이면 (batch=1 가정 경로에서는) 커스텀 스토퍼 사용
        stopping_criteria = None
        if eom_id is None:
            class _StopOnSubsequence(StoppingCriteria):
                def __init__(self, stop_sequences: List[List[int]], device):
                    super().__init__()
                    self.stop_sequences = [torch.tensor(s, device=device, dtype=torch.long) for s in stop_sequences]
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    seq = input_ids[0]  # batch=1
                    for stop_ids in self.stop_sequences:
                        L = stop_ids.numel()
                        if L == 0 or seq.numel() < L:
                            continue
                        if torch.equal(seq[-L:], stop_ids):
                            return True
                    return False
            eom_ids = self.tokenizer.encode(self.EOM, add_special_tokens=False)
            stopping_criteria = StoppingCriteriaList([_StopOnSubsequence([eom_ids], device=self.device)])

        # (4) 생성: EOM이 나오면 즉시 중단, 그렇지 않으면 max_length까지 진행
        with torch.no_grad():
            # # performance(time) observation (starting point)
            # start_time = time.perf_counter()

            gen_kwargs = dict(
                input_ids=input_ids,
                 do_sample=True,
                 top_p=0.95,
                 temperature=temperature,
                 max_length=max_length,               # 안전망 유지
                 pad_token_id=self.tokenizer.pad_token_id,
            )
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            elif eom_id is not None:
                gen_kwargs["eos_token_id"] = eom_id
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            output = self.model.generate(**gen_kwargs)
            # torch.cuda.empty_cache()

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

    #
    # Batch version of code_infilling
    #

    def _make_eom_stopper(self):
        """EOM(<|endofmask|>) 토큰 시퀀스를 접미사로 만나면 즉시 중단시키는 StoppingCriteriaList 생성."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        eom_ids = self.tokenizer.encode(self.EOM, add_special_tokens=False)

        class _StopOnSubsequence(StoppingCriteria):
            def __init__(self, stop_sequences, device):
                super().__init__()
                self.stop_sequences = [torch.tensor(s, device=device, dtype=torch.long) for s in stop_sequences]
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                # batch-aware
                for seq in input_ids:
                    for stop_ids in self.stop_sequences:
                        L = stop_ids.numel()
                        if seq.numel() >= L and torch.equal(seq[-L:], stop_ids):
                            return True
                return False

        return StoppingCriteriaList([_StopOnSubsequence([eom_ids], device=self.device)])

    
    def generate_batch(self, prompts, max_to_generate=128, temperature=0.2, generators=None):
        """
        여러 프롬프트를 한 번에 생성한다. 반환은 각 샘플의 '전체 디코드 문자열'(BOS 제거)이다.
        - generate()와 정책 동일: top_p=0.95, temperature, max_length=입력길이+max_to_generate
        - 조기 종료(안전): EOM 단일 토큰이면 eos_token_id 활용
        - 재현성: Hugging Face가 리스트 형태의 per-sample generator를 지원하는 버전이면 `generators`에 전달.
        """
        assert isinstance(prompts, list)
        if len(prompts) == 0:
            return []

        # EOM 단일 토큰 여부 확인
        eom_id = self._eom_token_id()
        if eom_id is None:
            # 배치 조기종료는 다중 토큰에서 안전하지 않다. 정확성을 우선하여 단일 샘플 경로로 폴백.
            logging.warning("EOM is multi-token for this tokenizer; falling back to per-sample generation for correctness.")
            return [self.generate(p, max_to_generate=max_to_generate, temperature=temperature) for p in prompts]

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            add_special_tokens=False,     # ← eos/pad 의존 제거
        )
        # 혹시 토크나이저 구현에 따라 들어오면 방어적으로 제거
        inputs = {k: v.to(self.device) for k, v in enc.items() if k != "token_type_ids"}
        max_length = inputs["input_ids"].shape[1] + max_to_generate
        if max_length > 2048:
            logging.warning("warning: max_length %d exceeds context window 2048", max_length)

        gen_arg = generators if generators is not None else None
        # 호환성: 일부 버전은 per-sample generator 리스트를 지원하지 않음
        if isinstance(gen_arg, list):
            gen_arg = gen_arg if hasattr(torch.Generator, "__iter__") else gen_arg[0]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # input_ids + attention_mask (token_type_ids 없음)
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                max_length=max_length,
                eos_token_id=eom_id,                  # ← 배치-안전한 조기 종료
                pad_token_id=self.tokenizer.pad_token_id,
                generator=gen_arg,
            )
        # torch.cuda.empty_cache()

        decoded = []
        pad_id = self.tokenizer.pad_token_id

        # 중요: 왼쪽 패딩 길이를 행(row)별로 계산하여 잘라낸 뒤 디코딩
        for row, out in enumerate(outputs):
            pad_len = int((inputs["input_ids"][row] == pad_id).sum().item()) if pad_id is not None else 0
            trimmed_ids = out[pad_len:]  # 왼쪽 PAD 제거

            detok = self.tokenizer.decode(trimmed_ids, clean_up_tokenization_spaces=False)
            if detok.startswith(self.BOS):
                detok = detok[len(self.BOS):]
            decoded.append(detok)

        return decoded



    def infill_batch(self, maskedCodes, max_to_generate=128, temperature=0.2, rng_seeds=None):
        """
        infill()과 동일한 동작을 '배치'로 수행한다.
        - extra_sentinel=True 동작 재현: 마지막 part 뒤에도 <|mask:N|> 부착
        - 마스크별(0..N-1) 순서로 스텝을 진행하며, 각 스텝은 배치(generate_batch)로 병렬 생성
        - 각 스텝에서 현재 마스크 토큰(<|mask:i|>)을 한 번 더 붙인 프롬프트로 생성(기존 infill과 동일)
        - EOM 기준으로 자른 뒤(EOM 제거) parts 사이에 infill만 끼워 최종 text 구성
        - rng_seeds: 재현성이 필요하면 샘플 수와 동일한 길이의 정수 리스트를 넘겨 per-sample 시드를 고정
        (주의: HF 버전에 따라 per-sample generator 리스트가 지원되지 않을 수 있음)
        """
        assert isinstance(maskedCodes, list)
        all_parts = [mc.split("<insert>") for mc in maskedCodes]
        num_gaps = [len(p) - 1 for p in all_parts]
        if not all_parts:
            return []

        # base prompt: extra_sentinel=True에 맞춰 각 part 뒤에 마스크 부착 (마지막 part 포함)
        base_prompts = []
        for parts in all_parts:
            if len(parts) == 1:
                base_prompts.append(parts[0])
            else:
                prompt = ""
                for sentinel_ix, part in enumerate(parts):
                    prompt += part
                    prompt += f"<|mask:{sentinel_ix}|>"
                base_prompts.append(prompt)

        dyn_prompts = list(base_prompts)
        collected_infills = [[] for _ in all_parts]
        max_gaps = max(num_gaps)

        # per-sample generator 준비(선택)
        generators = None
        if rng_seeds is not None:
            assert len(rng_seeds) == len(maskedCodes), "rng_seeds length must match maskedCodes"
            generators = [torch.Generator(device=self.device).manual_seed(int(s)) for s in rng_seeds]

        for gap_idx in range(max_gaps):
            active = [i for i, g in enumerate(num_gaps) if gap_idx < g]
            if not active:
                continue

            # 현재 마스크를 한 번 더 붙인 스텝 프롬프트(= infill과 동일)
            step_prompts = [dyn_prompts[i] + f"<|mask:{gap_idx}|>" for i in active]

            # per-step generator 매핑(선택)
            step_generators = None
            if generators is not None:
                step_generators = [generators[i] for i in active]

            # 배치 생성
            detoks = self.generate_batch(
                step_prompts,
                max_to_generate=max_to_generate,
                temperature=temperature,
                generators=step_generators,
            )

            # 각 샘플 후처리(EOM 자르기, infill만 추출, 누적)
            for row, detok in enumerate(detoks):
                i = active[row]
                prefix = step_prompts[row]
                if len(detok) < len(prefix):
                    completion = ""  # 안전장치
                else:
                    completion = detok[len(prefix):]

                if self.EOM not in completion:
                    completion += self.EOM
                completion = completion[: completion.index(self.EOM) + len(self.EOM)]
                infilled = completion[:-len(self.EOM)]

                collected_infills[i].append(infilled)
                dyn_prompts[i] += completion  # 다음 스텝 컨텍스트로 누적

        # 최종 조립
        results = []
        for parts, infills in zip(all_parts, collected_infills):
            if len(parts) == 1:
                results.append(parts[0])
            else:
                if len(infills) != len(parts) - 1:
                    infills = (infills + [""] * (len(parts) - 1))[:len(parts) - 1]
                merged = []
                for p, f in zip(parts[:-1], infills):
                    merged.append(p)
                    merged.append(f)
                merged.append(parts[-1])
                results.append("".join(merged))
        return results


    def code_infilling_batch(self, maskedCodes, max_to_generate=128, temperature=0.2, rng_seeds=None):
        """
        배치 인필링의 쉬운 엔트리 포인트. infill_batch(...) 호출로 구현을 위임.
        """
        return self.infill_batch(
            maskedCodes,
            max_to_generate=max_to_generate,
            temperature=temperature,
            rng_seeds=rng_seeds,
        )
