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
    model.eval()  # <-- 추론 전용 모드로 고정 (드롭아웃/Autograd 오버헤드 제거)

    logging.info(f"loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
    )
    tokenizer.padding_side = "left"

    if tokenizer.bos_token != "<|endoftext|>":
        raise ValueError(
            f"Unexpected bos_token: {tokenizer.bos_token!r}. "
            f"InCoder requires '<|endoftext|>' as bos_token."
        )
    logging.info(f"bos_token check passed: {tokenizer.bos_token!r}")

    model.resize_token_embeddings(len(tokenizer))

    ids = dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    for k, v in ids.items():
        setattr(model.config, k, v)

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
        - top_p=0.95, temperature, max_length=입력길이+max_to_generate (기존 정책 유지)
        - EOM 단일 토큰이면 eos_token_id로 조기 종료
        - pad_to_multiple_of=8 + attention_mask 명시 (커널 효율)
        """
        assert isinstance(prompts, list)
        if len(prompts) == 0:
            return []

        eom_id = self._eom_token_id()
        if eom_id is None:
            # 정확성 우선: EOM 다중 토큰이면 배치 안전 조기종료 불가 → 단일 샘플 경로로 폴백
            logging.warning("EOM is multi-token for this tokenizer; falling back to per-sample generation for correctness.")
            return [self.generate(p, max_to_generate=max_to_generate, temperature=temperature) for p in prompts]

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,           # <-- 텐서코어/메모리 정렬에 유리
            return_attention_mask=True,     # <-- 명시적 마스크
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        inputs = {k: v.to(self.device) for k, v in enc.items() if k != "token_type_ids"}
        max_length = inputs["input_ids"].shape[1] + max_to_generate
        if max_length > 2048:
            logging.warning("warning: max_length %d exceeds context window 2048", max_length)

        gen_arg = generators if generators is not None else None
        if isinstance(gen_arg, list):
            # 일부 HF 버전은 리스트 미지원 → 첫 번째만 사용
            gen_arg = gen_arg if hasattr(torch.Generator, "__iter__") else gen_arg[0]

        with torch.inference_mode():  # <-- no_grad보다 더 얕은 오버헤드
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                max_length=max_length,
                eos_token_id=eom_id,
                pad_token_id=self.tokenizer.pad_token_id,
                generator=gen_arg,
            )

        decoded = []
        pad_id = self.tokenizer.pad_token_id

        for row, out in enumerate(outputs):
            # 왼쪽 PAD 제거 후 디코드 (padding_side='left' 유지)
            pad_len = int((inputs["input_ids"][row] == pad_id).sum().item()) if pad_id is not None else 0
            trimmed_ids = out[pad_len:]

            detok = self.tokenizer.decode(trimmed_ids, clean_up_tokenization_spaces=False)
            if detok.startswith(self.BOS):
                detok = detok[len(self.BOS):]
            decoded.append(detok)

        return decoded
    
    @staticmethod
    def _bucket_by_length(indices, lengths, max_bs=20, max_span=128):
        """
        길이 기반 버킷팅:
        - 같은 버킷 안에서는 (최대길이 - 최소길이) <= max_span 이 되도록 묶음
        - 한 버킷의 최대 배치 크기는 max_bs
        """
        pairs = sorted(((i, lengths[i]) for i in indices), key=lambda x: x[1])
        batches, cur, base = [], [], None
        for i, L in pairs:
            if not cur:
                cur, base = [i], L
            elif (len(cur) < max_bs) and (L - base <= max_span):
                cur.append(i)
            else:
                batches.append(cur); cur, base = [i], L
        if cur:
            batches.append(cur)
        return batches

    def infill_batch(self, maskedCodes, max_to_generate=128, temperature=0.2, rng_seeds=None,
                    bucket_max_bs=20, bucket_max_span=128):
        """
        infill()과 동일한 동작을 '배치'로 수행(동일 출력 보장).
        개선점:
        - 스텝별(step_prompts)로 길이 버킷팅하여 패딩 낭비 최소화
        - generate_batch는 pad_to_multiple_of=8, inference_mode 사용
        파라미터:
        - bucket_max_bs: 버킷 내 최대 배치 크기(기본 20)
        - bucket_max_span: 같은 버킷 내 허용 길이 차(토큰 수, 기본 128)
        """
        assert isinstance(maskedCodes, list)
        all_parts = [mc.split("<insert>") for mc in maskedCodes]
        num_gaps = [len(p) - 1 for p in all_parts]
        if not all_parts:
            return []

        # base prompt: extra_sentinel=True (마지막 part 뒤에도 마스크 부착)
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

            # 현재 마스크를 한 번 더 붙인 스텝 프롬프트(원본 infill과 동일)
            step_prompts = [dyn_prompts[i] + f"<|mask:{gap_idx}|>" for i in active]
            step_generators = None
            if generators is not None:
                step_generators = [generators[i] for i in active]

            # ===== 길이 버킷팅 시작 =====
            # 길이(토큰 수) 계산 (special tokens 미포함)
            step_lengths = [len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in step_prompts]
            idxs = list(range(len(step_prompts)))
            buckets = self._bucket_by_length(idxs, step_lengths, max_bs=bucket_max_bs, max_span=bucket_max_span)

            # 버킷별로 generate_batch 호출 → 원래 순서로 재배치
            detoks_all = [None] * len(step_prompts)
            for bucket in buckets:
                sub_prompts = [step_prompts[j] for j in bucket]
                sub_gens = [step_generators[j] for j in bucket] if step_generators is not None else None
                sub_out = self.generate_batch(
                    sub_prompts,
                    max_to_generate=max_to_generate,
                    temperature=temperature,
                    generators=sub_gens,
                )
                for j, out in zip(bucket, sub_out):
                    detoks_all[j] = out
            # ===== 길이 버킷팅 끝 =====

            # 각 샘플 후처리(EOM 자르기, infill만 추출, 다음 스텝 컨텍스트 누적)
            for row, detok in enumerate(detoks_all):
                i = active[row]
                prefix = step_prompts[row]
                if detok is None:
                    completion = ""
                elif len(detok) < len(prefix):
                    completion = ""
                else:
                    completion = detok[len(prefix):]

                if self.EOM not in completion:
                    completion += self.EOM
                completion = completion[: completion.index(self.EOM) + len(self.EOM)]
                infilled = completion[:-len(self.EOM)]

                collected_infills[i].append(infilled)
                dyn_prompts[i] += completion

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
