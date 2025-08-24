# IncoderModel.py
"""
이 모듈은 배치 로직을 전부 제거하고,
- 조기 종료(early-stop) 기반 단일 샘플 생성만 남겼습니다.
- 기존 반환 규약(BOS 제거 후 문자열 반환)과 infill/code_infilling 동작은 동일합니다.
"""

import logging
from typing import List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# --- float16 + padding 환경에서 NaN 방지를 위한 causal mask 몽키패치 (HF 버전 차이 흡수) ---
def _safe_make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    past_key_values_length: int = 0,
    device: torch.device = None,
    **_: dict,
):
    bsz, tgt_len = input_ids_shape
    if device is None:
        device = torch.device("cpu")
    mask = torch.full((tgt_len, tgt_len), fill_value=-1e4, dtype=dtype, device=device)
    ar = torch.arange(tgt_len, device=device)
    mask.masked_fill_(ar < (ar + 1).view(tgt_len, 1), 0)

    if past_key_values_length > 0:
        prev = torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device)
        mask = torch.cat([prev, mask], dim=-1)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


try:
    transformers.models.xglm.modeling_xglm._make_causal_mask = _safe_make_causal_mask
except Exception:
    pass


def load_model(model_path, tokenizer_path, device):
    logging.info(f"loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)
    model.eval()  # 추론 전용

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

    # 토큰 사전 크기 동기화 (신규 토큰이 실제로 추가된 경우에만 늘어남)
    model.resize_token_embeddings(len(tokenizer))

    # 모델/GenerationConfig에 special token id를 명시적으로 고정
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
    BOS = "<|endoftext|>"
    EOM = "<|endofmask|>"

    def __init__(self, model_path, tokenizer_path, device):
        self.model, self.tokenizer = load_model(model_path, tokenizer_path, device)
        self.device = device

    def make_sentinel(self, i: int) -> str:
        return f"<|mask:{i}|>"

    def _eom_token_id(self):
        """
        EOM("<|endofmask|>")가 단일 토큰이면 해당 ID(int)를 반환.
        다중 토큰이면 None을 반환.
        """
        try:
            ids = self.tokenizer.encode(self.EOM, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    # --- generate(): 조기 종료(early-stop)로 불필요한 토큰 생성을 차단 ---
    # - EOM이 단일 토큰: eos_token_id로 종료
    # - EOM이 다중 토큰: 커스텀 StoppingCriteria로 "토큰 접미사 일치" 시 종료
    def generate(self, input_text: str, max_to_generate: int = 128, temperature: float = 0.2) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList

        eom_id = self._eom_token_id()

        # 입력 인코딩 (special tokens 미추가)
        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device) if hasattr(enc, "attention_mask") else None

        max_length = max_to_generate + input_ids.flatten().size(0)
        if max_length > 2048:
            logging.warning("warning: max_length %d > context window %d", max_length, 2048)

        # 다중 토큰 EOM이면 커스텀 스토퍼 사용
        stopping_criteria = None
        if eom_id is None:
            class _StopOnSubsequence(StoppingCriteria):
                def __init__(self, stop_sequences: List[List[int]], device):
                    super().__init__()
                    self.stop_sequences = [torch.tensor(s, device=device, dtype=torch.long) for s in stop_sequences]
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    seq = input_ids[0]  # batch=1 가정
                    for stop_ids in self.stop_sequences:
                        L = stop_ids.numel()
                        if L and seq.numel() >= L and torch.equal(seq[-L:], stop_ids):
                            return True
                    return False

            eom_ids = self.tokenizer.encode(self.EOM, add_special_tokens=False)
            stopping_criteria = StoppingCriteriaList([_StopOnSubsequence([eom_ids], device=self.device)])

        with torch.inference_mode():
            gen_kwargs = dict(
                input_ids=input_ids,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            elif eom_id is not None:
                gen_kwargs["eos_token_id"] = eom_id

            output = self.model.generate(**gen_kwargs)

        # 디코드: 전체를 디코드한 뒤 BOS 제거 (기존 규약 유지)
        detok = self.tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
        if detok.startswith(self.BOS):
            detok = detok[len(self.BOS):]
        return detok

    def infill(self, parts: List[str], max_to_generate: int = 128, temperature: float = 0.2,
               extra_sentinel: bool = True, max_retries: int = 1):
        assert isinstance(parts, list)
        retries_attempted = 0
        done = False

        while (not done) and (retries_attempted < max_retries):
            retries_attempted += 1

            # (1) 프롬프트 구성
            if len(parts) == 1:
                prompt = parts[0]
            else:
                prompt = ""
                for sentinel_ix, part in enumerate(parts):
                    prompt += part
                    if extra_sentinel or (sentinel_ix < len(parts) - 1):
                        prompt += f"<|mask:{sentinel_ix}|>"

            infills = []
            complete = []
            done = True

            # (2) 갭마다 생성
            for sentinel_ix, _ in enumerate(parts[:-1]):
                complete.append(parts[sentinel_ix])
                prompt += f"<|mask:{sentinel_ix}|>"

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
            'text': text,
            'parts': parts,
            'infills': infills,
            'retries_attempted': retries_attempted,
        }

    def code_infilling(self, maskedCode: str, max_to_generate: int = 128, temperature: float = 0.2) -> str:
        parts = maskedCode.split("<insert>")
        result = self.infill(parts, max_to_generate=max_to_generate, temperature=temperature)
        return result["text"]
