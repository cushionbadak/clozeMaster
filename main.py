# main.py
import os
import re
import random
import logging
import argparse
from tqdm import tqdm
import subprocess
import time
import csv

# [PATCH] 프로세스당 CPU 스레드 폭주 방지 (가능한 한 일찍 설정)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# [PATCH] 종료/정리 유틸
import atexit
import gc
import signal

from model.IncoderModel import InCoder
from utils.masking import ClozeMask


def get_rs_files(directory, suffix=['rs']):
    rs_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in suffix:
                rs_files.append(os.path.join(root, file))
    return rs_files


def compare_text(text1, text2):
    text1 = text1.replace(' ', '').replace('\n', '').replace('\t', '')
    text2 = text2.replace(' ', '').replace('\n', '').replace('\t', '')
    return text1 == text2


def compile_rust(filepath, rsfile, opt):
    cmd = f"rustc {rsfile} -C opt-level={opt} --out-dir temp"
    time_limit = 60  # 재현을 빠르게 하려면 60, 안정성은 180
    p = subprocess.Popen(cmd, cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         shell=True, text=True)
    try:
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        low = err.lower()
        if "internal compiler error" in low or "compiler unexpectedly panicked" in low:
            return "ice", err
        elif returncode == 137:
            return "mem err", err
        elif "process didn't exit successfully" in low:
            return "crash", err
        else:
            return "ok", err
    except subprocess.TimeoutExpired:
        p.terminate()
        return "timeout", ""


def get_err(err):
    err_info = re.findall(r"thread 'rustc' panicked at.*?\n", err, re.DOTALL)
    if len(err_info) == 0:
        err_info = ""
    else:
        err_info = err_info[0]

    start = err.find("query stack during panic:")
    end = err.find("end of query stack")
    stack_info = err[start:end]
    stack_info = stack_info.split("\n")
    stack_info = [info[:info.find("]")+1] for info in stack_info if "]" in info]
    stack_info = "\n".join(stack_info)
    return err_info, stack_info


def add_csv(filename, columns, new_line_list):
    need_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if need_header:
            writer.writerow(columns)
        writer.writerow(new_line_list)


def ensure_file_path_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


# ---------------------------
# 프로세스 전역 (각 프로세스 1회 초기화)
# ---------------------------
_PROC_INCODER = None
_PROC_CLOZE_MASK = None
_PROC_READY = False

# [PATCH] 워커 종료 시 VRAM/리소스 정리
def _proc_finalize():
    global _PROC_INCODER, _PROC_CLOZE_MASK
    try:
        if _PROC_INCODER is not None and hasattr(_PROC_INCODER, "model"):
            try:
                _PROC_INCODER.model.to("cpu")
            except Exception:
                pass
        _PROC_INCODER = None
        _PROC_CLOZE_MASK = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

# [PATCH] SIGTERM/SIGINT 수신 시 정리 후 즉시 종료
def _sigterm_then_exit(signum, frame):
    _proc_finalize()
    os._exit(0)


def _proc_initializer(model_path, tokenizer_path, device_str):
    """
    각 워커 프로세스에서 1회 실행되어 모델/토크나이저/마스커를 로드.
    """
    # [PATCH] 종료 훅/시그널 → 먼저 등록
    atexit.register(_proc_finalize)
    try:
        signal.signal(signal.SIGTERM, _sigterm_then_exit)
        signal.signal(signal.SIGINT, _sigterm_then_exit)
    except Exception:
        pass

    # [PATCH] 프로세스당 CPU 스레드 수 제한 (프리즈 방지)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    global _PROC_INCODER, _PROC_CLOZE_MASK, _PROC_READY
    device = torch.device(device_str)
    _PROC_INCODER = InCoder(model_path, tokenizer_path, device)
    _PROC_CLOZE_MASK = ClozeMask()
    _PROC_READY = True


def _process_one_file(rs_file, temperature):
    """
    단일 파일을 처리하여 (masked_file_path, file_name) 리스트를 반환.
    배치 없이, 조기 종료가 활성화된 InCoder.generate 경로만 사용.
    """
    if not _PROC_READY:
        raise RuntimeError("Process not initialized")

    incoder = _PROC_INCODER
    cloze_mask = _PROC_CLOZE_MASK

    newfiles_local = []

    with open(rs_file, 'r', errors='ignore') as f:
        code = f.read()

    # 매우 긴 파일은 건너뜀(기존 동작 유지)
    if len(code) > 500:
        return newfiles_local

    masked_codes = cloze_mask.mask_singel_code(code)

    filename = os.path.basename(rs_file)
    stem, ext = os.path.splitext(filename)
    out_dir = os.path.dirname(rs_file).replace('dataset', 'target_dataset')
    os.makedirs(out_dir, exist_ok=True)

    for idx, masked in enumerate(masked_codes, start=1):
        new_code = incoder.code_infilling(masked, temperature=temperature)
        newfilename = f"{stem}_{idx}{ext}"
        masked_file = os.path.join(out_dir, newfilename)
        with open(masked_file, 'w') as wf:
            wf.write(new_code)
        newfiles_local.append((masked_file, newfilename))

    return newfiles_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/clozeMaster/model/Incoder1b")
    parser.add_argument('--tokenizer_path', type=str, default="/clozeMaster/model/Incoder1b")
    parser.add_argument('--rs_files', type=str, default='./dataset/history_codes')

    parser.add_argument('--csv_file', type=str, default='./log/bug.csv')
    parser.add_argument('--log_file', type=str, default='./log/demo.log')
    parser.add_argument('--multi_opt', action='store_true', help='여러 opt-level로 컴파일')

    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()

    ensure_file_path_exists(args.log_file)
    ensure_file_path_exists(args.csv_file)

    logging.basicConfig(level=logging.INFO,
                        filename=args.log_file,
                        filemode="w",
                        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # [PATCH] 콘솔 로그도 추가(걸리면 바로 보이게)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-8s - %(message)s"))
    logging.getLogger("").addHandler(console)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda:0" if device.type == "cuda" and torch.cuda.device_count() > 0 else "cpu"

    rs_files = get_rs_files(args.rs_files)
    random.shuffle(rs_files)

    logging.info('============================\n rs_files num:{}\n=========================\n'.format(len(rs_files)))
    logging.info("settings: temperature=%.3f, workers=%d", args.temperature, args.workers)

    newfiles = []

    if args.workers <= 1:
        _proc_initializer(args.model_path, args.tokenizer_path, device_str)
        for rs_file in tqdm(rs_files):
            res = _process_one_file(rs_file, temperature=args.temperature)
            newfiles.extend(res)
        _proc_finalize()
    else:
        # [PATCH] spawn 컨텍스트 고정
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_proc_initializer,
            initargs=(args.model_path, args.tokenizer_path, device_str)
        ) as ex:
            futures = [ex.submit(_process_one_file, rs_file, args.temperature) for rs_file in rs_files]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                exc = fut.exception()
                if exc:
                    logging.exception("Worker failed with %s: %s", type(exc).__name__, exc)
                    print(f"Worker failed with {type(exc).__name__}: {exc}")
                    continue
                res = fut.result()
                newfiles.extend(res)
            # [PATCH] 명시적 종료
            ex.shutdown(wait=True, cancel_futures=True)

    logging.info('============================\n newfiles num:{}\n=========================\n'.format(len(newfiles)))

    opts = ['0', '1', '2', '3', 's', 'z']
    csv_file = args.csv_file

    for masked_file, newfilename in tqdm(newfiles):
        if args.multi_opt:
            for opt in opts:
                status, err = compile_rust(os.path.dirname(masked_file), newfilename, opt)
                err_info, stack_info = get_err(err)
                if status != "ok":
                    add_csv(csv_file, ["filename", "opt", "status", "err_info", "stack_info"],
                            [masked_file, opt, status, err_info, stack_info])
                logging.info('filename:%s opt:%s status:%s err_info:%s stack_info:%s',
                             newfilename, opt, status, err_info, stack_info)
        else:
            opt = "0"
            status, err = compile_rust(os.path.dirname(masked_file), newfilename, opt)
            err_info, stack_info = get_err(err)
            if status != "ok":
                add_csv(csv_file, ["filename", "opt", "status", "err_info", "stack_info"],
                        [masked_file, opt, status, err_info, stack_info])
            logging.info('filename:%s opt:%s status:%s err_info:%s stack_info:%s',
                         newfilename, opt, status, err_info, stack_info)


if __name__ == "__main__":
    # [PATCH] 일부 환경에서 set_start_method 중복 호출 충돌 방지: 여기선 사용 안 함.
    main()
