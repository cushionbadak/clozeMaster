import os
import re
import random
import logging

import argparse
from tqdm import tqdm
import subprocess
import time
import csv
import pandas as pd

import torch
import tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer
# 중복 import 제거
# import pandas as pd

from model.IncoderModel import InCoder
from utils.masking import ClozeMask


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_rs_files(directory, suffix=['rs']):
    rs_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.split('.')[-1] in suffix:
                rs_files.append(os.path.join(root, file))
    return rs_files

def compare_text(text1, text2):
    text1 = text1.replace(' ', '').replace('\n', '').replace('\t', '')
    text2 = text2.replace(' ', '').replace('\n', '').replace('\t', '')
    return text1 == text2

def compile_rust(filepath, rsfile, opt):
    cmd = "rustc {} -C opt-level={}  --out-dir temp".format(rsfile, opt)
    time_limit = 60  # stable is 180; while 60 make the reproduction faster
    p = subprocess.Popen(cmd, cwd=filepath, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    try:
        out, err = p.communicate(timeout=time_limit)
        returncode = p.returncode
        if "internal compiler error" in err.lower() or "compiler unexpectedly panicked" in err.lower():
            return "ice", err
        elif returncode == 137:
            return "mem err", err
        elif "process didn't exit successfully" in err.lower():
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
    stack_info = [info[:info.find("]")+1] for info in stack_info]
    stack_info = "\n".join(stack_info)
    return err_info, stack_info

def add_csv(filename, columns, new_line_list):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(columns)
        writer.writerow(new_line_list)

def ensure_file_path_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/clozeMaster/model/Incoder1b")
    parser.add_argument('--tokenizer_path', type=str, default="/clozeMaster/model/Incoder1b")

    parser.add_argument('--rs_files', type=str, default='./dataset/history_codes')

    parser.add_argument('--csv_file', type=str, default='./log/bug.csv')
    parser.add_argument('--log_file', type=str, default='./log/demo.log')
    parser.add_argument('--multi_opt', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4,
                        help="외부 배치 분할 크기(기본: 4). masked_codes를 이 크기로 나눠 순차 처리.")
    parser.add_argument('--bucket-max-bs', dest='bucket_max_bs', type=int, default=20,
                        help="길이 버킷 최대 배치 크기 (generate_batch 내부, 기본: 20).")
    parser.add_argument('--bucket-max-span', dest='bucket_max_span', type=int, default=128,
                        help="동일 버킷 내 허용 길이 차(토큰 수, 기본: 128).")
    parser.add_argument(
        '--no_batch', action='store_true',
        help="배치 사용 없이 단일 호출(code_infilling) 경로로 실행."
    )

    args = parser.parse_args()
    ensure_file_path_exists(args.log_file)
    ensure_file_path_exists(args.csv_file)

    logging.basicConfig(level=logging.INFO,
                        filename=args.log_file,
                        filemode="w",
                        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    incoder = InCoder(model_path, tokenizer_path, device)
    cloze_mask = ClozeMask()

    opts = ['0', '1', '2', '3', 's', 'z']
    csv_file = args.csv_file

    rs_files = get_rs_files(args.rs_files)
    random.shuffle(rs_files)

    logging.info('============================\n rs_files num:{}\n=========================\n'.format(len(rs_files)))
    logging.info("settings: batch_size=%d, bucket_max_bs=%d, bucket_max_span=%d, no_batch=%s",
                 args.batch_size, args.bucket_max_bs, args.bucket_max_span, args.no_batch)

    newfiles = []
    for rs_file in tqdm(rs_files):
        with open(rs_file, 'r', errors='ignore') as f:
            code = f.read()
        if len(code) > 500:
            continue
        masked_codes = cloze_mask.mask_singel_code(code)

        filename = os.path.basename(rs_file)
        stem, ext = os.path.splitext(filename)

        out_dir = os.path.dirname(rs_file).replace('dataset', 'target_dataset')
        os.makedirs(out_dir, exist_ok=True)

        if args.no_batch:
            logging.info("no-batch 모드: code_infilling 경로 사용")
            for idx, masked in enumerate(masked_codes, start=1):
                new_code = incoder.code_infilling(masked, temperature=0.2)
                newfilename = f"{stem}_{idx}{ext}"
                masked_file = os.path.join(out_dir, newfilename)
                with open(masked_file, 'w') as f:
                    f.write(new_code)
                newfiles.append((masked_file, newfilename))
        else:
            # 외부 분할 배치 루프
            for start in range(0, len(masked_codes), args.batch_size):
                batch = masked_codes[start:start + args.batch_size]

                # 1순위: code_infilling_batch가 버킷 인자를 받는 경우
                try:
                    new_codes = incoder.code_infilling_batch(
                        batch, temperature=0.2, rng_seeds=None,
                        bucket_max_bs=args.bucket_max_bs,
                        bucket_max_span=args.bucket_max_span
                    )
                except TypeError:
                    logging.warning("code_infilling_batch가 bucket 인자를 지원하지 않음 → infill_batch로 폴백 시도")
                    # 2순위: infill_batch 직접 호출
                    try:
                        new_codes = incoder.infill_batch(
                            batch, temperature=0.2,
                            bucket_max_bs=args.bucket_max_bs,
                            bucket_max_span=args.bucket_max_span
                        )
                    except TypeError:
                        logging.warning("infill_batch도 bucket 인자를 지원하지 않음 → 기본 배치 경로 사용")
                        new_codes = incoder.code_infilling_batch(batch, temperature=0.2)

                for local_idx, new_code in enumerate(new_codes, start=1):
                    global_idx = start + local_idx
                    newfilename = f"{stem}_{global_idx}{ext}"
                    masked_file = os.path.join(out_dir, newfilename)
                    with open(masked_file, 'w') as f:
                        f.write(new_code)
                    newfiles.append((masked_file, newfilename))

    logging.info('============================\n newfiles num:{}\n=========================\n'.format(len(newfiles)))

    bug_count = 0
    for masked_file, newfilename in tqdm(newfiles):
        if args.multi_opt:
            for opt in opts:
                status, err = compile_rust(os.path.dirname(masked_file), newfilename, opt)
                err_info, stack_info = get_err(err)
                if status != "ok":
                    bug_count += 1
                    add_csv(csv_file, ["filename", "opt", "status", "err_info", "stack_info"],
                            [masked_file, opt, status, err_info, stack_info])
                logging.info('filename:{} opt:{} status:{} err_info:{} stack_info:{}'
                             .format(newfilename, opt, status, err_info, stack_info))
        else:
            opt = "0"
            status, err = compile_rust(os.path.dirname(masked_file), newfilename, opt)
            err_info, stack_info = get_err(err)
            if status != "ok":
                add_csv(csv_file, ["filename", "opt", "status", "err_info", "stack_info"],
                        [masked_file, opt, status, err_info, stack_info])
            logging.info('filename:{} opt:{} status:{} err_info:{} stack_info:{}'
                         .format(newfilename, opt, status, err_info, stack_info))
