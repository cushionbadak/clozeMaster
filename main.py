# main.py
import os
import re
import random
import logging
import argparse
from tqdm import tqdm
import subprocess
import csv
import atexit
import gc
import signal

# 안정성: 한 프로세스당 CPU 스레드 과도 사용 억제
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from model.IncoderModel import InCoder
from utils.masking import ClozeMask


# ============== 유틸 ==============

def get_rs_files(directory, suffix=('rs',)):
    files = []
    for root, _, fs in os.walk(directory):
        for f in fs:
            if f.split('.')[-1] in suffix:
                files.append(os.path.join(root, f))
    return files

def add_csv(filename, columns, new_line):
    need_header = (not os.path.exists(filename)) or (os.path.getsize(filename) == 0)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', newline='') as fp:
        wr = csv.writer(fp)
        if need_header:
            wr.writerow(columns)
        wr.writerow(new_line)

def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

def compile_rust(dirpath, rsfile, opt):
    cmd = f"rustc {rsfile} -C opt-level={opt} --out-dir temp"
    p = subprocess.Popen(cmd, cwd=dirpath, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         shell=True, text=True)
    try:
        out, err = p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p.terminate()
        return "timeout", ""

    rc = p.returncode
    low = err.lower()
    if "internal compiler error" in low or "compiler unexpectedly panicked" in low:
        return "ice", err
    if rc == 137:
        return "mem err", err
    if "process didn't exit successfully" in low:
        return "crash", err
    return "ok", err

def parse_err(err):
    # 요약용
    m = re.findall(r"thread 'rustc' panicked at.*?\n", err, re.DOTALL)
    err_info = m[0] if m else ""
    s, e = err.find("query stack during panic:"), err.find("end of query stack")
    stack = err[s:e]
    lines = []
    for line in stack.split("\n"):
        if "]" in line:
            lines.append(line[:line.find("]")+1])
    return err_info, "\n".join(lines)

def partition_indices(n_items, split, split_index_1based):
    # i % split == split_index-1
    keep = []
    k = split_index_1based - 1
    for i in range(n_items):
        if (i % split) == k:
            keep.append(i)
    return keep


# ============== 전역(단일 프로세스 자원) ==============
_INCODER = None
_MASKER = None

def _finalize():
    global _INCODER, _MASKER
    try:
        if _INCODER is not None and hasattr(_INCODER, "model"):
            try:
                _INCODER.model.to("cpu")
            except Exception:
                pass
        _INCODER = None
        _MASKER = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

def _sig_exit(signum, frame):
    _finalize()
    os._exit(0)

def init_model(model_path, tokenizer_path, device_str):
    atexit.register(_finalize)
    try:
        signal.signal(signal.SIGTERM, _sig_exit)
        signal.signal(signal.SIGINT, _sig_exit)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    global _INCODER, _MASKER
    device = torch.device(device_str)
    _INCODER = InCoder(model_path, tokenizer_path, device)
    _MASKER = ClozeMask()

def process_one_file(rs_path, temperature, out_root):
    """
    단일 .rs 파일을 변이 생성하여 out_root 하위에 저장.
    반환: [(mutant_abs_path, mutant_filename), ...]
    """
    with open(rs_path, 'r', errors='ignore') as f:
        code = f.read()
    if len(code) > 500:
        return []

    masked_list = _MASKER.mask_singel_code(code)

    fname = os.path.basename(rs_path)
    stem, ext = os.path.splitext(fname)

    # 입력 디렉터리 구조를 보존하려면 relpath 기준을 잡는다.
    # 여기선 rs_files 루트에서의 상대 경로를 유지하고 싶다면,
    # 호출자가 out_root만 주고, 아래에 적절히 하위 폴더를 구성해도 된다.
    out_dir = out_root  # 단순화: 한 루트에 평탄화 저장
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, masked in enumerate(masked_list, 1):
        new_code = _INCODER.code_infilling(masked, temperature=temperature)
        out_name = f"{stem}_{i}{ext}"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, 'w') as wf:
            wf.write(new_code)
        results.append((out_path, out_name))
    return results


# ============== 메인 ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', type=str, default="/clozeMaster/model/Incoder1b")
    ap.add_argument('--tokenizer_path', type=str, default="/clozeMaster/model/Incoder1b")
    ap.add_argument('--rs_files', type=str, default='./dataset/history_codes')

    ap.add_argument('--out_root', type=str, default='./target_dataset',
                    help='생성된 변이 파일을 저장할 루트 디렉터리')

    # 로그/CSV (분할 시 기본값을 쓰면 자동 접미사 부여)
    ap.add_argument('--csv_file', type=str, default='./log/bug.csv')
    ap.add_argument('--log_file', type=str, default='./log/demo.log')

    ap.add_argument('--multi_opt', action='store_true')
    ap.add_argument('--temperature', type=float, default=0.2)

    # 분할 실행(원시 병렬)
    ap.add_argument('--split', type=int, default=1, help='총 분할 개수')
    ap.add_argument('--split-index', type=int, default=1, help='이 실행이 담당할 1-base 분할 번호')
    ap.add_argument('--emit-splits-dir', type=str, default=None,
                    help='지정 시 split_1.txt … split_N.txt 파일을 생성(옵션)')

    # 선택적 섞기(모든 프로세스 동일 seed면 분배 일관)
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--shuffle-seed', type=int, default=0)

    args = ap.parse_args()

    if args.split < 1:
        raise SystemExit("--split must be >= 1")
    if not (1 <= args.split_index <= max(1, args.split)):
        raise SystemExit("--split-index must be in [1, --split]")

    # 분할 태그
    split_tag = f"s{args.split_index}-of-{args.split}" if args.split > 1 else "all"

    # 기본 로그/CSV면 자동 접미사
    def suffix_if_default(path, default_root, default_name):
        default_path = os.path.join(default_root, default_name)
        if os.path.normpath(path) == os.path.normpath(default_path) and args.split > 1:
            r, ext = os.path.splitext(path)
            return f"{r}.{split_tag}{ext}"
        return path

    args.log_file = suffix_if_default(args.log_file, "./log", "demo.log")
    args.csv_file = suffix_if_default(args.csv_file, "./log", "bug.csv")
    ensure_parent(args.log_file)
    ensure_parent(args.csv_file)

    logging.basicConfig(
        level=logging.INFO,
        filename=args.log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)-7s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s - %(message)s"))
    logging.getLogger("").addHandler(console)

    # 파일 수집
    files = get_rs_files(args.rs_files)
    files.sort()

    if args.shuffle:
        rnd = random.Random(args.shuffle_seed)
        rnd.shuffle(files)
        logging.info("Shuffle enabled (seed=%d)", args.shuffle_seed)

    # 분할 파일 목록 미리 출력(옵션)
    if args.emit_splits_dir:
        os.makedirs(args.emit_splits_dir, exist_ok=True)
        buckets = [[] for _ in range(args.split)]
        for i, p in enumerate(files):
            buckets[i % args.split].append(p)
        for k in range(args.split):
            with open(os.path.join(args.emit_splits_dir, f"split_{k+1}.txt"), "w") as f:
                for p in buckets[k]:
                    f.write(p + "\n")
        logging.info("Wrote split lists to %s", args.emit_splits_dir)

    # 이 프로세스가 담당할 분할 서브셋으로 축소
    if args.split > 1:
        keep = set(partition_indices(len(files), args.split, args.split_index))
        files = [p for i, p in enumerate(files) if i in keep]

    # 출력 루트: 분할 태그 하위 경로 사용
    out_root = args.out_root if args.split == 1 else os.path.join(args.out_root, split_tag)
    os.makedirs(out_root, exist_ok=True)

    logging.info("Total files for this run: %d", len(files))
    logging.info("Split: %s", split_tag)
    logging.info("Out root: %s", out_root)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda:0" if device.type == "cuda" and torch.cuda.device_count() > 0 else "cpu"
    init_model(args.model_path, args.tokenizer_path, device_str)

    # 변이 생성
    newfiles = []
    for fp in tqdm(files):
        newfiles.extend(process_one_file(fp, temperature=args.temperature, out_root=out_root))

    _finalize()

    logging.info("Generated mutants: %d", len(newfiles))

    # 컴파일 검증
    opts = ['0', '1', '2', '3', 's', 'z']
    for mpath, mname in tqdm(newfiles):
        d = os.path.dirname(mpath)
        if args.multi_opt:
            for opt in opts:
                status, err = compile_rust(d, mname, opt)
                ei, si = parse_err(err)
                if status != "ok":
                    add_csv(args.csv_file, ["filename", "opt", "status", "err_info", "stack_info"],
                            [mpath, opt, status, ei, si])
                logging.info("filename=%s opt=%s status=%s", mname, opt, status)
        else:
            status, err = compile_rust(d, mname, "0")
            ei, si = parse_err(err)
            if status != "ok":
                add_csv(args.csv_file, ["filename", "opt", "status", "err_info", "stack_info"],
                        [mpath, "0", status, ei, si])
            logging.info("filename=%s opt=0 status=%s", mname, status)


if __name__ == "__main__":
    main()
