# ClozeMaster-demo

Artifacts for "ClozeMaster: Fuzzing Rust Compiler by Harnessing LLMs for Infilling Masked Real Programs".

## Introduction to this custom branch

This repository is a fork of the original `https://github.com/clozeMasterPro/clozeMaster`.  
The `faster` branch introduces two main updates designed to improve execution speed and simplify parallel workflows:

1. **Early stop on EOM**  
   Decoding stops immediately once the model generates the special token `<|endofmask|>`.  
   This avoids producing redundant tokens and shortens generation time.

2. **Raw parallelism via dataset splitting**  
   Instead of relying on complex in-process worker pools, this branch supports *external parallel execution*:  
   - The input dataset (`--rs_files`) can be divided into *N deterministic shards* with `--split` and `--split-index`.  
   - Each `main.py` process runs independently on its shard, using its own log and output directory.  
   - Users can launch multiple processes in parallel (e.g., on the same GPU with sufficient VRAM, or across GPUs/machines).  
   - Optional `--emit-splits-dir` makes the exact file lists (`split_1.txt … split_N.txt`) visible for orchestration or debugging.

Additional notes:

- The infilling protocol (sentinels like `<|mask:i|>`) and post-processing are unchanged.  
- Special tokens are explicitly fixed: `BOS=<|endoftext|>`, `EOM=<|endofmask|>`, `PAD=<pad>`.  
- Sampling remains conservative (`top_p=0.95`, `temperature` configurable).  

### use this branch in docker

1. run docker container with origina clozemaster docker image
2. overwrite `main.py` and `model/IncoderModel.py` file
3. Run `python3 main.py --workers N` for N parallelism.

### Parallelism Options (raw parallel via dataset splitting)

This branch removes fragile in-process workers and instead supports **raw parallelism** by
splitting the dataset and launching multiple independent `main.py` processes.

#### Key flags

- `--split N`  
  Total number of shards (equal to how many parallel processes you plan to launch).

- `--split-index K` (1-based)  
  Which shard this process should run. Valid range: `1..N`.

- `--emit-splits-dir DIR` (optional)  
  Write deterministic file lists for all shards: `split_1.txt … split_N.txt`.  
  Useful for inspection or submitting to external schedulers. Splitting still works without this.

- `--shuffle` and `--shuffle-seed SEED` (optional)  
  If you need randomized order, enable `--shuffle`.  
  Use the **same** `--shuffle-seed` across all processes to keep shards consistent.  
  Determinism pipeline: `sorted files` → `optional deterministic shuffle` → `modulo partition`.

- `--out_root PATH`  
  Where generated mutants are saved.  
  If `--split > 1`, outputs go under a tagged subfolder:  
  `PATH/sK-of-N/…` (e.g., `target_dataset/s2-of-3/*.rs`).

- `--log_file` and `--csv_file`  
  If you keep defaults (`./log/demo.log`, `./log/bug.csv`) **and** use `--split > 1`,  
  the program auto-suffixes by shard to avoid clobbering:  
  `./log/demo.sK-of-N.log`, `./log/bug.sK-of-N.csv`.  
  If you pass custom paths, they are respected as-is.

> Notes
>
> - All processes must use the **same** `--rs_files`, `--split`, and (if used) `--shuffle`/`--shuffle-seed` to obtain a consistent partition.
> - Splitting is **deterministic** for a given input tree and options.
> - Internal worker pools are **removed**; true parallelism = how many `main.py` you start.
> - By default the code picks `cuda:0`. To use multiple GPUs, set per-process `CUDA_VISIBLE_DEVICES` (each process will then use its own `cuda:0`).

### Quick examples

**Run a single process (no split)**

```bash
python3 main.py --rs_files ./dataset/history_codes
```

**3-way parallel on one GPU (sufficient VRAM)**

```bash
python3 main.py --split 3 --split-index 1
python3 main.py --split 3 --split-index 2
python3 main.py --split 3 --split-index 3
```

**Make shard lists first, then run (deterministic shuffle)**

```bash
# 1) Inspect deterministic shards
python3 main.py --split 3 --emit-splits-dir ./splits --shuffle --shuffle-seed 42

# 2) Launch three processes with the same shuffle/seed
python3 main.py --split 3 --split-index 1 --shuffle --shuffle-seed 42
python3 main.py --split 3 --split-index 2 --shuffle --shuffle-seed 42
python3 main.py --split 3 --split-index 3 --shuffle --shuffle-seed 42
```

**Bash loop launcher (N shards on one machine)**
Not recommended - it's not easy to recover when it goes wrong.

```bash
N=4
for K in $(seq 1 $N); do
  CUDA_VISIBLE_DEVICES=0 \
  nohup python3 main.py --split $N --split-index $K \
        --rs_files ./dataset/history_codes \
        --out_root ./target_dataset \
        --temperature 0.2 \
        > "run.$K.out" 2>&1 &
done
```

### Troubleshooting & guarantees

#### Determinism

The file list is sorted, then optionally shuffled with a fixed seed, then partitioned by index modulo N.
Same inputs + same flags ⇒ same shards.

#### Imbalance by remainder

If the file count M is not divisible by N, some shards get one extra file (⌈M/N⌉ vs ⌊M/N⌋).

#### Changing the dataset mid-run

Adding/removing files in --rs_files between starting processes will change partitions.
If stability is critical, generate split lists first (--emit-splits-dir) and avoid modifying the tree.

#### Logs & outputs

Each shard writes to its own sK-of-N output dir and suffixed log/CSV (when defaults are used),
so runs are isolated and can be resumed independently.

## Introduction

ClozeMaster is a novel fuzzing tool that leverages large language models (LLMs) to generate effective test cases for Rust compilers. The key idea behind ClozeMaster is to identify the bracket structure of given code and use it to guide the generation of new test cases through masked token completion.
<br>
This approach is very simple and easy to implement, and has achieved good practical application results in detecting defects in compilers of complex programming languages with limited training data (such as Rust). It is also easily transferable to the compilers of other relatively mature languages (such as C/C++).

## Install from Docker Image

![](https://camo.githubusercontent.com/01a2f5a54eeb55937da4855adcecdf816f84aedca15ddf624cdeea870e646377/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5265636f6d6d656e6465642d5965732d627269676874677265656e)

We highly recommend using Docker images to directly run our method framework, which can avoid the failure of reproduction due to issues like dependency packages.

```sh
docker pull clozemaster/cloze:v1.0
docker run -it --net=host --gpus all --name cloze -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all clozemaster/cloze:v1.0
```

Under the `/clozeMaster` directory in the container, you can see all our project files and datasets.
Activate the py38 environment with conda, and run the main.py script under the `/clozeMaster` directory. You will be able to see the running logs under `./log` and the generated test code under `./target_data`.

```sh
cd /clozeMaster
conda activate py38
python main.py
```

## Install from Source Code

![](https://camo.githubusercontent.com/bbadbad4f2dfb3e652072d7e3d5725c7245ba1e2ff0f76f49d3e323c42b04385/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5265636f6d6d656e6465642d4e6f2d726564)

Before using this tool, please ensure that the following development tools are installed on your computer:

- python>=3.8
- rustc (1.73)

You have to install all the libraries listed in `requirements.txt`

```sh
pip install -r requirements.txt
```

Additionally, ClozeMaster utilizes the [Incoder-1B](https://huggingface.co/facebook/incoder-1B), so please make sure your computer has sufficient memory and GPU resources to run the local inference of the LLM.

## Usage

## Overflow

```sh
conda activate py38
python main.py --model_path your_model_path \
--tokenizer_path your_tokenizer_path \
--rs_files your_rust_files_path \
--csv_file path_which_you_want_to_log_the_bug_information \
--log_file path_which_you_want_to_log_the_fuzzing_progress
```

or you can simply run `python main.py` and you can see the runtime output in ./log/*

## Reproduction

You can reproduce the process of how we use clozeMaster to generate a testcase to find a new bug on the rust compiler.

```sh
mkdir temp
python reproduce.py --seedfile ./reproduce/116681 --time 200 
```

- seedfile: The default folder address for storing seed file(seed.rs);and the reproduce output will be stored at `./reproduce/116681/reproduce_bug.rs`
- time: The maximum number of generations you want clozemaster to attempt
For the test cases that time out, you can use the following command to view its error message.

```sh
rustup default nightly
rustc -Z time-passes ./reproduce/116681/reproduce_bug.rs
```

After that, you should change the rustc to stable version.

```sh
rustup default 1.73
```

## Bug found by our tool

### Rust

#### rustc

[117696](https://github.com/rust-lang/rust/issues/117696)  
[117634](https://github.com/rust-lang/rust/issues/117634)  
[117443](https://github.com/rust-lang/rust/issues/117443)  
[117275](https://github.com/rust-lang/rust/issues/117275)  
[117275](https://github.com/rust-lang/rust/issues/117275)  
[117261](https://github.com/rust-lang/rust/issues/117261)  
[117257](https://github.com/rust-lang/rust/issues/117257)  
[116687](https://github.com/rust-lang/rust/issues/116687)  
[116681](https://github.com/rust-lang/rust/issues/116681)  
[116647](https://github.com/rust-lang/rust/issues/116647)  
[116624](https://github.com/rust-lang/rust/issues/116624)  
[116519](https://github.com/rust-lang/rust/issues/116519)  
[116287](https://github.com/rust-lang/rust/issues/116287)  
[115555](https://github.com/rust-lang/rust/issues/115555)  
[115435](https://github.com/rust-lang/rust/issues/115435)  
[115433](https://github.com/rust-lang/rust/issues/115433)  
[115407](https://github.com/rust-lang/rust/issues/115407)  
[115314](https://github.com/rust-lang/rust/issues/115314)  
[114464](https://github.com/rust-lang/rust/issues/114464)  
[114463](https://github.com/rust-lang/rust/issues/114463)  
[114317](https://github.com/rust-lang/rust/issues/114317)  
[118285](https://github.com/rust-lang/rust/issues/118285)  
[117657](https://github.com/rust-lang/rust/issues/117657)  
[117151](https://github.com/rust-lang/rust/issues/117151)  
[117080](https://github.com/rust-lang/rust/issues/117080)  
[116784](https://github.com/rust-lang/rust/issues/116784)  
[116783](https://github.com/rust-lang/rust/issues/116783)  
[116780](https://github.com/rust-lang/rust/issues/116780)  
[116554](https://github.com/rust-lang/rust/issues/116554)  
[115842](https://github.com/rust-lang/rust/issues/115842)  
[115599](https://github.com/rust-lang/rust/issues/115599)  
[114327](https://github.com/rust-lang/rust/issues/114327)  
[114324](https://github.com/rust-lang/rust/issues/114324)  

#### mrust

[322](https://github.com/thepowersgang/mrustc/issues/322)  
[321](https://github.com/thepowersgang/mrustc/issues/321)  
[320](https://github.com/thepowersgang/mrustc/issues/320)  
[318](https://github.com/thepowersgang/mrustc/issues/318)  
<!--
### C
#### LLVM
[87957](https://github.com/llvm/llvm-project/issues/87957)  
[89493](https://github.com/llvm/llvm-project/issues/89493)  
[90330](https://github.com/llvm/llvm-project/issues/90330)  

#### GCC
[114634](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114634)  
[114638](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114638)  
[114858](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=114858)  
[115173](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=115173)  
-->
