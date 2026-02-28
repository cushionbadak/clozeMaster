# ClozeMaster-demo
Artifacts for "ClozeMaster: Fuzzing Rust Compiler by Harnessing LLMs for Infilling Masked Real Programs".

## fix/oom-kill branch

This branch fixes the OOM (Out-Of-Memory) kill issue that caused the fuzzing process to crash after running for many hours:

- **Zombie rustc processes**: Previously, timed-out `rustc` compilations were not properly killed (`shell=True` only terminated the intermediate shell, leaving `rustc` alive). Over time, hundreds of zombie processes accumulated and exhausted system memory, causing the OS to kill the python process. Now uses direct process execution with process group kill (`SIGKILL`) to ensure full cleanup.
- **CUDA OOM**: Previously, a single GPU out-of-memory error during LLM inference would crash the entire run. Now catches `torch.cuda.OutOfMemoryError`, clears the GPU cache, and skips to the next input.
- **temp/ cleanup**: Compiled binaries in `temp/` were never cleaned up, accumulating over time. Now automatically removed after each source file is fully processed.
- **Result archiving**: `archive_results.sh` packages `log/` and `target_dataset/` into a timestamped `.tar.gz` (extracts into a single folder).
- **Resumable runs**: `run_resume.sh` archives previous logs and invokes `main.py --resume` so only unprocessed files are sent through LLM inference.
- **Elapsed time**: `elapsed.sh` shows how long the current (or finished) run has been going by reading log timestamps.

### Resuming an interrupted run

If the process was interrupted (OOM kill, Ctrl-C, etc.), use the resume script to continue from where it left off:

```sh
bash run_resume.sh                                   # resume with defaults
bash run_resume.sh --rs_files ./dataset/history_codes # pass extra args
```

`run_resume.sh` does three things before calling `main.py --resume`:
1. Moves `log/demo.log` to `log/demo_<timestamp>.log` (since main.py overwrites the log)
2. Copies `log/bug.csv` to `log/bug_<timestamp>.csv` (snapshot backup; the original keeps accumulating)
3. Invokes `python main.py --resume`, which skips source files whose first variant (`<name>_1.rs`) already exists in `target_dataset/`

You can also use the flag directly, but note that `main.py` always overwrites `demo.log` on startup (`filemode="w"`), so **use `run_resume.sh` to preserve previous logs**:
```sh
# Recommended: archives logs before running
bash run_resume.sh

# Without log archiving (demo.log will be overwritten, bug.csv is safe):
python main.py --resume
```

### Checking elapsed time

While `main.py` is running (or after it finishes), check how long it has been going:

```sh
bash elapsed.sh                              # defaults to ./log/demo.log
bash elapsed.sh ./log/demo_20260301_120000.log  # check an archived log
```

### Quick Setup (Ubuntu)

We provide a setup script that installs all dependencies on a fresh Ubuntu machine:
```sh
git clone -b fix/oom-kill https://github.com/cushionbadak/clozeMaster.git
cd clozeMaster
bash setup_ubuntu.sh
```
To specify a Rust nightly toolchain version:
```sh
RUST_TOOLCHAIN=nightly-2025-09-02 bash setup_ubuntu.sh
```

The script handles:
- System packages (git, curl, build-essential)
- Rust compiler via rustup (configurable nightly toolchain)
- Miniconda + Python 3.8 environment (`py38`)
- PyTorch (CUDA 11.8), tokenizers, pandas, and other Python dependencies
- [Incoder-1B](https://huggingface.co/facebook/incoder-1B) model weights download

After setup, place your `.rs` test files in `dataset/history_codes/` (nested directories are supported) and run:
```sh
conda activate py38
python main.py
```

---
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



