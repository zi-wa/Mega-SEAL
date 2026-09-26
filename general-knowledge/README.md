# SEAL - general-knowledge

This is an implementation of SEAL for the *general knowledge incorporation* setting, where the goal is to update or integrate new information from a passage into weights.

## Usage

The python files in src/ have documentation on function. Here is some information on how to run the pipelines used in the paper's experiments.

### 1. Create Data
Use `make_squad_data.sh` (or `make_squad_data_openai.sh`) to create the synthetic data used in subsequent RL training or evaluation.

```bash
sbatch general-knowledge/scripts/make_squad_data.sh
```

### 2. TTT server
Run the `TTT_server`. This sets up a [ZMQ](https://zeromq.org/) port that takes input parameters like training data and corresponding questions, and then runs rounds of training a temporary lora adapter and evaluating on the questions. This is then called for both RL training rewards and evaluation.

```bash
sbatch general-knowledge/scripts/TTT_server.sh
```

### 3. Query server
To query the server, run either `query_server` or `CPT` for either the single-passage or multi-passage setting respectively. This can be set to run on training documents for a round of ReST-EM RL training, or on validation documents for evaluation. 

```bash
sbatch general-knowledge/scripts/query_server.sh
```

### 4. RL Training
To run a round of ReST-EM, after running `query_server` on training documents, build the SFT dataset (more documentation in the python file):

```bash
python3 general-knowledge/src/EM/build_SFT_dataset.py <path/to/result/of/run.json>
```

Then, run the training script on this dataset:

```bash
sbatch general-knowledge/scripts/train_SFT.sh
```

### 5. Continual Self-Edits
To run the continual self-edits experiment (Section 5):

```bash
sbatch general-knowledge/scripts/continual_self_edits.sh
```

## Self-generated evaluation questions (qagen)

Single-GPU Windows port of the knowledge-incorporation loop (transformers + PEFT, no vLLM) in
which the model writes its own evaluation questions for a passage and uses them as the ReST-EM
reward for its self-edits. Code in `src/qagen/`.

```bat
setup_win.bat   REM venv, torch cu126, requirements-win.txt, config.py from config.example.py
run_all.bat     REM resumable; results in results/qagen/<RUN_NAME>/, adapters in models/qagen/<RUN_NAME>/
```

`OPENAI_API_KEY` is read from the environment. Compared with the SEAL code: Qwen2.5-3B, judge
gpt-5.6-luna, self-edits capped at 1024 tokens, no EOS padding of TTT sequences, 3 self-edits per
validation passage, SFT batch 10 by gradient accumulation.
