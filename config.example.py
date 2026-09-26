# Copy to config.py and edit. config.py is gitignored.
# The OpenAI key is not kept here: it is read from the OPENAI_API_KEY environment variable.

RUN_NAME = "run2"
CACHE_FROM_RUN = "run1"  # judge cache and judge validation copied from this finished run

MODEL_NAME = "Qwen/Qwen2.5-3B"  # SEAL's scaled-down model (paper B.7), reference results in the repo
MODEL_CLASS = "AutoModelForCausalLM"
DEVICE = "cuda"

# judge frozen before the main run; cache keys include the model name
GRADER_MODEL = "gpt-5.6-luna"
GRADER_REASONING_EFFORT = "none"
GRADER_WORKERS = 16
GRADER_PRICE_IN = 0.20  # USD per 1M input tokens
GRADER_PRICE_OUT = 1.20  # USD per 1M output tokens
CROSS_GRADER_MODEL = "gpt-4.1-2025-04-14"  # SEAL's judge, agreement check only
CROSS_GRADER_SAMPLE = 400
SELF_CONSISTENCY_SAMPLE = 300

# scale
OUTER_ITERATIONS = 3
OUTER_PASSAGES = 40  # B, disjoint across iterations
QA_CANDIDATES = 6  # M
SELF_EDITS = 5  # K, SEAL B.2
TTT_SEEDS = 3  # SEAL B.2 averages each self-edit over 3 seeds
SE_RL_ROUNDS = 2  # SEAL B.2
SE_RL_PASSAGES = 50  # SEAL B.2
SE_RL_RESERVE = 50  # replacements for passages whose question set falls short
DEV_PASSAGES = 30
VAL_PASSAGES = 200  # SEAL B.4
VAL_SELF_EDITS = 3  # fresh self-edits per passage, one TTT each

# generation
GEN_TEMPERATURE = 1.0  # SEAL B.2
GEN_TOP_P = 0.95
GEN_BATCH = 16
SELF_EDIT_MAX_TOKENS = 1024
QA_GEN_MAX_TOKENS = 512
QA_GEN_MAX_PAIRS = 10
QA_QUESTIONS = 15  # reward questions per passage, same count for every passage
QA_GEN_SAMPLES = 8  # completions of the 5-question prompt pooled per passage
ANSWER_MAX_TOKENS = 64

# inner loop LoRA (SEAL code defaults)
TTT_LORA_RANK = 32
TTT_LORA_ALPHA = 64
TTT_TARGET_MODULES = ("q_proj", "v_proj")
TTT_EPOCHS = 10
TTT_LR = 1e-3
TTT_MAX_SEQ_LEN = 2048
SEAL_PAD_QUIRK = False  # SEAL pads to max length with eos and trains on it; dev ablation only

# ReST-EM supervised finetuning (SEAL B.2)
SFT_LORA_RANK = 64
SFT_LORA_ALPHA = 128
SFT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SFT_EPOCHS = 2
SFT_LR = 3e-4
SFT_BATCH = 10

# self-edits for the external-generator baseline
GPT_SELF_EDIT_MODEL = "gpt-4.1-2025-04-14"

SQUAD_TRAIN = "general-knowledge/data/squad_train.json"
SQUAD_VAL = "general-knowledge/data/squad_val.json"
RESULTS_ROOT = "general-knowledge/results/qagen"
ADAPTER_ROOT = "models/qagen"
