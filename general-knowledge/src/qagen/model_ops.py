"""Policy weights: seeded sampling, inner-loop LoRA adaptation, ReST-EM merges, rollback."""
import contextlib
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

import config

MERGED_WEIGHT_SUFFIXES = tuple(f".{module}.weight" for module in config.SFT_TARGET_MODULES)

Example = Tuple[Optional[str], str]  # (prompt or None, target); None keeps loss on the whole sequence


class Policy:
    """The model plus every weight operation the two loops need."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ttt_count = 0
        self.ttt_seconds = 0.0

    @classmethod
    def load(cls, model_name: str = config.MODEL_NAME, device: str = config.DEVICE) -> "Policy":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = "left"  # batched generation; training runs one sequence at a time
        model, loading_info = getattr(transformers, config.MODEL_CLASS).from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            output_loading_info=True,
        )
        # a text-only class over a multimodal checkpoint leaves vision/audio keys unused, which is
        # fine, but a missing or reshaped text key would mean silently random weights
        tied = set(getattr(model, "_tied_weights_keys", None) or ())
        broken = [key for key in loading_info["missing_keys"] if key not in tied]
        broken += list(loading_info["mismatched_keys"])
        if broken:
            raise RuntimeError(f"checkpoint keys missing or mismatched: {broken[:10]}")
        model.to(device)
        model.eval()
        return cls(model, tokenizer, device)

    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        seed: int,
        greedy: bool = False,
    ) -> List[str]:
        completions: List[str] = []
        for start in range(0, len(prompts), config.GEN_BATCH):
            chunk = list(prompts[start : start + config.GEN_BATCH])
            batch = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            torch.manual_seed(seed + start)  # chunk position keeps repeat runs identical
            generated = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy,
                temperature=None if greedy else config.GEN_TEMPERATURE,
                top_p=None if greedy else config.GEN_TOP_P,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = generated[:, batch["input_ids"].shape[1] :]
            completions.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return completions

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    @contextlib.contextmanager
    def adapted(self, train_sequences: Sequence[str], seed: int):
        """SEAL's inner loop: temporary LoRA on the self-edit, dropped again afterwards."""
        started = time.time()
        adapter = self._train_lora(
            [(None, sequence) for sequence in train_sequences],
            rank=config.TTT_LORA_RANK,
            alpha=config.TTT_LORA_ALPHA,
            targets=config.TTT_TARGET_MODULES,
            epochs=config.TTT_EPOCHS,
            learning_rate=config.TTT_LR,
            batch_size=1,
            seed=seed,
            pad_to_max=config.SEAL_PAD_QUIRK,
        )
        self.ttt_count += 1
        self.ttt_seconds += time.time() - started
        try:
            yield
        finally:
            adapter.unload()

    def finetune_and_merge(
        self,
        pairs: Sequence[Tuple[str, str]],
        seed: int,
        adapter_dir: Optional[str] = None,
    ) -> None:
        """ReST-EM update: supervised finetuning on the kept generations, merged into the weights."""
        adapter = self._train_lora(
            list(pairs),
            rank=config.SFT_LORA_RANK,
            alpha=config.SFT_LORA_ALPHA,
            targets=config.SFT_TARGET_MODULES,
            epochs=config.SFT_EPOCHS,
            learning_rate=config.SFT_LR,
            batch_size=config.SFT_BATCH,
            seed=seed,
        )
        if adapter_dir:
            adapter.save_pretrained(adapter_dir)
        self.model = adapter.merge_and_unload()
        self.model.eval()

    def checkpoint(self) -> Dict[str, torch.Tensor]:
        """Copies of the linears a merge can touch; unmerging instead would drift in bf16."""
        return {
            name: weight.detach().clone()
            for name, weight in self.model.named_parameters()
            if name.endswith(MERGED_WEIGHT_SUFFIXES)
        }

    @torch.no_grad()
    def restore(self, snapshot: Dict[str, torch.Tensor]) -> None:
        for name, weight in self.model.named_parameters():
            if name in snapshot:
                weight.copy_(snapshot[name])

    def apply_adapters(self, adapter_dirs: Iterable[str]) -> None:
        """Rebuild a later policy from saved outer-loop adapters, merged in order."""
        for adapter_dir in adapter_dirs:
            loaded = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model = loaded.merge_and_unload()
        self.model.eval()

    def _train_lora(
        self,
        examples: Sequence[Example],
        *,
        rank: int,
        alpha: int,
        targets: Sequence[str],
        epochs: int,
        learning_rate: float,
        batch_size: int,
        seed: int,
        pad_to_max: bool = False,
    ):
        adapter = get_peft_model(
            self.model,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.0,
                target_modules=list(targets),
                task_type="CAUSAL_LM",
            ),
        )
        adapter.train()
        rows = [self._encode(prompt, target, pad_to_max) for prompt, target in examples]
        trainable = [weight for weight in adapter.parameters() if weight.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
        total_steps = epochs * math.ceil(len(rows) / batch_size)
        decay = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1.0 - step / total_steps)
        )
        shuffle = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            order = torch.randperm(len(rows), generator=shuffle).tolist()
            for position, row_index in enumerate(order):
                input_ids, labels = rows[row_index]
                loss = adapter(input_ids=input_ids, labels=labels, use_cache=False).loss
                (loss / batch_size).backward()
                if (position + 1) % batch_size == 0 or position + 1 == len(rows):
                    optimizer.step()
                    decay.step()
                    optimizer.zero_grad(set_to_none=True)
        adapter.eval()
        return adapter

    def _encode(self, prompt: Optional[str], target: str, pad_to_max: bool):
        end_of_text = self.tokenizer.eos_token_id
        if prompt is None:
            token_ids = self.tokenizer(target).input_ids + [end_of_text]
            labels = list(token_ids)
        else:
            prompt_ids = self.tokenizer(prompt).input_ids
            target_ids = self.tokenizer(target, add_special_tokens=False).input_ids + [end_of_text]
            token_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
        if pad_to_max and len(token_ids) < config.TTT_MAX_SEQ_LEN:
            filler = config.TTT_MAX_SEQ_LEN - len(token_ids)
            token_ids = token_ids + [end_of_text] * filler
            labels = labels + [end_of_text] * filler  # SEAL trains on this padding
        token_ids = token_ids[: config.TTT_MAX_SEQ_LEN]
        labels = labels[: config.TTT_MAX_SEQ_LEN]
        return (
            torch.tensor([token_ids], device=self.device),
            torch.tensor([labels], device=self.device),
        )
