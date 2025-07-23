'''

python nemo_salm_chunked.py \
    pretrained_name=/home/ext_ponceponte_oscar_mayo_edu/speech_ker/asr/models_weights/canary-qwen/model.safetensors \
    inputs=asr_manifest.json \
    output_manifest=generations.json \
    batch_size=1   user_prompt="Transcribe the following:"
'''



# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet, fastcopy
from lhotse.dataset import IterableDatasetWrapper
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from transformers import GenerationConfig

from nemo.collections.common.data.lhotse import NeMoMultimodalConversation
from nemo.collections.common.data.lhotse.cutset import cut_to_conversation, guess_parse_cutset
from nemo.collections.common.data.lhotse.dataloader import tokenize_with_prompt
from nemo.collections.common.data.lhotse.text_adapters import TextTurn
from nemo.collections.speechlm2 import SALM, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils import logging


@dataclass
class SalmEvalConfig:
    pretrained_name: str
    inputs: str
    batch_size: int = 64
    max_new_tokens: int = 128
    output_manifest: str = "generations.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    extra_eos_tokens: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    # ─────────────────────────────────────────────────────────────
    # NEW: chunk size in seconds for non-overlapping inference
    chunk_size: float = 10.0
         


def chunk_cut_nonoverlapping(cut, chunk_size: float):
    """
    Given a single Cut, split it into floor(duration/chunk_size) non-overlapping
    cuts of length chunk_size. Drop any remainder shorter than chunk_size.
    """
    num_chunks = int(cut.duration // chunk_size)
    for i in range(num_chunks):
        yield cut.truncate(
            offset=i * chunk_size,
            duration=chunk_size,
            keep_excessive_supervisions=False,
        )


@hydra_runner(config_name="SalmEvalConfig", schema=SalmEvalConfig)
def main(cfg: SalmEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # 1) Load and prepare the model
    model = (
        SALM.from_pretrained(cfg.pretrained_name)
        .eval()
        .to(getattr(torch, cfg.dtype))
        .to(cfg.device)
    )

    # 2) Parse the input CutSet
    raw_cuts = guess_parse_cutset(cfg.inputs)

    # 3) Chunk into non-overlapping fixed-size segments
    chunked_cuts = raw_cuts.flat_map(
        lambda cut: list(chunk_cut_nonoverlapping(cut, cfg.chunk_size)),
        apply_fn=None,
    )

    # 4) Convert each chunk to a conversation and tokenize
    conversations = (
        chunked_cuts
        .map(
            partial(
                cut_to_conversation,
                audio_locator_tag=model.audio_locator_tag,
                token_equivalent_duration=model.token_equivalent_duration,
            )
        )
        .map(
            partial(
                attach_system_and_user_turns,
                system_prompt=cfg.system_prompt,
                user_prompt=cfg.user_prompt,
            ),
            apply_fn=None,
        )
        .map(strip_response_if_any, apply_fn=None)
        .map(
            partial(
                tokenize_with_prompt,
                tokenizer=model.tokenizer,
                prompt_format=model.cfg.prompt_format,
            ),
            apply_fn=None,
        )
    )

    # 5) Sort by length, batch, and wrap in SALMDataset
    conversations = sort_by_length(conversations)
    dloader = torch.utils.data.DataLoader(
        dataset=IterableDatasetWrapper(
            dataset=SALMDataset(model.tokenizer),
            sampler=lhotse.dataset.DynamicCutSampler(conversations, max_cuts=cfg.batch_size),
        ),
        num_workers=1,
        batch_size=None,
    )

    # 6) Prepare EOS tokens
    eos_tokens = [model.text_eos_id]
    if cfg.extra_eos_tokens is not None:
        for t in cfg.extra_eos_tokens:
            tid = model.tokenizer.token_to_id(t)
            assert tid is not None, f"Token '{t}' is not in the model's vocabulary."
            eos_tokens.append(tid)

    writer = SequentialJsonlWriter(cfg.output_manifest)

    # 7) Inference loop
    num_answer_tokens = []
    infer_durations = []
    for batch_idx, batch in enumerate(dloader):
        ts = perf_counter()
        answer_ids = model.generate(
            prompts=batch["input_ids"].to(model.device, non_blocking=True),
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            generation_config=GenerationConfig(
                max_new_tokens=cfg.max_new_tokens,
                bos_token_id=model.text_bos_id,
                eos_token_id=eos_tokens,
                pad_token_id=model.text_pad_id,
            ),
        )
        answer_ids = answer_ids.cpu()
        batch_infer_duration = perf_counter() - ts

        batch_contexts = [model.tokenizer.ids_to_text(example) for example in batch["input_ids"]]
        answer_ids = [parse_hyp(ans, eos_tokens) for ans in answer_ids]
        batch_num_answer_tokens = [len(ans) for ans in answer_ids]
        batch_answers = [model.tokenizer.ids_to_text(ans) for ans in answer_ids]
        for conv, ctx, ans in zip(batch["conversations"], batch_contexts, batch_answers):
            conv.turns.append(TextTurn(role="assistant", value=ans))
            writer.write(conv.to_dict())

        num_answer_tokens.extend(batch_num_answer_tokens)
        infer_durations.append(batch_infer_duration)
        if cfg.verbose:
            tps = sum(batch_num_answer_tokens) / batch_infer_duration
            logging.info(f"Batch {batch_idx}: TPS={tps:.2f}")

    # 8) Final throughput logging
    rtfx = sum(num_answer_tokens) / sum(infer_durations)
    logging.info(f"Overall TPS: {rtfx:.2f}")


def attach_system_and_user_turns(
    conversation: NeMoMultimodalConversation,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
) -> NeMoMultimodalConversation:
    if system_prompt is None and user_prompt is None:
        return conversation
    turns = conversation.turns
    if user_prompt is not None and not any(isinstance(t, TextTurn) and t.role == "user" for t in turns):
        turns = [TextTurn(role="user", value=user_prompt)] + turns
    if system_prompt is not None and not any(t.role == "system" for t in turns):
        turns = [TextTurn(role="system", value=system_prompt)] + turns
    return fastcopy(conversation, turns=turns)


def strip_response_if_any(
    conversation: NeMoMultimodalConversation,
) -> NeMoMultimodalConversation:
    turns = conversation.turns
    # strip trailing assistant turns
    while turns and turns[-1].role == "assistant":
        turns = turns[:-1]
    return fastcopy(conversation, turns=turns)


def sort_by_length(conversations: CutSet) -> CutSet:
    return CutSet(sorted(conversations, key=lambda c: c.total_length, reverse=True))


def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = torch.isin(answer, torch.tensor(eos_tokens)).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    return answer[: end[0]]


if __name__ == '__main__':
    main()


