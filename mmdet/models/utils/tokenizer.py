# Copyright (c) OpenMMLab. All rights reserved.
import collections
import os

from mmengine.fileio import list_from_file
from transformers import (AutoTokenizer, BartTokenizer, BasicTokenizer,
                          BertTokenizer, BertTokenizerFast, LlamaTokenizer,
                          WordpieceTokenizer)

from mmdet.registry import TOKENIZER
from .huggingface import register_hf_tokenizer

@register_hf_tokenizer()
class BlipTokenizer(BertTokenizerFast):
    """"BlipTokenizer inherit BertTokenizerFast (fast, Rust-based)."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *init_inputs,
        **kwargs,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            **kwargs,
        )
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        return tokenizer
