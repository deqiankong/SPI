#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import sys
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from tqdm.auto import tqdm
from collections import defaultdict
import nltk
import numpy as np

from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from src.data_utils.dialogue_reader import DialogReader, preprocess
from src.data_utils.data_collator import Seq2SeqDataCollatorWithLV, frange_cycle_zero_linear
from src.modules.evaluator import DialogEvaluator
from src.utils.utils import add_special_tokens
from src.models.bart_modeling import BartForConditionalGenerationWithLV, BartForConditionalGenerationWithLangvegin

logger = get_logger(__name__)

MODEL_CLASS = {
    "t5-base": None,
    "t5-large": None,
    "bart-base": BartForConditionalGenerationWithLangvegin,
    "bart-large": BartForConditionalGenerationWithLangvegin,
}

def get_model_class(model_name_or_path):
    if "t5-base" in model_name_or_path:
        key = "t5-base"
    elif "t5-large" in model_name_or_path:
        key = "t5-large"
    elif "bart-base" in model_name_or_path:
        key = "bart-base"
    elif "bart-large" in model_name_or_path:
        key = "bart-large"
    else:
        key = model_name_or_path
    return MODEL_CLASS.get(key, None)


def get_epoch_result(training_args, results, prefix="eval/"):
    metric_for_best_model = training_args.metric_for_best_model
    if metric_for_best_model is None or metric_for_best_model == "": 
        metric_for_best_model = prefix+"loss"
    greater_is_better = training_args.greater_is_better

    if metric_for_best_model in results:
        return results[metric_for_best_model] if greater_is_better else -results[metric_for_best_model]
    elif prefix+metric_for_best_model in results:
        return results[prefix+metric_for_best_model] if greater_is_better else -results[prefix+metric_for_best_model]
    else:
        raise ValueError("Metric for best model is not found!")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    early_stop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do early stopping in the traning process."}
    )
    early_stopping_patience: Optional[int] = field(
        default=1,
        metadata={"help": "`metric_for_best_model` to stop training when the specified metric worsens for "
        "`early_stopping_patience` evaluation calls."}
    )
    # learn_prior: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "Whether to train prior model or not. "
    #     "If False, only prior/posterior of the latent state will be used without KL loss."}
    # )
    # from_prior: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "Whether to use the prior latent states during generator forward. "
    #     "If False, use posterior latent states."}
    # )
    attend_latent: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether the decoder should attend to the latent variable z "
        "through cross attention. If False, the latent states will not be concatenated "
        "with encoder states."}
    )
    attend_latent_w_self: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether the decoder should attend to the latent variable z "
        "through self attention. If True, the latent states will be concatenated with "
        "K and V of the decoder seq and perform self attention."}
    )
    fuse_z: Optional[str] = field(
        default=None,
        metadata={"help": "Instead of attending to z, we fuse z with encoder hidden "
        "states directly by concatenating z and h in dim=2 and pass by a linear layer."
        "When `residual`, z and h will be added together directly. When `concate`, "
        "only fused feature is kept. When `None`, do not use this method."}
    )
    no_kn_decoding: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to input knowledge feature into decoder or not."}
    )
    sample_latent: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do reparameterization or not."}
    )
    kn_selector: Optional[str] = field(
        default="linear", metadata={"help": "The method used to select the most possible knowledge"}
    )
    target_kl: Optional[float] = field(
        default="1.0", metadata={"help": "The minimum value of KL loss that will be taken into consideration."}
    )
    latent_size: Optional[int] = field(
        default=None, metadata={"help": "The latent hidden size of the latent model."}
    )
    cls_ratio: Optional[float] = field(
        default=1.0, metadata={"help": "loss = lm_loss + cls_ratio * cls_loss"}
    )
    # Use knowledge feature / dialogue feature / the whole feature for knowledge selection and z
    use_feature: Optional[str] = field(
        default="kn",
        metadata={
            "help": "Use knowledge feature (kn) / dialogue feature (dial) / the whole feature (all) for "
            "knowledge selection and z."
        },
    )
    use_z_for_cls: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use z (instead of feature directly) for knowledge selection."}
    )
    top_k_kn: Optional[int] = field(
        default=1, metadata={"help": "We keep top-K knowledge candidates for a second step of "
        "knowledge selection"}
    )
    pseudo_confidence: Optional[float] = field(
        default=1.0, metadata={"help": ""}
    )
    pseudo_label_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to only use pseudo label for knowledge selection training or not."}
    )
    oracle: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the oracle knoweldge for training/evaluation."}
    )
    random_choice: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to randomly choose knoweldge for training/evaluation."}
    )

    # langevin dynamics
    g_l_steps: Optional[int] = field(
        default=5, metadata={"help": "Number of update steps for langevin dynamics."}
    )
    g_l_step_size: Optional[float] = field(
        default=0.3, metadata={"help": "Step size of the update for langevin dynamics."}
    )
    verbose: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to display the CE loss during langevin dynamics or not."}
    )
    remove_noise: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to remove the noise term during Langevin gradient descent."}
    )
    add_z_mse: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to minimize the MSE loss between z_g_0 and z_g_k or not."}
    )
    vae_kl_weights: Optional[float] = field(
        default="1.0", metadata={"help": ""}
    )
    categorical_prior: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use categorial prior hypothesis instead of uniform prior+initializer."}
    )

    # generate
    gen_with_noise: bool = field(
        default=False,
        metadata={
            "help": "Whether to generate response with diversity."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The path to the dataset folder."}
    )
    preproc_dir: Optional[str] = field(
        default="./data/processed_wow", metadata={"help": "The path to the pre-processed dataset folder."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    history_length: Optional[int] = field(
        default=1, metadata={"help": "The length of the history turns."},
    )
    max_knowledge: Optional[int] = field(
        default=34, metadata={"help": "The maximum number knowledge sentences for each data samples."},
    )
    pad_knowledge: Optional[bool] = field(
        default=False, metadata={"help": "Whether to pad the knowledge length to "},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    metric_name: Optional[str] = field(
        default="f1",
        metadata={
            "help": "The name of the metric for evaluation"
        },
    )
    merge_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to merge two validation sets."
        },
    )
    # use_less_samples: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "There are two mode to load the wizard of wikepedia data: load the last data sample "
    #         "in some cases or not. Thus, it causes the second mode to have less data samples. If True, "
    #         "then use mode 2, else use mode 1."
    #     },
    # )

    # Latent Variable Related
    ratio_increase: Optional[float] = field(
        default=0.25,
        metadata={"help": "We use the cyclical schedule to anneal β. ``ratio_increase'' denotes the ratio "
        "of increasing period of β in one cycle"},
    )
    ratio_zero: Optional[float] = field(
        default=0.5,
        metadata={"help": "We use the cyclical schedule to anneal β. ``ratio_zero'' denotes the ratio "
        "of constant period of β staying as 0 in one cycle"},
    )
    n_cycle: Optional[int] = field(
        default=10,
        metadata={"help": "We use the cyclical schedule to anneal β for how many cycles."},
    )
    max_kl_weight: Optional[float] = field(
        default=0.5,
        metadata={"help": "During annealing, the maximize value of β."},
    )

    # For other evaluation/training args
    eval_selection: bool = field(
        default=False,
        metadata={
            "help": "Evaluate knowledge selection task performance. Will be not considered (always True)"
            "if the generation performance is evaluated."
        },
    )
    with_tracking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable experiment trackers for logging."
        },
    )
    max_patience: Optional[int] = field(
        default=10,
        metadata={"help": "Stop training when the performance is not improved for max_patience "
                  "number of epochs."},
    )


    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length



def main():
    import datasets

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=training_args.report_to, logging_dir=training_args.output_dir) if data_args.with_tracking else Accelerator()
    )
    if data_args.source_prefix is None and training_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)
        os.environ['PYTHONHASHSEED'] = str(training_args.seed)

    # Make output dirs
    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Update config for VAE
    config.latent_size = model_args.latent_size if model_args.latent_size is not None else config.d_model
    config.sample_latent = model_args.sample_latent
    config.kn_selector   = model_args.kn_selector  
    config.target_kl     = model_args.target_kl  
    config.attend_latent = model_args.attend_latent  
    config.attend_latent_w_self = model_args.attend_latent_w_self
    config.fuse_z        = model_args.fuse_z
    config.cls_ratio     = model_args.cls_ratio
    config.no_kn_decode  = model_args.no_kn_decoding
    config.use_feature   = model_args.use_feature
    config.use_z_for_cls = model_args.use_z_for_cls
    config.oracle        = model_args.oracle
    config.random_choice = model_args.random_choice
    config.g_l_steps     = model_args.g_l_steps
    config.g_l_step_size = model_args.g_l_step_size
    config.verbose       = model_args.verbose
    config.remove_noise  = model_args.remove_noise
    config.add_z_mse     = model_args.add_z_mse
    config.gen_with_noise = model_args.gen_with_noise
    config.top_k_kn      = model_args.top_k_kn
    config.pseudo_confidence = model_args.pseudo_confidence
    config.pseudo_label_only = model_args.pseudo_label_only
    config.categorical_prior = model_args.categorical_prior
    assert not (config.fuse_z and (config.attend_latent or config.attend_latent_w_self)), "We cannot fuse z with h, while attending to z."
    assert not (config.fuse_z and config.use_z_for_cls)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True if "bart" in model_args.model_name_or_path else False,
        model_max_length=max(data_args.max_source_length, data_args.max_target_length)
    )

    ModelClass = get_model_class(model_args.model_name_or_path)
    if ModelClass is None:
        raise ValueError("The selected model type '{model_args.model_name_or_path}' is not supported!")

    model = ModelClass.from_pretrained(model_args.model_name_or_path, config=config)
    ignore_keys = model.get_ignore_keys()

    if training_args.gradient_checkpointing:
        model.config.use_cache = False
    add_special_tokens(model, tokenizer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load and tokenize data
    datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=data_args.preproc_dir)
    train_dataset, eval_datasets, test_datasets = preprocess(training_args, data_args, datasets, tokenizer, merge_eval=True if training_args.do_train or data_args.merge_eval else False)

    # Compare VAE_KL_WEIGHTS for annealing \beta
    if training_args.do_train:
        # len_dataloader = math.ceil(len(datasets["train"]) / training_args.per_device_train_batch_size)
        # num_update_steps_per_epoch = len_dataloader // training_args.gradient_accumulation_steps
        # num_iters = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
        # vae_kl_weights = frange_cycle_zero_linear(
        #     num_iters,
        #     training_args.num_train_epochs, 
        #     stop=data_args.max_kl_weight, 
        #     n_cycle=data_args.n_cycle,
        #     ratio_increase=data_args.ratio_increase,
        #     ratio_zero=data_args.ratio_zero
        # )
        vae_kl_weights = model_args.vae_kl_weights
    else:
        vae_kl_weights = None

    # Data collator
    train_data_collator = Seq2SeqDataCollatorWithLV(
        tokenizer, 
        data_args, 
        model.config.decoder_start_token_id, 
        vae_kl_weights=vae_kl_weights, 
        accumulation_steps=training_args.gradient_accumulation_steps,
        return_posterior=False,
        no_kn_decoding=model_args.no_kn_decoding,
    )
    eval_data_collator = Seq2SeqDataCollatorWithLV(
        tokenizer, 
        data_args, 
        model.config.decoder_start_token_id, 
        vae_kl_weights=1.0,
        return_posterior=False,
        no_kn_decoding=model_args.no_kn_decoding,
    )

    if training_args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=training_args.per_device_train_batch_size
        )
    else:
        train_dataloader = None
    if training_args.do_eval:
        if data_args.merge_eval:
            eval_dataloader = DataLoader(eval_datasets, collate_fn=eval_data_collator, batch_size=training_args.per_device_eval_batch_size)
            eval_unseen_dataloader = None
        else:
            eval_dataset, eval_unseen_dataset = eval_datasets
            eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_data_collator, batch_size=training_args.per_device_eval_batch_size)
            eval_unseen_dataloader = DataLoader(eval_unseen_dataset, collate_fn=eval_data_collator, batch_size=training_args.per_device_eval_batch_size)
    else:
        eval_dataloader = None
    if training_args.do_predict:
        test_dataset, test_unseen_dataset = test_datasets
        test_dataloader = DataLoader(test_dataset, collate_fn=eval_data_collator, batch_size=training_args.per_device_eval_batch_size)
        if test_unseen_dataset is not None:
            test_unseen_dataloader = DataLoader(test_unseen_dataset, collate_fn=eval_data_collator, batch_size=training_args.per_device_eval_batch_size)
        else:
            test_unseen_dataloader = None

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    if training_args.do_train:
        training_args.num_train_epochs = int(training_args.num_train_epochs)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
        if training_args.max_steps == -1:
            training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        else:
            training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = int(math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps))
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
    else:
        model = accelerator.prepare(model)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(training_args.save_strategy, "isdigit"):
        checkpointing_steps = training_args.save_strategy
        if training_args.save_strategy.isdigit():
            checkpointing_steps = int(training_args.save_strategy)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if data_args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(training_args)
            experiment_config.update(vars(model_args))
            experiment_config.update(vars(data_args))

            # TensorBoard cannot log Enums, need the raw value
            experiment_config.pop("__cached__setup_devices", None)
            for k, v in experiment_config.items():
                if v is None:
                    experiment_config[k] = ""
                elif isinstance(v, list):
                    experiment_config[k] = "".join([str(i) for i in v])
                elif type(v) not in [int, float, str, bool, torch.Tensor]:
                    experiment_config[k] = v.value

            accelerator.init_trackers("main_no_trainer", experiment_config)

    # Metric
    evaluator = DialogEvaluator(
        metric_name=data_args.metric_name,
        tokenizer=tokenizer,
        eval_selection=data_args.eval_selection,
        eval_gen=training_args.predict_with_generate,
    )
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    if training_args.do_train:
        # Train!
        total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {training_args.max_steps}")
        logger.info(
                f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(training_args.max_steps), disable=not accelerator.is_local_main_process, ncols=100)
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        if training_args.resume_from_checkpoint:
            if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
                accelerator.load_state(training_args.resume_from_checkpoint)
                path = os.path.basename(training_args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
        
    
        # Define best result to decide which checkpoint to save
        best_result, patience = -math.inf, 0
        for epoch in range(starting_epoch, training_args.num_train_epochs):
            if model_args.verbose: print(f"============ Epoch {epoch} ============")
            model.train()
            if data_args.with_tracking:
                total_loss = 0

                # losses for display
                total_lm_loss, total_cls_loss, total_kl_loss = 0, 0, 0
                batch_lm_loss, batch_cls_loss, batch_kl_loss = 0, 0, 0

            samples_seen = 0
            pseudo_labels = []
            for step, batch in enumerate(train_dataloader):
                if model_args.verbose: print(f"Batch: {step}|")
                # We need to skip steps until we reach the resumed step
                if training_args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                outputs = model(**batch)
                loss = outputs.loss

                # losses for display
                lm_loss  = outputs.lm_loss
                cls_loss = outputs.cls_loss
                kl_loss  = outputs.kl_loss

                # We keep track of the loss at each epoch
                if data_args.with_tracking:
                    total_loss += loss.detach().float()
                    total_lm_loss  += lm_loss if lm_loss is not None else 0.0
                    total_cls_loss += cls_loss if cls_loss is not None else 0.0
                    total_kl_loss  += kl_loss if kl_loss is not None else 0.0

                    # losses for display
                    batch_lm_loss  += lm_loss  / training_args.gradient_accumulation_steps if lm_loss is not None else 0.0
                    batch_cls_loss += cls_loss / training_args.gradient_accumulation_steps if cls_loss is not None else 0.0
                    batch_kl_loss  += kl_loss  / training_args.gradient_accumulation_steps if kl_loss is not None else 0.0

                loss = loss / training_args.gradient_accumulation_steps
                
                accelerator.backward(loss)
                if step % training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                    if step % (training_args.gradient_accumulation_steps*training_args.logging_steps) == 0 and step != 0:
                        logger.info(f"Step {step // training_args.gradient_accumulation_steps} | LM loss {round(batch_lm_loss, 4)} | CLS loss {round(batch_cls_loss, 4)} | KL loss {round(batch_kl_loss, 4)}")

                        if data_args.with_tracking:
                            training_states = {}
                            training_states["train_lm_loss"]  = batch_lm_loss 
                            training_states["train_cls_loss"] = batch_cls_loss
                            training_states["train_kl_loss"]  = batch_kl_loss 
                            training_states["epoch"] = epoch
                            training_states["step"] = completed_steps
                            accelerator.log(training_states, step=completed_steps)

                    batch_lm_loss, batch_cls_loss, batch_kl_loss = 0, 0, 0

                if outputs.pseudo_labels is not None:
                    batch_pseudo_labels = outputs.pseudo_labels
                    batch_pseudo_labels = accelerator.gather(batch_pseudo_labels)

                    batch_pseudo_labels = batch_pseudo_labels.cpu()
                    num_sample_per_batch = batch_pseudo_labels.shape[0]
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(train_dataloader) - 1:
                            batch_pseudo_labels = batch_pseudo_labels[: len(train_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += num_sample_per_batch
                    batch_pseudo_labels = batch_pseudo_labels.squeeze(-1).tolist()
                    pseudo_labels.extend(batch_pseudo_labels)

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if training_args.output_dir is not None:
                            output_dir = os.path.join(training_args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                

                if completed_steps >= training_args.max_steps:
                    break

            if training_args.do_eval:
                results = evaluate(training_args, data_args, tokenizer, model, accelerator, eval_dataloader, evaluator)

            if data_args.with_tracking:
                results["train_loss"] = total_loss.item() / len(train_dataloader)
                results["train_lm_loss"]  = total_lm_loss / len(train_dataloader)
                results["train_cls_loss"] = total_cls_loss / len(train_dataloader)
                results["train_kl_loss"]  = total_kl_loss / len(train_dataloader)
                results["epoch"] = epoch
                results["step"] = completed_steps
                results = rewrite_logs(results)
                accelerator.log(results, step=completed_steps)

            # Save the current states of the model, optimizer, scaler, RNG generators, and registered objects.
            # Also save the tokenizer here
            if training_args.save_strategy == "epoch":
                output_dir = f"epoch_{epoch}"
                if training_args.output_dir is not None:
                    output_dir = os.path.join(training_args.output_dir, output_dir)
                accelerator.save_state(output_dir)
                if accelerator.is_main_process:
                    config.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    with open(os.path.join(output_dir, "pseudo_labels.json"), "w") as f:
                        json.dump(pseudo_labels, f)
            
            if training_args.do_eval:
                epoch_result = get_epoch_result(training_args, results, prefix="eval/")
                if epoch_result > best_result:
                    best_result = epoch_result
                    patience = 0
                else:
                    patience += 1
                if patience > data_args.max_patience:
                    logger.info("Out of patience!")
                    accelerator.end_training()
                    break

        if patience <= data_args.max_patience:
            accelerator.end_training()

    
    if training_args.do_eval:
        eval_dataloader = accelerator.prepare_data_loader(eval_dataloader)
        eval_results = evaluate(training_args, data_args, tokenizer, model, accelerator, eval_dataloader, evaluator)
        if eval_unseen_dataloader is not None:
            eval_unseen_dataloader = accelerator.prepare_data_loader(eval_unseen_dataloader)
            eval_unseen_results = evaluate(training_args, data_args, tokenizer, model, accelerator, eval_unseen_dataloader, evaluator, prefix="eval_unseen_")
            eval_results.update(eval_unseen_results)

        with open(os.path.join(training_args.output_dir, f"eval_results.json"), "w") as f:
            json.dump(eval_results, f)
    
    if training_args.do_predict:
        test_dataloader = accelerator.prepare_data_loader(test_dataloader)
        test_results = evaluate(training_args, data_args, tokenizer, model, accelerator, test_dataloader, evaluator, prefix="test_")
        if test_unseen_dataloader is not None:
            test_unseen_dataloader = accelerator.prepare_data_loader(test_unseen_dataloader)
            test_unseen_results = evaluate(training_args, data_args, tokenizer, model, accelerator, test_unseen_dataloader, evaluator, prefix="test_unseen_")
            test_results.update(test_unseen_results)

        with open(os.path.join(training_args.output_dir, f"test_results.json"), "w") as f:
            json.dump(test_results, f)


def rewrite_logs(d):
    new_d = {}
    eval_unseen_prefix = "eval_unseen_"
    eval_unseen_prefix_len = len(eval_unseen_prefix)
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_unseen_prefix = "test_unseen_"
    test_unseen_prefix_len = len(test_unseen_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_unseen_prefix):
            new_d["eval_unseen/" + k[eval_unseen_prefix_len:]] = v
        elif k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_unseen_prefix):
            new_d["test_unseen/" + k[test_unseen_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


def evaluate(training_args, data_args, tokenizer, model, accelerator, eval_dataloader, evaluator, prefix="eval_", epoch=None):
    if "test" in prefix:
        logger.info("***** Running Prediction *****")
    else:
        logger.info("***** Running Evaluation *****")
    logger.info(f"  Num batches = {len(eval_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_eval_batch_size}")

    model.eval()
    if training_args.predict_with_generate:
        generations, golds = [], []

        if data_args.val_max_target_length is None:
            data_args.val_max_target_length = data_args.max_target_length

        gen_kwargs = {
            "max_length": data_args.val_max_target_length if data_args is not None else config.max_length,
            "num_beams": data_args.num_beams,
            "early_stopping": True,
        }
        decoder_start_token_id = model.config.decoder_start_token_id
        bos_token_id = model.config.bos_token_id

        samples_seen = 0
        eval_metric = defaultdict(float)
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Generate responses", ncols=100):
            with torch.no_grad():
                batch_size = batch["decoder_shapes"][0]
                decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=batch["input_ids"].device) * decoder_start_token_id
                batch["input_ids"] = batch["input_ids"].view(batch["decoder_shapes"][0], batch["decoder_shapes"][1], batch["decoder_shapes"][2])
                batch["attention_mask"] = batch["attention_mask"].view(batch["decoder_shapes"][0], batch["decoder_shapes"][1], batch["decoder_shapes"][2])
                batch["knowledge_mask"] = batch["knowledge_mask"].view(batch["decoder_shapes"][0], batch["decoder_shapes"][1], batch["decoder_shapes"][2])
                generated_tokens = model.generate( #accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                    decoder_knowledge_mask=batch["knowledge_mask"],
                    decoder_cxt_input_ids=batch["cxt_input_ids"] if "cxt_input_ids" in batch else None,
                    decoder_cxt_attention_mask=batch["cxt_attention_mask"] if "cxt_attention_mask" in batch else None,
                    decoder_shapes=batch["decoder_shapes"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                if not data_args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if data_args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)

                generations.extend(decoded_preds)
                golds.extend(decoded_labels)

        eval_metric = evaluator.compute(generations, golds, prefix=prefix)
        # save generations and labels
        with open(os.path.join(training_args.output_dir, prefix+"generations.txt"), "w") as f:
            f.write("\n".join([text.strip() for text in generations])+"\n")
        with open(os.path.join(training_args.output_dir, prefix+"golds.txt"), "w") as f:
            f.write("\n".join([text.strip() for text in golds])+"\n")

    else:
        eval_metric = {}


    samples_seen = 0
    metrics = defaultdict(float)
    num_samples = 0
    cache_preds = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Evaluate knowledge selection and PPL", ncols=100):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.classification_logits
        predictions, references = accelerator.gather((predictions, batch["classification_labels"]))

        predictions = predictions.cpu()
        references  = references.cpu()
        num_sample_per_batch = references.shape[0]
        num_samples += num_sample_per_batch
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += num_sample_per_batch
        
        batch_metrics = evaluator.compute_cls(predictions.numpy(), references.numpy(), prefix=prefix)
        batch_metrics[prefix+"loss"]  = outputs.loss.item()
        batch_metrics[prefix+"lm_loss"]  = outputs.lm_loss
        batch_metrics[prefix+"cls_loss"] = outputs.cls_loss
        batch_metrics[prefix+"kl_loss"]  = outputs.kl_loss
        metrics = update_metrics(metrics, batch_metrics, num_sample_per_batch=num_sample_per_batch)
    
        preds = np.argmax(predictions.numpy(), axis=1).tolist()
        cache_preds.extend(preds)
    
    with open(os.path.join(training_args.output_dir, prefix+"kn_predictions.json"), "w") as f:
        json.dump(cache_preds, f)

    metrics = update_metrics(metrics, div=num_samples)
    if "ppl" in data_args.metric_name:
        metrics[prefix+"ppl"] = round(math.exp(metrics[prefix+"lm_loss"]), 2)
    eval_metric.update(metrics)
    if epoch is not None:
        logger.info(f"epoch {epoch}: {eval_metric}")
    else:
        for k, v in eval_metric.items():
            logger.info(f"{k} = {v}")


    return eval_metric


def update_metrics(metrics, batch_metrics=None, num_sample_per_batch=1, div=1):
    if batch_metrics is None:
        for k, v in metrics.items():
            metrics[k] = round(metrics[k] / div, 2) if metrics[k] is not None else 0.0
    else:
        for k, v in batch_metrics.items():
            metrics[k] += v * num_sample_per_batch if v is not None else 0.0
    return metrics

if __name__ == "__main__":
    main()