import os
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import defaultdict

import transformers
from transformers import (
    BartTokenizer,
    BartTokenizerFast,
)
from transformers.utils import logging

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import concatenate_datasets, load_from_disk

from src.data_utils.wow import load_wow_data

SPEAKERS = ['<speaker1>', '<speaker2>']
logger = logging.get_logger(__name__)

class DialogDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class DialogReader():
    def __init__(self, training_args, data_args, tokenizer):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.pad_token_id = tokenizer.pad_token_id

        self.max_source_length = self.data_args.max_source_length
        self.max_target_length = self.data_args.max_target_length
        self.pad_to_max_length = self.data_args.pad_to_max_length
        self.max_knowledge     = self.data_args.max_knowledge

        self.cache_dir  = self.data_args.preproc_dir
        self.load_cache = not data_args.overwrite_cache

        self.threads = data_args.preprocessing_num_workers if hasattr(data_args, 'preprocessing_num_workers') else 1
        self.pad_knowledge = data_args.pad_knowledge if hasattr(data_args, 'pad_knowledge') else False
        
    def load_data(self):
        if self.data_args.dataset_name == "wow":
            self.datasets = load_wow_data(self.training_args, self.data_args)
        elif self.data_args.dataset_name == "cmu_dog":
            self.datasets = load_cmu_data(self.training_args, self.data_args)
        else:
            raise NotImplementedError
        
        return self.datasets
    
    def _extract_cache_path(self, split, merge_eval=False):
        if split != "valid" and split != "eval":
            return os.path.join(self.cache_dir, f"cache_{self.data_args.dataset_config_name}_{split}_sl{self.max_source_length}_tl{self.max_target_length}_kn{self.max_knowledge}_pad{self.pad_to_max_length}")
        else:
            return os.path.join(self.cache_dir, f"cache_{self.data_args.dataset_config_name}_{split}_sl{self.max_source_length}_tl{self.max_target_length}_kn{self.max_knowledge}_pad{self.pad_to_max_length}_merge{merge_eval}")

    def tokenize_datasets(self, merge_eval=False):
        if self.training_args.do_train:
            logger.info("Prepare tokenized training set.")
            start_time = time.time()
            cache_path = self._extract_cache_path("train")
            if os.path.exists(cache_path) and self.load_cache:
                train_dataset = torch.load(cache_path)
            else:
                train_dataset = self._get_tokenized_dataset(self.datasets["train"])
                torch.save(train_dataset, cache_path)
            logger.info(f"Preparing training set takes {time.time()-start_time} seconds.")
        else:
            train_dataset = None
        
        if self.training_args.do_eval:
            logger.info("Prepare tokenized validation dataset.")
            cache_path = self._extract_cache_path("valid", merge_eval=merge_eval)
            if os.path.exists(cache_path) and self.load_cache:
                valid_datasets = torch.load(cache_path)
            else:
                self.max_knowledge = -1
                if merge_eval:
                    valid_datasets = self._get_tokenized_dataset(self.datasets["valid"][0] + self.datasets["valid"][1])
                else:
                    valid_dataset = self._get_tokenized_dataset(self.datasets["valid"][0])
                    valid_unseen_dataset = self._get_tokenized_dataset(self.datasets["valid"][1])
                    valid_datasets = (valid_dataset, valid_unseen_dataset)
                
                torch.save(valid_datasets, cache_path)
        else:
            valid_dataset = None
            valid_unseen_dataset = None
            valid_datasets = (valid_dataset, valid_unseen_dataset)
        
        if self.training_args.do_predict:
            logger.info("Prepare tokenized test dataset.")
            cache_path = self._extract_cache_path("test")
            if os.path.exists(cache_path) and self.load_cache:
                test_datasets = torch.load(cache_path)
            else:
                self.max_knowledge = -1
                test_dataset = self._get_tokenized_dataset(self.datasets["test"][0])
                test_unseen_dataset = self._get_tokenized_dataset(self.datasets["test"][1])
                test_datasets = (test_dataset, test_unseen_dataset)

                torch.save(test_datasets, cache_path)
        else:
            test_dataset = None
            test_unseen_dataset = None
            test_datasets = (test_dataset, test_unseen_dataset)
        
        return train_dataset, valid_datasets, test_datasets
    
    def _prepare_tokenized_sample(self, history, knowledges, kn_label, kn_text, response, speaker_idx):
        task = self.data_args.dataset_config_name
        if task == "default":
            """Basic baseline setting for KGD: dialogue history+gold kn"""
            input_seq = ""
            for utter in history:
                input_seq += utter + SPEAKERS[speaker_idx]
                speaker_idx = (speaker_idx + 1) % 2
            input_seq += kn_text
            label = response

            input_ids, attention_mask = self._tokenize_lines(self.tokenizer, input_seq, return_tensor=True)
            labels = self._tokenize_lines(self.tokenizer, label, is_label=True, return_tensor=True)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        elif task == "nokn":
            """Basic baseline setting for KGD: dialogue history without kn"""
            input_seq = ""
            for utter in history:
                input_seq += utter + SPEAKERS[speaker_idx]
                speaker_idx = (speaker_idx + 1) % 2
            label = response

            input_ids, attention_mask = self._tokenize_lines(self.tokenizer, input_seq, return_tensor=True)
            labels = self._tokenize_lines(self.tokenizer, label, is_label=True, return_tensor=True)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        elif task == "random":
            # TODO randomly shuffle the knowledge and truncate when it reaches the maximum length
            raise NotImplementedError
        elif task == "fid":
            """Tokenize data sample for methods based on/similar to Fusion-in-Decoder"""
            input_seq = ""
            for utter in history:
                input_seq += utter + SPEAKERS[speaker_idx]
                speaker_idx = (speaker_idx + 1) % 2
            input_seqs = self._append_knowledge(input_seq, knowledges)
            label = response

            input_ids, attention_mask = self._tokenize_lines(self.tokenizer, input_seqs, return_tensor=True)
            labels = self._tokenize_lines(self.tokenizer, label, is_label=True, return_tensor=True)

            # if self.max_knowledge > 0:
            #     input_ids, attention_mask = self._pad_knowledge(input_ids, attention_mask)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        else:
            raise NotImplementedError

    def _get_tokenized_dataset(self, samples):
        # samples = samples[:100]
        tokenized_samples = self._tokenize_samples(samples)
        return DialogDataset(tokenized_samples)
    

    def _tokenize_samples(self, samples):
        tokenized_samples = []

        for sample in tqdm(samples, total=len(samples), desc="Tokenize data samples"):
            history = sample["history"]
            knowledges = sample["knowledges"]
            kn_label = sample["knowledge_label"]
            kn_text = sample["knowledge_text"]
            response = sample["response"]

            history_len = len(history)
            speaker_idx = 0 if history_len % 2 == 1 else 1

            tokenized_sample = self._prepare_tokenized_sample(
                history, 
                knowledges, 
                kn_label, 
                kn_text, 
                response, 
                speaker_idx
            )
            tokenized_samples.append(tokenized_sample)

        return tokenized_samples

    
    def _tokenize_lines(self, tokenizer, line, joint_line=None, is_label=False, return_tensor=False):
        inputs = tokenizer(
            line,
            joint_line,
            max_length=self.max_target_length if is_label else self.max_source_length,
            padding="max_length" if self.pad_to_max_length else True,
            truncation=True,
        )
        if is_label:
            pad_token_id = self.pad_token_id
            labels = torch.LongTensor(inputs["input_ids"])
            labels[labels==pad_token_id] = -100
            return labels if return_tensor else labels.tolist()
        else:
            return (torch.LongTensor(inputs["input_ids"]), torch.LongTensor(inputs["attention_mask"])) if return_tensor else (inputs["input_ids"], inputs["attention_mask"])
    
    def _append_knowledge(self, history, knowledge_list):
        return [history + kn for kn in knowledge_list]
    
    def _pad_knowledge(self, input_ids, attention_mask):
        kn_len, seq_len = input_ids.shape[0], input_ids.shape[1]
        if kn_len == self.max_knowledge:
            return input_ids, attention_mask

        padding_seq = torch.ones(self.max_knowledge-kn_len, seq_len, dtype=torch.int64)
        input_ids = torch.cat([input_ids, padding_seq * self.pad_token_id], dim=0)
        attention_mask = torch.cat([attention_mask, padding_seq * 0], dim=0)
        return input_ids, attention_mask
    
    def _pad_knowledge_tolist(self, input_ids, is_mask=False, max_knowledge=-1):
        kn_len, seq_len = len(input_ids), len(input_ids[0])
        if max_knowledge == -1 or kn_len == max_knowledge:
            return input_ids

        if is_mask:
            padding_seq = [[0]*seq_len for _ in range(max_knowledge-kn_len)]
            input_ids.extend(padding_seq)
        else:
            padding_seq = [[self.pad_token_id]*seq_len for _ in range(max_knowledge-kn_len)]
            input_ids.extend(padding_seq)

        return input_ids
    
    def _split_by_stride(self, stride, input_ids, attention_mask=None, knowledge_mask=None, keep=-1):
        split_input_ids, split_attention_mask, split_knowledge_mask = [], [], []
        curr = 0
        for d in stride:
            new_split_input = input_ids[curr:curr+d].copy() if keep == -1 else input_ids[curr:curr+min(d, keep)].copy()
            if self.pad_knowledge and self.max_knowledge != -1: 
                new_split_input = self._pad_knowledge_tolist(new_split_input, max_knowledge=min(self.max_knowledge, keep) if keep>0 else self.max_knowledge)
            split_input_ids.append(new_split_input)

            if attention_mask is not None:
                new_split_mask = attention_mask[curr:curr+d].copy() if keep == -1 else attention_mask[curr:curr+min(d, keep)].copy() 
                if self.pad_knowledge and self.max_knowledge != -1: 
                    new_split_mask = self._pad_knowledge_tolist(new_split_mask, is_mask=True, max_knowledge=min(self.max_knowledge, keep) if keep>0 else self.max_knowledge)
                split_attention_mask.append(new_split_mask)
            if knowledge_mask is not None:
                new_split_knowledge = knowledge_mask[curr:curr+d].copy() if keep == -1 else knowledge_mask[curr:curr+min(d, keep)].copy() 
                if self.pad_knowledge and self.max_knowledge != -1: 
                    new_split_knowledge = self._pad_knowledge_tolist(new_split_knowledge, is_mask=True, max_knowledge=min(self.max_knowledge, keep) if keep>0 else self.max_knowledge)
                split_knowledge_mask.append(new_split_knowledge)

            curr += d
        return split_input_ids, split_attention_mask, split_knowledge_mask


    def _prepare_tokenized_inputs(self, samples, tokenizer):
        input_seqs = samples["input_ids"]
        labels = samples["labels"]

        task = self.data_args.dataset_config_name
        if task == "vae":
            """Tokenize data sample for methods based on/similar to VAE"""
            stride = samples["stride"]
            posterior_input_seqs = samples["posterior_input_ids"]
            knowledges = samples["knowledges"]
            kn_labels = samples["kn_labels"]

            input_ids, attention_mask = self._tokenize_lines(tokenizer, input_seqs, joint_line=knowledges)
            posterior_input_ids, posterior_attention_mask = self._tokenize_lines(tokenizer, posterior_input_seqs, joint_line=knowledges)

            labels = self._tokenize_lines(tokenizer, labels, is_label=True)

            # Make knowledge masks by attention_mask(H+k) - attention_mask(H)
            _, knowledge_mask = self._tokenize_lines(tokenizer, input_seqs)
            _, posterior_knowledge_mask = self._tokenize_lines(tokenizer, posterior_input_seqs)

            attention_mask           = torch.LongTensor(attention_mask)
            posterior_attention_mask = torch.LongTensor(posterior_attention_mask)
            knowledge_mask           = attention_mask - torch.LongTensor(knowledge_mask)
            posterior_knowledge_mask = posterior_attention_mask - torch.LongTensor(posterior_knowledge_mask) 

            attention_mask           = attention_mask.tolist()
            posterior_attention_mask = posterior_attention_mask.tolist()
            knowledge_mask           = knowledge_mask.tolist()
            posterior_knowledge_mask = posterior_knowledge_mask.tolist()

            if isinstance(stride, list):
                input_ids, attention_mask, knowledge_mask = self._split_by_stride(stride, input_ids, attention_mask=attention_mask, knowledge_mask=knowledge_mask)
                posterior_input_ids, posterior_attention_mask, posterior_knowledge_mask = self._split_by_stride(stride, posterior_input_ids, attention_mask=posterior_attention_mask, knowledge_mask=posterior_knowledge_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "knowledge_mask": knowledge_mask,
                "labels": labels,
                "classification_labels": kn_labels,
                "posterior_input_ids": posterior_input_ids,
                "posterior_attention_mask": posterior_attention_mask,
                "posterior_knowledge_mask": posterior_knowledge_mask,
            }
        elif "posterior" in task:
            """Tokenize data sample for methods based on/similar to posterior inference"""
            stride = samples["stride"]
            knowledges = samples["knowledges"]
            kn_labels = samples["kn_labels"]

            input_ids, attention_mask = self._tokenize_lines(tokenizer, input_seqs, joint_line=knowledges)

            labels = self._tokenize_lines(tokenizer, labels, is_label=True)

            # Make knowledge masks by attention_mask(H+k) - attention_mask(H)
            _, knowledge_mask = self._tokenize_lines(tokenizer, input_seqs)

            attention_mask = torch.LongTensor(attention_mask)
            knowledge_mask = attention_mask - torch.LongTensor(knowledge_mask)
            attention_mask = attention_mask.tolist()
            knowledge_mask = knowledge_mask.tolist()

            if isinstance(stride, list):
                input_ids, attention_mask, knowledge_mask = self._split_by_stride(stride, input_ids, attention_mask=attention_mask, knowledge_mask=knowledge_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "knowledge_mask": knowledge_mask,
                "labels": labels,
                "classification_labels": kn_labels,
            }
        elif task == "latent":
            """Baseline setting: not to add knowledge into inputs of decoder"""
            stride = samples["stride"]
            knowledges = samples["knowledges"]
            kn_labels = samples["kn_labels"]

            input_ids, attention_mask = self._tokenize_lines(tokenizer, input_seqs, joint_line=knowledges)

            labels = self._tokenize_lines(tokenizer, labels, is_label=True)

            # Make knowledge masks by attention_mask(H+k) - attention_mask(H)
            ctx_input_ids, ctx_attention_mask = self._tokenize_lines(tokenizer, input_seqs)

            attention_mask = torch.LongTensor(attention_mask)
            knowledge_mask = attention_mask - torch.LongTensor(ctx_attention_mask)
            attention_mask = attention_mask.tolist()
            knowledge_mask = knowledge_mask.tolist()

            if isinstance(stride, list):
                input_ids, attention_mask, knowledge_mask = self._split_by_stride(stride, input_ids, attention_mask=attention_mask, knowledge_mask=knowledge_mask, keep=-1)
                ctx_input_ids, ctx_attention_mask, _ = self._split_by_stride(stride, ctx_input_ids, attention_mask=ctx_attention_mask, keep=1)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "knowledge_mask": knowledge_mask,
                "ctx_input_ids": ctx_input_ids,
                "ctx_attention_mask": ctx_attention_mask,
                "labels": labels,
                "classification_labels": kn_labels,
            }
        else:
            raise NotImplementedError


    def _prepare_sequence(self, samples):
        task = self.data_args.dataset_config_name
        outputs = defaultdict(list)
        history = samples["history"]
        response = samples["response"]

        if task == "default" or task == "nokn":
            """Basic baseline setting for KGD: dialogue history+gold kn"""
            outputs["input_ids"] = history
            outputs["labels"] = response
        elif task == "random":
            # TODO randomly shuffle the knowledge and truncate when it reaches the maximum length
            raise NotImplementedError
        elif task == "fid":
            """Tokenize data samples for methods based on/similar to Fusion-in-Decoder"""
            knowledges = samples["knowledges"]
            kn_label = samples["knowledge_label"]


            outputs["input_ids"].extend(input_seqs)
            outputs["labels"] = response
            outputs["stride"].append(len(input_seqs))

        elif task == "vae":
            """Tokenize data samples for methods based on/similar to VAE"""
            posterior_history = samples["posterior_history"]
            knowledges = samples["knowledges"]
            kn_label = samples["knowledge_label"]

            outputs["labels"] = response
            outputs["kn_labels"] = kn_label

            for input_seqs, posterior_input_seqs, knowledge in zip(history, posterior_history, knowledges):

                outputs["input_ids"].extend(input_seqs)
                outputs["posterior_input_ids"].extend(posterior_input_seqs)
                outputs["knowledges"].extend(knowledge)
                outputs["stride"].append(len(input_seqs))
        
        elif "posterior" in task or task == "latent":
            """Tokenize data samples for methods based on/similar to posterior inference"""
            knowledges = samples["knowledges"]
            kn_label = samples["knowledge_label"]

            outputs["labels"] = response
            outputs["kn_labels"] = kn_label

            for input_seqs, knowledge in zip(history, knowledges):

                outputs["input_ids"].extend(input_seqs)
                outputs["knowledges"].extend(knowledge)
                outputs["stride"].append(len(input_seqs))

        else:
            raise NotImplementedError
        
        return outputs


    def __call__(self, samples, tokenizer):
        outputs = self._prepare_sequence(samples)
        tokenized_samples = self._prepare_tokenized_inputs(outputs, tokenizer)
        return tokenized_samples





def preprocess(training_args, data_args, datasets, tokenizer, merge_eval=False):
    if "holle" in data_args.dataset_name:
        return preprocess_holle(training_args, data_args, datasets, tokenizer, merge_eval=merge_eval)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    task = data_args.dataset_config_name
    cache_appx = ""
    PADKN = data_args.pad_knowledge if hasattr(data_args, "pad_knowledge") else False
    if PADKN: cache_appx += "_pad"

    processor_kwargs = {
        "training_args": training_args,
        "data_args": data_args,
        "tokenizer": tokenizer,
    }

    dialogue_preprocessor = DialogReader(**processor_kwargs)
    
    def preprocess_function(samples):
        return dialogue_preprocessor(samples, tokenizer)
    
    def preprocess_function_eval(samples):
        dialogue_preprocessor.max_knowledge = -1
        return dialogue_preprocessor(samples, tokenizer)

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        
        cache_path = os.path.join(data_args.preproc_dir, f"cache_wow_train_{data_args.dataset_config_name}"+cache_appx)
        if os.path.exists(cache_path) and (not data_args.overwrite_cache):
            train_dataset = load_from_disk(cache_path)
        else:
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    batch_size=10,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess train dataset",
                )
                train_dataset.save_to_disk(cache_path)

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset        = datasets["validation"]
        eval_unseen_dataset = datasets["validation_unseen"]
        
        dataset_config_name_for_save = data_args.dataset_config_name.replace("_lr2", "").replace("_lr4", "").replace("_lr8", "").replace("_lr16", "")
        cache_path = os.path.join(data_args.preproc_dir, f"cache_wow_valid_{dataset_config_name_for_save}"+cache_appx)
        cache_unseen_path = os.path.join(data_args.preproc_dir, f"cache_wow_valid_unseen_{dataset_config_name_for_save}"+cache_appx)
        if os.path.exists(cache_path) and os.path.exists(cache_unseen_path) and (not data_args.overwrite_cache):
            eval_dataset = load_from_disk(cache_path)
            eval_unseen_dataset = load_from_disk(cache_unseen_path)
        else:
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function_eval,
                    batched=True,
                    batch_size=100,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess valid dataset",
                )
                eval_dataset.save_to_disk(cache_path)
            with training_args.main_process_first(desc="validation unseen dataset map pre-processing"):
                eval_unseen_dataset = eval_unseen_dataset.map(
                    preprocess_function_eval,
                    batched=True,
                    batch_size=100,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess valid unseen dataset",
                )
                eval_unseen_dataset.save_to_disk(cache_unseen_path)
        
        if data_args.max_eval_samples is not None:
            eval_dataset        = eval_dataset.select(range(data_args.max_eval_samples))
            eval_unseen_dataset = eval_unseen_dataset.select(range(data_args.max_eval_samples))

        eval_datasets = concatenate_datasets([eval_dataset, eval_unseen_dataset]) if merge_eval else (eval_dataset, eval_unseen_dataset)
    else:
        eval_datasets = (None, None)
    

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset        = datasets["test"]
        test_unseen_dataset = datasets["test_unseen"]
        
        dataset_config_name_for_save = data_args.dataset_config_name.replace("_lr2", "").replace("_lr4", "").replace("_lr8", "").replace("_lr16", "")
        cache_path = os.path.join(data_args.preproc_dir, f"cache_wow_test_{dataset_config_name_for_save}"+cache_appx)
        cache_unseen_path = os.path.join(data_args.preproc_dir, f"cache_wow_test_unseen_{dataset_config_name_for_save}"+cache_appx)
        if os.path.exists(cache_path) and os.path.exists(cache_unseen_path) and (not data_args.overwrite_cache):
            test_dataset = load_from_disk(cache_path)
            test_unseen_dataset = load_from_disk(cache_unseen_path)
        else:
            with training_args.main_process_first(desc="test dataset map pre-processing"):
                test_dataset = test_dataset.map(
                    preprocess_function_eval,
                    batched=True,
                    batch_size=100,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess test dataset",
                )
                test_dataset.save_to_disk(cache_path)
            with training_args.main_process_first(desc="test unseen dataset map pre-processing"):
                test_unseen_dataset = test_unseen_dataset.map(
                    preprocess_function_eval,
                    batched=True,
                    batch_size=100,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess test unseen dataset",
                )
                test_unseen_dataset.save_to_disk(cache_unseen_path)
        
        if data_args.max_eval_samples is not None:
            test_dataset        = test_dataset.select(range(data_args.max_eval_samples))
            test_unseen_dataset = test_unseen_dataset.select(range(data_args.max_eval_samples))

        test_datasets = (test_dataset, test_unseen_dataset)
    else:
        test_datasets = (None, None)
    
    return train_dataset, eval_datasets, test_datasets


def preprocess_holle(training_args, data_args, datasets, tokenizer, merge_eval=False):
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    task = data_args.dataset_config_name
    cache_appx = ""
    PADKN = data_args.pad_knowledge if hasattr(data_args, "pad_knowledge") else False
    if PADKN: cache_appx += "_pad"

    processor_kwargs = {
        "training_args": training_args,
        "data_args": data_args,
        "tokenizer": tokenizer,
    }

    dialogue_preprocessor = DialogReader(**processor_kwargs)
    
    def preprocess_function(samples):
        return dialogue_preprocessor(samples, tokenizer)
    
    def preprocess_function_eval(samples):
        dialogue_preprocessor.max_knowledge = -1
        return dialogue_preprocessor(samples, tokenizer)

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        
        cache_path = os.path.join(data_args.preproc_dir, f"cache_holle_train_{data_args.dataset_config_name}"+cache_appx)
        if os.path.exists(cache_path) and (not data_args.overwrite_cache):
            train_dataset = load_from_disk(cache_path)
        else:
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    batch_size=10,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess train dataset",
                )
                train_dataset.save_to_disk(cache_path)

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        
        cache_path = os.path.join(data_args.preproc_dir, f"cache_holle_test_{data_args.dataset_config_name}"+cache_appx)
        if os.path.exists(cache_path) and (not data_args.overwrite_cache):
            eval_dataset = load_from_disk(cache_path)
        else:
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function_eval,
                    batched=True,
                    batch_size=100,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess valid dataset",
                )
                eval_dataset.save_to_disk(cache_path)
        
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        eval_datasets = eval_dataset
    else:
        eval_datasets = None
    

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset        = datasets["test"]
        
        cache_path = os.path.join(data_args.preproc_dir, f"cache_holle_test_{data_args.dataset_config_name}"+cache_appx)
        if os.path.exists(cache_path) and (not data_args.overwrite_cache):
            test_dataset = load_from_disk(cache_path)
        else:
            with training_args.main_process_first(desc="test dataset map pre-processing"):
                test_dataset = test_dataset.map(
                    preprocess_function_eval,
                    batched=True,
                    batch_size=100,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Preprocess test dataset",
                )
                test_dataset.save_to_disk(cache_path)
        
        if data_args.max_eval_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_eval_samples))

        test_datasets = (test_dataset, None)
    else:
        test_datasets = (None, None)
    
    return train_dataset, eval_datasets, test_datasets

