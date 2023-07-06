import os
import json
import random
import numpy as np
from transformers import set_seed
from tqdm import tqdm

from src.data_utils.utils import load_jsons
from src.data_utils.wow_less import parse_knowledge

set_seed(42)

def load_samples(path, history_length=1, max_knowledge=34):
    episodes = load_preproc_samples(path)
    
    samples = []
    for episode_num, episode in enumerate(tqdm(episodes, ncols=70)):
        history = []
        for example_num, example in enumerate(episode):
            # Tokenize inputs and convert to tokens
            history.append(example['text'])
            if "train" in path:
                response = example['labels'][0]
            else:
                response = example['eval_labels'][0]
            chosen_topic = example['chosen_topic']

            # Set up knowledge
            checked_knowledge = example['title'] + ' __knowledge__ ' + example['checked_sentence']
            knowledge_list = [checked_knowledge] + \
                [k for k in example['knowledge'].rstrip().split('\n')]
            for idx, k in enumerate(knowledge_list[1:]):
                if k == checked_knowledge:
                    break
            else:
                # Sometimes, knowledge does not include checked_sentnece
                idx = None
                print("Knowledge does not include checked sentence.")
            if idx is not None:
                del knowledge_list[idx + 1]
            
            knowledges, knowledge_label = parse_knowledge(knowledge_list, max_knowledge)

            truncated_history = history[-(2 * history_length + 1):].copy()
            sample = {
                        "idx": f"d{str(episode_num)}_t{str(example_num)}",
                        "history": truncated_history,
                        "knowledges": knowledges,
                        "knowledge_label": knowledge_label,
                        "knowledge_text": checked_knowledge,
                        "response": response,
                    }
            if 'multi_eval_labels' in example:
                responses = [response for response in example['multi_eval_labels']]
                sample['responses'] = responses
            if 'multi_checked_sentences' in example:
                gt_knowledge_sentences = [example['title'] + ' __knowledge__ ' + checked_sentence
                                        for checked_sentence
                                        in example['multi_checked_sentences']]
                sample['knowledge_texts'] = gt_knowledge_sentences
            
            history.append(response)
            samples.append(sample)
    return samples


def load_preproc_samples(path):
    return load_jsons(path)


def load_split_samples(data_args, splits):
    datasets = []
    for split in splits:
        filename = split.replace(".json", f"_hl{data_args.history_length}_kn{data_args.max_knowledge if 'train.json' in splits else -1}")+".json"
        cache_path = os.path.join(data_args.preproc_dir, filename)
        if os.path.exists(cache_path):
            datasets.append(load_preproc_samples(cache_path))
        else:
            data_path = os.path.join(data_args.data_dir, split)
            datasets.append(load_samples(data_path, history_length=data_args.history_length, max_knowledge=data_args.max_knowledge if "train.json" in splits else -1))

            os.makedirs(data_args.preproc_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                for sample in datasets[-1]:
                    f.write(json.dumps(sample)+"\n")
    
    if len(datasets) == 1:
        return datasets[0]
    else:
        return tuple(datasets)


def load_holle_data(training_args, data_args):

    if training_args.do_train:
        train_dataset = load_split_samples(data_args, ["train.json"])
    else:
        train_dataset = None
    
    if training_args.do_eval:
        valid_dataset = load_split_samples(data_args, ["test.json"])
    else:
        valid_dataset = None

    if training_args.do_predict:
        test_dataset = load_split_samples(data_args, ["test.json"])
    else:
        test_dataset = None
    
    datasets = {
        "train": train_dataset,
        "valid": test_dataset,
        "test":  test_dataset,
    }
    return datasets



if __name__ == "__main__":
    from transformers import Seq2SeqTrainingArguments, HfArgumentParser
    from main import DataTrainingArguments

    parser = HfArgumentParser((DataTrainingArguments, Seq2SeqTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train   = True
    training_args.do_eval    = True
    training_args.do_predict = True

    data_args.data_dir = "./data/holle"

    load_holle_data(training_args, data_args)