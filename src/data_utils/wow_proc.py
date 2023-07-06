"""This script is modified from https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/wizard_of_wikipedia/agents.py"""

import os
import json
import random
import numpy as np
from transformers import set_seed

from src.data_utils.utils import load_jsons

TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
TOKEN_LABEL = '__label__'
TOKEN_END_LABEL = '__endlabel__'

set_seed(42)

def _first_val(dictionary):
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ''


def _first_key(dictionary):
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ''

def _get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.
    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = _first_val(title_dict) if title_dict else ''
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[1:-1])
            if (
                cand_title1
                and cand_title1 in k_dict
                and sentence in k_dict[cand_title1]
            ):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


def parse_knowledge(knowledge_list, max_knowledge):
    if len(knowledge_list) <= max_knowledge or max_knowledge < 0:
        knowledges = knowledge_list
    else:
        keepers = 1 + np.random.choice(len(knowledge_list) - 1, max_knowledge, False)
        keepers[0] = 0
        knowledges = []
        for idx in keepers:
            knowledges.append(knowledge_list[idx])

    # keep the gold knowledge in the first place
    knowledge_label = 0
    return knowledges, knowledge_label


def len_episode(dialog):
    wizard_first = 'Wizard' in dialog[0]['speaker']
    dialog_len = (len(dialog) - 1) // 2 if wizard_first else len(dialog) // 2
    return dialog_len * 2 - 1 if wizard_first else dialog_len * 2, dialog_len


def load_samples(path, history_length=1, max_knowledge=34):
    with open(path, "r") as f:
        data = json.load(f)

    samples = []
    for dialog_idx, element in enumerate(data):

        wizard_eval = element.get("wizard_eval", -1)
        persona = element.get("persona", "")

        chosen_topic = element.get("chosen_topic", "")
        chosen_topic_passage = element["chosen_topic_passage"]
        dialog = element["dialog"]

        max_len_per_d, dd = len_episode(dialog)
        history = []
        dialog = dialog[:max_len_per_d]
        for turn_idx, turn in enumerate(dialog):

            speaker = turn["speaker"]
            utterance = turn["text"].strip()
            
            if turn_idx == 0 and "wizard" in speaker.lower():
                history.append(chosen_topic)
            elif turn_idx == 0:
                utterance = chosen_topic + "\n" + utterance

            if "wizard" in speaker.lower():
                apprentice_ret_passages = wizard_ret_passages = {}
                if turn_idx != 0:
                    apprentice_entry = dialog[turn_idx - 1]
                    apprentice_ret_passages = apprentice_entry["retrieved_passages"]
                if turn_idx - 2 >= 0:
                    wizard_prev_entry = dialog[turn_idx - 2]
                    wizard_ret_passages = wizard_prev_entry["retrieved_passages"]

                knowledge_dict = {chosen_topic: chosen_topic_passage}
                for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
                    for passage in ret_passes:
                        for k, v in passage.items():
                            if k not in knowledge_dict.keys():
                                knowledge_dict[k] = v

                wizard_entry = turn

                title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
                selected_knowledge = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, sentence)

                knowledge_list = []
                knowledge_label = None
                for title, passage in knowledge_dict.items():
                    for p in passage:
                        cand = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, p)
                        knowledge_list.append(cand)
                        if cand == selected_knowledge:
                            knowledge_label = len(knowledge_list) - 1
                
                # assert knowledge_label is not None
                # move the gold knowledge into the first place
                if knowledge_label is None:
                    if selected_knowledge == "no_passages_used __knowledge__ no_passages_used":
                        knowledge_list = [selected_knowledge] + knowledge_list
                    else:
                        # remove noisy samples: two data samples in the training set
                        continue
                else:
                    knowledge_list[0], knowledge_list[knowledge_label] = knowledge_list[knowledge_label], knowledge_list[0]

                knowledges, knowledge_label = parse_knowledge(knowledge_list, max_knowledge)

                # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
                truncated_history = history[-(2 * history_length + 1):].copy()

                sample = {
                    "idx": f"d{str(dialog_idx)}_t{str(turn_idx)}",
                    "history": truncated_history,
                    "knowledges": knowledges,
                    "knowledge_label": knowledge_label,
                    "knowledge_text": selected_knowledge,
                    "response": utterance,
                }
                samples.append(sample)
            
            history.append(utterance)
    print(f"Obtain {len(samples)} samples for this dataset.")
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


def load_wow_data(training_args, data_args):

    if training_args.do_train:
        train_dataset = load_split_samples(data_args, ["train.json"])
    else:
        train_dataset = None
    
    if training_args.do_eval:
        valid_dataset, valid_unseen_dataset = load_split_samples(data_args, ["valid_random_split.json", "valid_topic_split.json"])
    else:
        valid_dataset        = None
        valid_unseen_dataset = None
    if training_args.do_predict:
        test_dataset, test_unseen_dataset = load_split_samples(data_args, ["test_random_split.json", "test_topic_split.json"])
    else:
        test_dataset        = None
        test_unseen_dataset = None
    
    datasets = {
        "train": train_dataset,
        "valid": (valid_dataset, valid_unseen_dataset),
        "test":  (test_dataset , test_unseen_dataset),
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

    data_args.data_dir = "./data/wizard_of_wikipedia"

    load_wow_data(training_args, data_args)

