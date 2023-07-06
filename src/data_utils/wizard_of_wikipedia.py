import os
import json

import datasets
import numpy as np

from transformers import set_seed

from src.data_utils.dialogue_reader import SPEAKERS
from src.data_utils.utils import load_jsons

HLEN = 1
MAXKN = 34
TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'
TOKEN_LABEL = '__label__'
TOKEN_END_LABEL = '__endlabel__'

YOUR_LOCAL_DOWNLOAD = "data" 

_HOMEPAGE = "https://parl.ai/projects/wizard_of_wikipedia/"

_URLs = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"

_DESCRIPTION = """\
In open-domain dialogue intelligent agents should exhibit the use of knowledge, 
however there are few convincing demonstrations of this to date. The most popular 
sequence to sequence models typically "generate and hope" generic utterances that 
can be memorized in the weights of the model when mapping from input utterance(s) 
to output, rather than employing recalled knowledge as context. Use of knowledge 
has so far proved difficult, in part because of the lack of a supervised learning 
benchmark task which exhibits knowledgeable open dialogue with clear grounding. To 
that end we collect and release a large dataset with conversations directly grounded 
with knowledge retrieved from Wikipedia. We then design architectures capable of 
retrieving knowledge, reading and conditioning on it, and finally generating natural 
responses. Our best performing dialogue models are able to conduct knowledgeable 
discussions on open-domain topics as evaluated by automatic metrics and human 
evaluations, while our new benchmark allows for measuring further improvements 
in this important research direction.
"""

_CITATION = """\
@inproceedings{dinan2019wizard,
  author={Emily Dinan and Stephen Roller and Kurt Shuster and Angela Fan and Michael Auli and Jason Weston},
  title={{W}izard of {W}ikipedia: Knowledge-powered Conversational Agents},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2019},
}
"""

PSEUDO_LABELS = {
    "train": "data/processed_wow/train_pseudo_labels.json",
    "valid": "data/processed_wow/valid_random_split_pseudo_labels.json",
    "valid_unseen": "data/processed_wow/valid_topic_split_pseudo_labels.json",
    "test": "data/processed_wow/test_random_split_pseudo_labels.json",
    "test_unseen": "data/processed_wow/test_topic_split_pseudo_labels.json",
}

set_seed(42)

class WizardOfWikipedia(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="Basic baseline setting for KGD: dialogue history+gold kn",
        ),
        datasets.BuilderConfig(
            name="nokn",
            version=VERSION,
            description="Basic baseline setting for KGD: dialogue history without kn",
        ),
        datasets.BuilderConfig(
            name="fid",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to Fusion-in-Decoder",
        ),
        datasets.BuilderConfig(
            name="vae",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to VAE",
        ),
        datasets.BuilderConfig(
            name="posterior",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to posterior inference",
        ),
        datasets.BuilderConfig(
            name="latent",
            version=VERSION,
            description="Baseline setting: not to add knowledge into inputs of decoder",
        ),
        datasets.BuilderConfig(
            name="posterior_lr2",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to posterior inference",
        ),
        datasets.BuilderConfig(
            name="posterior_lr4",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to posterior inference",
        ),
        datasets.BuilderConfig(
            name="posterior_lr8",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to posterior inference",
        ),
        datasets.BuilderConfig(
            name="posterior_lr16",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to posterior inference",
        ),
        datasets.BuilderConfig(
            name="posterior_pseudo",
            version=VERSION,
            description="Tokenize data sample for methods based on/similar to posterior inference",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        if self.config.name == "default":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "history": datasets.Value("string"),
                    "response": datasets.Value("string"),
                }
            )
        elif self.config.name == "nokn":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "history": datasets.Value("string"),
                    "response": datasets.Value("string"),
                }
            )
        elif self.config.name == "fid":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "history": datasets.features.Sequence(
                        feature=datasets.Value("string")
                    ),
                    "knowledge_label": datasets.Value("int32"),
                    "response": datasets.Value("string"),
                }
            )
        elif self.config.name == "vae":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "history": datasets.features.Sequence(
                        feature=datasets.Value("string")
                    ),
                    "posterior_history": datasets.features.Sequence(
                        feature=datasets.Value("string")
                    ),
                    "knowledges": datasets.features.Sequence(
                        feature=datasets.Value("string")
                    ),
                    "knowledge_label": datasets.Value("int32"),
                    "response": datasets.Value("string"),
                }
            )
        elif "posterior" in self.config.name or self.config.name == "latent":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "history": datasets.features.Sequence(
                        feature=datasets.Value("string")
                    ),
                    "knowledges": datasets.features.Sequence(
                        feature=datasets.Value("string")
                    ),
                    "knowledge_label": datasets.Value("int32"),
                    "response": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # my_urls = _URLs
        # data_dir = dl_manager.download_and_extract(my_urls) 
        data_dir = YOUR_LOCAL_DOWNLOAD # point to local dir to avoid downloading the dataset again
        if "lr2" in self.config.name:
            train_path = os.path.join(data_dir, f"processed_wow/train_hl{HLEN}_kn{MAXKN}_lr2.json")
        elif "lr4" in self.config.name:
            train_path = os.path.join(data_dir, f"processed_wow/train_hl{HLEN}_kn{MAXKN}_lr4.json")
        elif "lr8" in self.config.name:
            train_path = os.path.join(data_dir, f"processed_wow/train_hl{HLEN}_kn{MAXKN}_lr8.json")
        elif "lr16" in self.config.name:
            train_path = os.path.join(data_dir, f"processed_wow/train_hl{HLEN}_kn{MAXKN}_lr16.json")
        else:
            train_path = os.path.join(data_dir, f"processed_wow/train_hl{HLEN}_kn{MAXKN}.json")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_path, 
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"processed_wow/valid_random_split_hl{HLEN}_kn-1.json"), 
                    "split": "valid"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("validation_unseen"),
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"processed_wow/valid_topic_split_hl{HLEN}_kn-1.json"), 
                    "split": "valid_unseen"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"processed_wow/test_random_split_hl{HLEN}_kn-1.json"), 
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split("test_unseen"),
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"processed_wow/test_topic_split_hl{HLEN}_kn-1.json"), 
                    "split": "test_unseen"
                },
            ),
        ]

    def _append_knowledge(self, history, knowledge_list):
        return [history + kn for kn in knowledge_list]

    def _generate_examples(self, filepath, split):
        data = load_jsons(filepath)

        if self.config.name == "posterior_pseudo":
            pseudo_label_path = PSEUDO_LABELS[split]
            with open(pseudo_label_path, "r") as f:
                pseudo_labels = json.load(f)        

        samples = []
        for i, sample in enumerate(data):

            id_ = sample["idx"]
            truncated_history = sample["history"]
            knowledges = sample["knowledges"]
            knowledge_label = sample["knowledge_label"]
            kn_text = sample["knowledge_text"]
            utterance = sample["response"]

            history_len = len(truncated_history)
            speaker_idx = 0 if history_len % 2 == 1 else 1

            if self.config.name == "default":
                input_seq = ""
                for utter in truncated_history:
                    input_seq += utter + SPEAKERS[speaker_idx]
                    speaker_idx = (speaker_idx + 1) % 2
                input_seq += kn_text
                example = {
                    "id": id_,
                    "history": input_seq,
                    "response": utterance,
                }
                
                yield id_, example
            elif self.config.name == "nokn":
                input_seq = ""
                for utter in truncated_history:
                    input_seq += utter + SPEAKERS[speaker_idx]
                    speaker_idx = (speaker_idx + 1) % 2
                example = {
                    "id": id_,
                    "history": input_seq,
                    "response": utterance,
                }
                
                yield id_, example
            elif self.config.name == "fid":
                input_seq = ""
                for utter in truncated_history:
                    input_seq += utter + SPEAKERS[speaker_idx]
                    speaker_idx = (speaker_idx + 1) % 2
                input_seqs = self._append_knowledge(input_seq, knowledges)
                example = {
                    "id": id_,
                    "history": input_seqs,
                    "knowledge_label": knowledge_label,
                    "response": utterance,
                }
                
                yield id_, example
            elif self.config.name == "vae":
                input_seq = ""
                posterior_input_seq = ""
                for utter in truncated_history:
                    input_seq += utter + SPEAKERS[speaker_idx]
                    speaker_idx = (speaker_idx + 1) % 2
                posterior_input_seq = input_seq + utterance + SPEAKERS[speaker_idx]

                input_seqs = [input_seq] * len(knowledges)
                posterior_input_seqs = [posterior_input_seq] * len(knowledges)
                example = {
                    "id": id_,
                    "history": input_seqs,
                    "posterior_history": posterior_input_seqs,
                    "knowledges": knowledges,
                    "knowledge_label": knowledge_label,
                    "response": utterance,
                }
                
                yield id_, example
            elif self.config.name == "posterior_pseudo":
                pseudo_label = pseudo_labels[i]
                input_seq = ""
                for utter in truncated_history:
                    input_seq += utter + SPEAKERS[speaker_idx]
                    speaker_idx = (speaker_idx + 1) % 2

                input_seqs = [input_seq] * len(knowledges)
                example = {
                    "id": id_,
                    "history": input_seqs,
                    "knowledges": knowledges,
                    "knowledge_label": pseudo_label,
                    "response": utterance,
                }
                
                yield id_, example
            elif "posterior" in self.config.name or self.config.name == "latent":
                input_seq = ""
                for utter in truncated_history:
                    input_seq += utter + SPEAKERS[speaker_idx]
                    speaker_idx = (speaker_idx + 1) % 2

                input_seqs = [input_seq] * len(knowledges)
                example = {
                    "id": id_,
                    "history": input_seqs,
                    "knowledges": knowledges,
                    "knowledge_label": knowledge_label,
                    "response": utterance,
                }
                
                yield id_, example

            else:
                raise NotImplementedError