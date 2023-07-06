import time
import torch
import numpy as np

from typing import Dict

class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, decoder_start_token_id):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] if torch.is_tensor(x["input_ids"]) else torch.LongTensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([x["attention_mask"] if torch.is_tensor(x["attention_mask"]) else torch.LongTensor(x["attention_mask"]) for x in batch])
        labels = torch.stack([x["labels"] if torch.is_tensor(x["labels"]) else torch.LongTensor(x["labels"]) for x in batch])

        shapes = input_ids.shape
        if len(shapes) == 3:
            # input_ids, attention_mask = trim_batch_3d(input_ids, self.pad_token_id, attention_mask=attention_mask)
            num_kn         = input_ids.shape[1]
            input_ids      = input_ids.view(-1, shapes[-1])
            attention_mask = attention_mask.view(-1, shapes[-1])

        labels = trim_batch(labels, -100)
        input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if len(shapes) == 3:
            encoder_attention_mask = attention_mask.clone().view(shapes[0], -1)
            batch.update({
                "decoder_encoder_attention_mask": encoder_attention_mask,
                "decoder_shapes": tuple([shapes[0], num_kn, input_ids.shape[1]]),
            })
        return batch



class Seq2SeqDataCollatorWithLV:
    def __init__(self, tokenizer, data_args, decoder_start_token_id, vae_kl_weights=1.0, accumulation_steps=1, return_posterior=True, no_kn_decoding=False):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."

        # Latent variable-specific configurations
        self.vae_kl_weights = vae_kl_weights
        # self.max_step = len(self.vae_kl_weights) if vae_kl_weights is not None else 0
        # self.step = 0
        # self.inner_count = 0
        self.accumulation_steps = accumulation_steps
        self.return_posterior = return_posterior
        self.is_latent = data_args.dataset_config_name == "latent"
        self.no_kn_decoding  = no_kn_decoding
        self.trunc_knowledge = (data_args.pad_knowledge and data_args.max_knowledge >0) if hasattr(data_args, 'pad_knowledge') else False

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] if torch.is_tensor(x["input_ids"]) else torch.LongTensor(x["input_ids"]) for x in batch])
        attention_mask = torch.stack([x["attention_mask"] if torch.is_tensor(x["attention_mask"]) else torch.LongTensor(x["attention_mask"]) for x in batch])
        knowledge_mask = torch.stack([x["knowledge_mask"] if torch.is_tensor(x["knowledge_mask"]) else torch.LongTensor(x["knowledge_mask"]) for x in batch])
        labels = torch.stack([x["labels"] if torch.is_tensor(x["labels"]) else torch.LongTensor(x["labels"]) for x in batch])
        classification_labels = torch.LongTensor([x["classification_labels"] for x in batch])

        shapes = input_ids.shape
        cls_mask = None
        if len(shapes) == 3:
            if self.trunc_knowledge: 
                input_ids, attention_mask, knowledge_mask, cls_mask = trim_batch_3d(
                    input_ids, 
                    self.pad_token_id, 
                    attention_mask=attention_mask, 
                    knowledge_mask=knowledge_mask, 
                    return_mask=True,
                )
            num_kn         = input_ids.shape[1]
            input_ids      = input_ids.view(-1, shapes[-1])
            attention_mask = attention_mask.view(-1, shapes[-1])
            knowledge_mask = knowledge_mask.view(-1, shapes[-1])

        labels = trim_batch(labels, -100)
        input_ids, attention_mask, knowledge_mask = trim_batch(
            input_ids, 
            self.pad_token_id, 
            attention_mask=attention_mask, 
            knowledge_mask=knowledge_mask
        )

        new_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "knowledge_mask": knowledge_mask,
            "labels": labels,
            "classification_labels": classification_labels,
            "vae_kl_weight": self.vae_kl_weights, # self.vae_kl_weights[self.step % self.max_step] if self.vae_kl_weights is not None else None,
        }
        if cls_mask is not None:
            new_batch.update({"decoder_cls_mask": cls_mask})

        if self.return_posterior:
            posterior_input_ids = torch.stack([x["posterior_input_ids"] if torch.is_tensor(x["posterior_input_ids"]) else torch.LongTensor(x["posterior_input_ids"]) for x in batch])
            posterior_attention_mask = torch.stack([x["posterior_attention_mask"] if torch.is_tensor(x["posterior_attention_mask"]) else torch.LongTensor(x["posterior_attention_mask"]) for x in batch])
            posterior_knowledge_mask = torch.stack([x["posterior_knowledge_mask"] if torch.is_tensor(x["posterior_knowledge_mask"]) else torch.LongTensor(x["posterior_knowledge_mask"]) for x in batch])

            if len(shapes) == 3:
                if self.trunc_knowledge: posterior_input_ids, posterior_attention_mask, posterior_knowledge_mask = trim_batch_3d(posterior_input_ids, self.pad_token_id, attention_mask=posterior_attention_mask, knowledge_mask=posterior_knowledge_mask)
                posterior_input_ids      = posterior_input_ids.view(-1, shapes[-1])
                posterior_attention_mask = posterior_attention_mask.view(-1, shapes[-1])
                posterior_knowledge_mask = posterior_knowledge_mask.view(-1, shapes[-1])

            posterior_input_ids, posterior_attention_mask, posterior_knowledge_mask = trim_batch(
                posterior_input_ids, 
                self.pad_token_id, 
                attention_mask=posterior_attention_mask, 
                knowledge_mask=posterior_knowledge_mask
            )
            new_batch.update({
                "posterior_input_ids": posterior_input_ids,
                "posterior_attention_mask": posterior_attention_mask,
                "posterior_knowledge_mask": posterior_knowledge_mask,
            })
        
        if self.is_latent:
            ctx_input_ids = torch.stack([x["ctx_input_ids"] if torch.is_tensor(x["ctx_input_ids"]) else torch.LongTensor(x["ctx_input_ids"]) for x in batch])
            ctx_attention_mask = torch.stack([x["ctx_attention_mask"] if torch.is_tensor(x["ctx_attention_mask"]) else torch.LongTensor(x["ctx_attention_mask"]) for x in batch])
            if len(shapes) == 3:
                if self.trunc_knowledge: ctx_input_ids, ctx_attention_mask = trim_batch_3d(ctx_input_ids, self.pad_token_id, attention_mask=ctx_attention_mask)
                ctx_input_ids = ctx_input_ids.view(-1, shapes[-1])
                ctx_attention_mask = ctx_attention_mask.view(-1, shapes[-1])
            
            seq_len_trim = attention_mask.shape[1]
            ctx_attention_mask = ctx_attention_mask[:, :seq_len_trim]
            new_batch.update({"ctx_attention_mask": ctx_attention_mask})
            if self.no_kn_decoding:
                ctx_input_ids = ctx_input_ids[:, :seq_len_trim]
                new_batch.update({"ctx_input_ids": ctx_input_ids})
            
        
        if len(shapes) == 3:
            # encoder_attention_mask = attention_mask.clone().view(shapes[0], num_kn, -1)
            new_batch.update({
                # "decoder_encoder_attention_mask": encoder_attention_mask,
                "decoder_shapes": tuple([shapes[0], num_kn, input_ids.shape[1]]),
            })
        
        # if self.max_step > 0:
        #     self.inner_count += 1
        #     if self.inner_count == self.accumulation_steps:
        #         self.inner_count = 0
        #         self.step += 1
        #     if self.step > self.max_step:
        #         print(f"The next step {self.step} will exceed the maximum step limitation {self.max_step} for ")
        return new_batch

    
def trim_batch(input_ids, pad_token_id, attention_mask=None, knowledge_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        outputs = input_ids[:, keep_column_mask]
    else:
        outputs = (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    
    if knowledge_mask is None:
        return outputs
    else:
        return outputs + (knowledge_mask[:, keep_column_mask],) if isinstance(outputs,tuple) else (outputs, knowledge_mask[:, keep_column_mask])

def trim_batch_3d(input_ids, pad_token_id, attention_mask=None, knowledge_mask=None, return_mask=False):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=2).ne(0).any(dim=0)
    if attention_mask is None:
        outputs = input_ids[:, keep_column_mask, :]
    else:
        outputs = (input_ids[:, keep_column_mask, :], attention_mask[:, keep_column_mask, :])
    
    if return_mask:
        mask = input_ids.ne(pad_token_id).any(dim=2).long()[:, keep_column_mask]
    if knowledge_mask is None:
        if return_mask:
            return outputs + (mask, ) 
        else:
            return outputs 
    else:
        if return_mask:
            return outputs + (knowledge_mask[:, keep_column_mask, :], mask,) if isinstance(outputs,tuple) else (outputs, knowledge_mask[:, keep_column_mask, :], mask)
        else:
            return outputs + (knowledge_mask[:, keep_column_mask, :],) if isinstance(outputs,tuple) else (outputs, knowledge_mask[:, keep_column_mask, :])


def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=10, ratio_increase=0.25, ratio_zero=0.5):
    """This function is taken from 
    https://github.com/ChunyuanLI/Optimus/blob/master/code/examples/big_ae/utils.py#L996"""
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                L[int(i+c*period)] = start
            else: 
                L[int(i+c*period)] = v
                v += step
            i += 1
    return L 


if __name__ == "__main__":
    L = frange_cycle_zero_linear(100, n_cycle=10)
    print(L)
    print(len(L))