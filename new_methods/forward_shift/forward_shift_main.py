from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook

from .forward_shift_hparams import ForwardShiftHyperParams
import copy
import functools
import json


def apply_forward_pass_shit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ForwardShiftHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    try:
        if 'gpt2' not in model.config._name_or_path:
            raise ValueError(f"The current implementation only supports gpt2 models")
    except:
        raise ValueError(f"The current implementation only supports gpt2 models")

    if hparams.algo_version == 1:
        deltas = execute_forward_pass_shift_v1(model, tok, requests, hparams)
    else:
        raise ValueError(f"Invalid algorithm version {hparams.algo_version}")


    # at this point model is already updated (not with the same weights as the original one)
    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy



def execute_forward_pass_shift_v1(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ForwardShiftHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """

    """

    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            request["target_new"]["str"] = " " + request["target_new"]["str"]
        print(
            f"Executing Forward Pass SHIFT (V1) algorithm for request: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    # (Save old weights for future restoration)
    weights_copy = {}
    for layer_index in hparams.layers:
        curr_mat = rgetattr(model, hparams.mlp_module_ff2.format(layer_index)).weight
        weights_copy[hparams.mlp_module_ff2.format(layer_index) + '.weight'] = curr_mat.detach().clone()

    # weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    # print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"].format(r["subject"]) for r in requests]
    targets = [r["target_new"]["str"] for r in requests]


    model.eval().requires_grad_(False)

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        if hparams.batch_size != 1:
            raise ValueError("Batch size must be 1 for this algorithm (current implementation limitation)")
        
        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to(model.device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                model.device
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id

            # collecting hidden states from all FF2 layers
            # even in the final version, we used only one layer
            hs_collector = wrap_model(model, 
                                            layers_to_check=[hparams.mlp_module_ff2], 
                                            return_hooks_handler=True,
                                            max_len=1)
            logits = model(**inputs).logits[0, -1]
            remove_collector_hooks(hs_collector)

            if logits.argmax() == target_ids[0,0]:
                print(f"Target token already predicted. Skipping.")
                continue

            target_token_emb = model.transformer.wte(torch.tensor(target_ids[0,0])).squeeze()  # take the first token (in case more than one)
            delta_i = target_token_emb
            for layer_index in hparams.layers:   # allow multiple layers. in practice, we used only one layer
                x_i = copy.deepcopy(hs_collector[layer_index][hparams.mlp_module_ff2]['input'][-1]).to(model.device)
                curr_mat = rgetattr(model, hparams.mlp_module_ff2.format(layer_index)).weight
                manual_grads = torch.zeros_like(curr_mat).to(model.device)
                manual_grads_curr_token = []
                # for i in range(len(x_i)):  
                #     manual_grads_curr_token.append(x_i[i] * delta_i)
                # manual_grads_curr_token = torch.stack(manual_grads_curr_token).to(model.device)
                manual_grads_curr_token = torch.outer(x_i, delta_i)  # same but not using loop
                
                try:
                    manual_grads += manual_grads_curr_token
                except:
                    manual_grads += manual_grads_curr_token.T
                updated_param = torch.nn.Parameter(manual_grads).to(model.device)
                w_new = curr_mat + hparams.lr * updated_param
                rsetattr(model, f'{hparams.mlp_module_ff2.format(layer_index)}.weight', torch.nn.Parameter(w_new))

            bs = inputs["input_ids"].shape[0]
            probs = torch.nn.functional.log_softmax(
                model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
            )
            loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                1
            ) / loss_mask.sum(1)
            loss = loss.mean()
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < hparams.minimum_loss_for_step:
            print(f"Loss is below threshold {loss_meter.avg} < {hparams.minimum_loss_for_step}. Break loop.")
            break

    deltas = {}
    for layer_index in hparams.layers:
        with torch.no_grad():
            curr_mat = rgetattr(model, hparams.mlp_module_ff2.format(layer_index)).weight
            original_mat = weights_copy[hparams.mlp_module_ff2.format(layer_index) + '.weight']
            deltas[hparams.mlp_module_ff2.format(layer_index) + '.weight'] = curr_mat - original_mat

            # Restore state of original model
            rsetattr(model, f'{hparams.mlp_module_ff2.format(layer_index)}.weight', torch.nn.Parameter(original_mat))

    print(f"Deltas successfully computed for {len(deltas.keys())} parameters")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




##### utils code. REF: https://github.com/shacharKZ/VISIT-Visualizing-Transformers #####
# a safe way to get attribute of an object
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

# a safe way to set attribute of an object
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def extract_hs_include_prefix(list_inputs, list_outputs, info='', max_len=256):
    '''
    return a hook function that extract the hidden states (hs) before and after the layer

    @ list_inputs: a list. it will be appended with the hs before the layer (torch.tensor)
    @ list_outputs: a list. it will be appended with the hs after the layer (torch.tensor)
    @ info: a string to use while debugging
    @ max_len: the maximum length of the list. if the list is longer than max_len, the oldest hs will be removed
    
    implemention note for future developers:
    - note we use the easiest way to save the hs, by just appending a copy of the hs to a list
    - if you are going to save this data later to a pickle file, you might want to first change the information 
        from torch.tensor wrapped with list, to pandas or numpy. from our experience that can save a lot of space
    - the information is saved without gradient. if you want to save the gradient you can try and also save it separately
    - use the info parameter to identify the layer you are extracting the hs from (we left the comment from our debugging. it might be useful for you)
    - you should verify that the model is not implemented in a way that the hs is not saved in the same order as the input or it processes 
        them inplace so this information is not representative
    '''
    def hook(module, input, output):
        # print(f'info: {info}, len(input): {len(input)}, len(output): {len(output)}')  # for debugging
        if list_inputs is not None and len(input) > 0 and input[0] is not None:
            last_tokens = input[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]

            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            if len(last_tokens.shape) == 1:
                last_tokens = [last_tokens]  # TODO a workaround for one token long inputs
            for last_token in last_tokens:
                last_token = last_token.squeeze()
                list_inputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)

        if list_outputs is not None and output[0] is not None:
            last_tokens = output[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]

            # print('last_tokens.shape', last_tokens.shape, f'[{info}]')
            if len(last_tokens.shape) == 1:
                last_tokens = [last_tokens]  # TODO a workaround for one token long inputs
            for last_token in last_tokens:
                last_token = last_token.squeeze()
                # print('last_token.shape', last_token.shape, f'[{info}]')
                list_outputs.append(last_token)

                if len(list_inputs) > max_len:
                    list_inputs.pop(0)
                
    return hook


def wrap_model(model,  
               layers_to_check = ['mlp', 'mlp.c_proj', 'mlp.c_fc', '', 'attn.c_attn', 'attn.c_proj', 'attn'],
               configs=None,
               max_len=256,
               return_hooks_handler=False,
               forward=True):
    '''
    a wrapper function for model to collect hidden states
    returns a dictionary that is updated during the forward pass of the model
    and contains the hidden states of the layers specified in layers_to_check for each layer (collcting inputs and outputs of each)
    the dictionary has the following structure:
    {
        layer_idx: {
            layer_type: {
                'input': [list of hidden states (torch.tensor)],
                'output': [list of hidden states (torch.tensor)]
            }
        }
    }
    you can easily access the hidden states of a specific layer by using the following code:
    hs_collector[layer_idx][layer_type]['input'/'outputs'] # list of hidden states of the input of the layer
    to get the hidden state for the last forward pass, you can use:
    hs_collector[layer_idx][layer_type]['input'/'outputs'][-1] # the last hidden state of the input of the layer

    @ model: a pytorch model (currently only support gpt2 models from transformers library)
    @ layers_to_check: a list of strings that specify the layers to collect hidden states from
    @ max_len: the maximum length of the list. if the list is longer than max_len, the oldest hs will be removed

    '''
    
    hs_collector = {}

    # NOTE: the following if is not in used but left from previous implementation
    # in the case of this script, we assume that the layers_to_check is a list of the layer names
    if type(layers_to_check) == str or type(configs) == str:  # assume config file in the format of GraphConfigs
        layers_to_check = layers_to_check if type(layers_to_check) == str else configs
        with open(layers_to_check, 'r') as f:
            tmp_data = json.load(f)
        layers_to_check = set()
        for cell in ["layer_format", "layer_mlp_format", "layer_attn_format", "ln1", 
                     "mlp_ff1", "mlp_ff2", "ln2", "attn_q", "attn_k", "attn_v", "attn_o",
                     "mlp_act"]:  # the defualts are the names of gpt2 layers
            layers_to_check.add(tmp_data[cell])
        layers_to_check = list(layers_to_check)
    elif configs is not None:
        layers_to_check = list(set(configs.layer_format, configs.layer_mlp_format, configs.layer_attn_format, 
                                   configs.ln1, configs.mlp_ff1, configs.mlp_ff2, configs.ln2, 
                                   configs.attn_q, configs.attn_k, configs.attn_v, configs.attn_o))
        
    if configs is not None and type(configs) != str and hasattr(configs, 'n_layer'):
        n_layer = configs.n_layer                           
    elif hasattr(model.config, 'n_layer'):  # gpt2, gpt-j
        n_layer = model.config.n_layer
    elif hasattr(model.config, 'num_layers'):  # gpt-neo
        n_layer = model.config.num_layers
    else:
        n_layer = model.config.num_hidden_layers  # llama2
    
    for layer_idx in range(n_layer):
        hs_collector[layer_idx] = {}
        for layer_type in layers_to_check:
            # the layer_key is key to access the layer in the hs_collector dictionary
            if type(layer_type) == list:
                layer_key, layer_type = layer_type
            else:
                layer_key = layer_type
            hs_collector[layer_idx][layer_key] = {}

            try:
                layer_with_idx = layer_type.format(layer_idx)
                # print(f'layer_with_idx: {layer_with_idx}, layer_type: {layer_type}')  # used for debugging
                layer_pointer = rgetattr(model, layer_with_idx)
            except:
                layer_with_idx = f'{layer_idx}{"." if len(layer_type) else ""}{layer_type}'
                 # "transformer.h" is gpt2's prefix
                layer_pointer = rgetattr(model, f"transformer.h.{layer_with_idx}")

            list_inputs = []
            list_outputs = []
            if forward:
                hooks_handler = layer_pointer.register_forward_hook(
                    extract_hs_include_prefix(
                        list_inputs=list_inputs, 
                        list_outputs=list_outputs, 
                        info=layer_with_idx,
                        max_len=max_len
                        )
                    )
            else:
                hooks_handler = layer_pointer.register_full_backward_hook(
                    extract_hs_include_prefix(
                        list_inputs=list_inputs, 
                        list_outputs=list_outputs, 
                        info=layer_with_idx,
                        max_len=max_len
                        )
                    )

            hs_collector[layer_idx][layer_key]['input'] = list_inputs
            hs_collector[layer_idx][layer_key]['output'] = list_outputs
            if return_hooks_handler:
                hs_collector[layer_idx][layer_key]['hooks_handler'] = hooks_handler

    return hs_collector


def remove_collector_hooks(hs_collector):
    '''
    remove all hooks in hs_collector
    '''
    for layer_idx in hs_collector:
        for layer_type in hs_collector[layer_idx]:
            # print(f'{layer_idx}: layer_type: {layer_type}')
            if 'hooks_handler' not in hs_collector[layer_idx][layer_type]:
                print(f'Warning: no hooks handler for layer {layer_idx} {layer_type}')
            else:
                hooks_handler = hs_collector[layer_idx][layer_type]['hooks_handler']
                hooks_handler.remove()


