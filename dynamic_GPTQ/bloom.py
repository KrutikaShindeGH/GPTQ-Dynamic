import math
import time
import torch
import torch.nn as nn
import transformers

from gptq import * 
from modelutils import *
from quant import *

def get_bloom(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto', device_map='auto')
    model.seqlen = 2048
    return model

def compute_sensitivity(model, dataloader, dev):
    sensitivities = {}
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            sensitivities[name] = 0.0
            
    with torch.enable_grad():
        for batch in dataloader:
            inputs = batch[0].to(dev)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            model.zero_grad()
            loss.backward()
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.grad is not None:
                    sensitivities[name] += (module.weight.grad ** 2).sum().item()
                    module.weight.grad = None
                    
    for name in sensitivities:
        sensitivities[name] /= len(dataloader)
    return sensitivities

def allocate_bits(sensitivities, target_avg_bits=3.5, max_bits=8, min_bits=2):
    layers = list(sensitivities.keys())
    num_layers = len(layers)
    bit_widths = {name: max_bits for name in layers}
    current_avg_bits = float(max_bits)
    sorted_layers = sorted(layers, key=lambda name: sensitivities[name])
    while current_avg_bits > target_avg_bits:
        reduced = False
        for name in sorted_layers:
            if bit_widths[name] > min_bits:
                bit_widths[name] -= 1
                current_avg_bits -= (1.0 / num_layers)
                reduced = True
                if current_avg_bits <= target_avg_bits:
                    break
        if not reduced:
            break
    return bit_widths

@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None):
    print('Profiling Layer Sensitivities (Phase 1)...')
    sensitivities = compute_sensitivity(model, dataloader, dev)
    print('Allocating Dynamic Bits (Phase 2)...')
    optimal_bits = allocate_bits(sensitivities, target_avg_bits=args.target_bits)
    print('Starting sequential quantization (Phase 3)...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            full_layer_name = 'transformer.h.%d.%s' % (i, name)
            bits = optimal_bits.get(full_layer_name, args.wbits)
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                bits, perchannel=True, sym=args.sym, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize)
            quantizers['transformer.h.%d.%s' % (i, name)] = gptq[name].quantizer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return quantizers

@torch.no_grad()
def bloom_eval(model, testenc, dev):
    print('Evaluation...')
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        layers[i] = layer.cpu() 
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    model.config.use_cache = use_cache

def bloom_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str, choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--nearest', action='store_true')
    parser.add_argument('--wbits', type=int, default=16)
    parser.add_argument('--target_bits', type=float, default=3.5)
    parser.add_argument('--groupsize', type=int, default=-1)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--new-eval', action='store_true')

    args = parser.parse_args()
    DEV = torch.device('cuda:0')

    model = get_bloom(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if not args.nearest:
        tick = time.time()
        quantizers = bloom_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        bloom_eval(model, testloader, DEV)

    if args.save:
        bloom_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

