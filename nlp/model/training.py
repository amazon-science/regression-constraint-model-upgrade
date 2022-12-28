import time
import math

import torch
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import get_parameter_names
from torch.cuda.amp import autocast
import numpy as np

from methods.prior_wd_optim import PriorWD
from model.evaluate import evaluate, calculate_metrics
from model.utils import custom_collate_fn
from experiment.utils import setup_seed

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(model, config, metrics_func, dataset, writer=None, log_batch=False, freeze_old_classes=False, fp16=False):
    print('config: {}'.format(config))
    start_time = time.time()
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if metrics_func is None:
        metrics_func = calculate_metrics

    model, config, optimizer, scheduler, train_dataloader = _prepare_training(model, config, dataset, freeze_old_classes)
    setup_seed(config['seed'])

    dev_metrics, _ = evaluate('dev', model, dataset, config, metrics_func, writer=writer, step=0)
    for key, value in dev_metrics.items():
        print('dev {}: {:.3f}'.format(key, value))

    step = 0
    for epoch in range(config['num_epochs']):
        model.train()

        epoch += 1
        print('epoch', epoch)
        epoch_losses = list()
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            step += 1
            optimizer.zero_grad()

            model_inputs = dict(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device),
                                labels=batch['labels'].to(device))
            if hasattr(model, 'requires_instance_ids'):
                model_inputs['instance_ids'] = batch['instance_ids']
                model_inputs['split'] = 'train'
                model_inputs['epoch'] = epoch
                model_inputs['total_epochs'] = config['num_epochs']

            # train step
            loss, logits, model_log = _get_outputs(model, model_inputs, fp16)

            _backward_and_optimizer_step(loss, model, config, scaler, optimizer, fp16)
            scheduler.step()
            if freeze_old_classes:
                # freeze old label ids weights
                old_label_ids, freeze_out_proj_weight, freeze_out_proj_bias = freeze_old_classes
                model.classifier.out_proj.weight.data[old_label_ids, :] = freeze_out_proj_weight.to(device)
                model.classifier.out_proj.bias.data[old_label_ids] = freeze_out_proj_bias.to(device)

            # logging metrics
            epoch_losses.append(loss.item())
            #writer.add_scalar('Loss/train', loss.item(), global_step=step)
            #for key, value in model_log.items():
            #    writer.add_scalar(key, value, global_step=step)
            if log_batch:
                predictions = torch.argmax(logits, dim=-1)
                batch_metrics = metrics_func('train', predictions, batch['labels'], batch['instance_ids'], strict=False)
                #for key, value in batch_metrics.items():
                    #if value is not None:
                    #    writer.add_scalar(f'{key}/train', value, global_step=step)

        # more metrics logging
        print('train loss: {:.5f}'.format(np.mean(epoch_losses)))

        dev_metrics, _ = evaluate('dev', model, dataset, config, metrics_func, writer=writer, step=step)
        for key, value in dev_metrics.items():
            print('dev {}: {:.3f}'.format(key, value))

        print('epoch time: {:.1f} sec'.format((time.time() - epoch_start_time)))
        print('')

    print('train time: {} sec'.format(int((time.time() - start_time))))

    return model


def _get_outputs(model, inputs, fp16=False):
    if fp16:
        with autocast():
            outputs = model(**inputs)
            logits, loss = outputs.logits, outputs.loss
    else:
        outputs = model(**inputs)
        logits, loss = outputs.logits, outputs.loss

    model_log = dict()
    if outputs.attentions is not None:
        model_log = outputs.attentions

    return loss, logits, model_log


def _backward_and_optimizer_step(loss, model, config, scaler, optimizer, fp16):
    if fp16:
        scaler.scale(loss).backward()
        if config['max_grad_norm'] > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if config['max_grad_norm'] > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
        optimizer.step()


def _prepare_training(model, config, dataset, freeze_old_classes):

    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    train_split = 'train'
    if config.get('data_type') == 'additional':
        train_split = 'train_added'
    train_dataloader = DataLoader(dataset[train_split],
                                  shuffle=True,
                                  generator=generator,
                                  collate_fn=custom_collate_fn,
                                  batch_size=config['batch_size'])

    config['num_training_steps'] = math.ceil(config['num_epochs'] * len(train_dataloader))
    config['warmup_steps'] = math.ceil(config['num_training_steps'] * config['warmup_ratio'])
    config['num_training_steps'] = config['num_epochs'] * len(train_dataloader)

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": config['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    model.to(device)

    if config.get('prior_wd', False):
        optimizer = AdamW(model.parameters(),
                          lr=config['learning_rate'],
                          weight_decay=0.0,
                          betas=(config['adam_beta1'], config['adam_beta2']),
                          eps=config['adam_epsilon']
                          )
        skip_classifier = False
        if freeze_old_classes:
            skip_classifier = True
        optimizer = PriorWD(optimizer,
                            value=config['prior_wd'],
                            skip_classifier=skip_classifier)
    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'],
                          betas=(config['adam_beta1'], config['adam_beta2']),
                          eps=config['adam_epsilon']
                          )

    scheduler = get_scheduler(name="linear",
                              optimizer=optimizer,
                              num_warmup_steps=config['warmup_steps'],
                              num_training_steps=config['num_training_steps'])

    return model, config, optimizer, scheduler, train_dataloader
