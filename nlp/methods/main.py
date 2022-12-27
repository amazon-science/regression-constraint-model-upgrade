import os

from transformers import RobertaForSequenceClassification
import torch
import numpy as np

from model.training import train
from model.evaluate import get_metrics, get_calculate_metrics_func, get_distance
from model.utils import save_hparams_to_dir, save_model_to_dir
from model.utils import load_dataset, get_config, get_writer
from model.utils import load_old_model_info
from methods.utils import get_hyperparameters
from experiment.utils import setup_seed

from methods import freeze_layers
from methods import mixout as mixout_module
from methods import distillation
from methods import ia3_layers
from methods import lora_layers
from methods import prior_wd_optim
from methods import l2_regularization
from methods import elastic_weight_consolidation


def get_run(
        hp,
        from_model='old',
        data_type='updated',
        ft_layers=None,
        mixout=None,
        prior_wd=None,
        l2=None,
        ewc=None,
        distill=None,
        gating=False,
        ia3=False,
        lora=False,
        add_classes_freeze=True,
        save_logits=True,
        save_model=False,
        add_hparam=None,
        log_batch=False,
        cname=None
):
    """
    Main method to oorganize and run the finetuning of a model. One can specify the model initialization, data type and
    finetuning method to be used. Also creates the experiment directory with proper naming and saves results.
    """
    assert data_type in ['updated', 'additional']
    assert from_model in ['old', 'pretrained']

    name = 'FT_{}_on_{}'.format('old' if from_model == 'old' else from_model, data_type)

    if gating:
        assert not save_model

    ft_layers, name = freeze_layers.validate_argument(ft_layers, name)
    mixout, name = mixout_module.validate_argument(mixout, name)
    prior_wd, name = prior_wd_optim.validate_argument(prior_wd, name)
    l2, name = l2_regularization.validate_argument(l2, name)
    ewc, name = elastic_weight_consolidation.validate_argument(ewc, name)
    distill, name = distillation.validate_argument(distill, name)
    ia3, name = ia3_layers.validate_argument(ia3, name)
    lora, name = lora_layers.validate_argument(lora, name)

    if not add_classes_freeze:
        name += '_nonFreezeCls'

    if cname is not None:
        name += '_' + cname

    def run_func(scenario, seed, dataset_files, old_model_dir, method_dir, fp16=False):

        assert scenario in ['add_data', 'add_classes']

        splits = ['train', 'dev', 'test']
        if data_type == 'additional':
            splits.append('train_added')
        dataset, dataset_info = load_dataset(splits, dataset_files)

        assert scenario == dataset_info['scenario']
        new_label_ids = None
        old_label_ids = None
        if scenario == 'add_classes':
            new_label_ids = dataset_info['new_label_ids']
            old_label_ids = dataset_info['old_label_ids']
        dataset_name = dataset_info['name']

        old_model_info = load_old_model_info(old_model_dir)
        pt_model_name = old_model_info['pt_model_name']
        old_instance_ids = old_model_info['old_instance_ids']

        metrics_func = get_calculate_metrics_func(old_model_dir)

        for split in splits:
            print(f'{split}_dataset', dataset[split])
        num_labels = len(dataset_info['labels'])

        if dataset_name in hp:
            _hyperparameters = hp[dataset_name]
        else:
            _hyperparameters = hp

        all_metrics = dict()
        all_logits = dict()

        for hparam, hparam_str in get_hyperparameters(_hyperparameters, add_hparam):
            print(f'run training for {name} with hparams: {hparam_str}')

            model_dir = os.path.join(method_dir, 'model_' + hparam_str)
            config = get_config(hparam, seed)
            writer = get_writer(model_dir, gating, distill)

            config['data_type'] = data_type

            # initialize model and classifier weights
            freeze_old_classes = False
            setup_seed(config['seed'])
            if from_model == 'old':

                model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))

                if scenario == 'add_classes':
                    init_classifier_weight = old_model_info['classifier_init']['old_model'][0]
                    init_classifier_bias = old_model_info['classifier_init']['old_model'][1]
                    if type(init_classifier_weight) == np.ndarray:
                        init_classifier_weight = torch.from_numpy(init_classifier_weight)
                        init_classifier_bias = torch.from_numpy(init_classifier_bias)
                    model.classifier.out_proj.weight.data[new_label_ids, :] = init_classifier_weight[new_label_ids, :]
                    model.classifier.out_proj.bias.data[new_label_ids] = init_classifier_bias[new_label_ids]

                    if add_classes_freeze:
                        freeze_out_proj_weight = model.classifier.out_proj.weight.detach()[old_label_ids, :].clone()
                        freeze_out_proj_bias = model.classifier.out_proj.bias.detach()[old_label_ids].clone()
                        freeze_old_classes = (old_label_ids, freeze_out_proj_weight, freeze_out_proj_bias)

            else:
                model = RobertaForSequenceClassification.from_pretrained(pt_model_name, num_labels=num_labels)

            skip_classifier = (scenario == 'add_classes')
            model = mixout_module.apply(model, mixout, config, old_model_dir, skip_classifier=skip_classifier)
            model = prior_wd_optim.apply(model, prior_wd, config)
            model = freeze_layers.apply_freeze_layers(model, ft_layers)
            model = l2_regularization.apply(model, l2, config, old_model_dir, skip_classifier=skip_classifier)
            model = elastic_weight_consolidation.apply(model, ewc, config, old_model_dir, skip_classifier=skip_classifier)
            model = distillation.apply(model, old_model_dir, distill, config, new_label_ids=new_label_ids)
            model = ia3_layers.apply(model, ia3, config, old_label_ids=old_label_ids)
            model = lora_layers.apply(model, lora, config)

            # direct to actual training of the model
            model = train(model,
                          config,
                          metrics_func=metrics_func,
                          dataset=dataset,
                          writer=writer,
                          log_batch=log_batch,
                          freeze_old_classes=freeze_old_classes,
                          fp16=fp16)

            writer.close()
            save_hparams_to_dir(config, model_dir)

            # evaluate new model
            metrics, logits = get_metrics(['train', 'dev', 'test'], model, dataset, config, metrics_func)
            metrics['hparam'] = hparam

            old_model = RobertaForSequenceClassification.from_pretrained(os.path.join(old_model_dir, 'model'))
            distance = get_distance(old_model, model)
            metrics['distance'] = distance

            all_metrics[hparam_str] = metrics
            all_logits[hparam_str] = logits

            print('distance', distance)
            print('train_metrics', metrics['train'])
            print('dev_metrics', metrics['dev'])
            print('test_metrics', metrics['test'])
            print('')

            if save_model:
                save_model_to_dir(model, model_dir)
            print('')

        if not save_logits:
            all_logits = None

        return all_metrics, all_logits

    return run_func, name
