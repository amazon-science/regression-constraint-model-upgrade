import os
import pickle
import argparse

from transformers import RobertaForSequenceClassification
import numpy as np
import boto3

from utils import setup_seed
from model.training import train
from model.evaluate import get_metrics, accuracy_metric
from model.utils import save_hparams_to_dir, save_model_to_dir
from model.utils import load_dataset, get_config, load_tokenizer#, get_writer
from experiment.utils import save_metrics, save_logits

fp16 = True
parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--base_path', default='./', type=str, help='Base path of code.')
parser.add_argument('--dataset', default='MASSIVE', help='Name of the dataset; MASSIVE, banking77, ag_news')
parser.add_argument('--scenario', default='add_data', help='Name of the data update scenario. add_data, add_classes')
parser.add_argument('--seeds', nargs="+", default=[1111],
                    help='Run experiments for different random seeds')
parser.add_argument('--output_dir', default='v1', type=str, help='output dir')
parser.add_argument('--data_type', default='old', type=str, help='old or updated data')
parser.add_argument('--bucket_name', default=None, type=str, help='s3 bucket name for saving result')
parser.add_argument('--load_from_s3', action=None, help='specify path to load from s3')
opts = parser.parse_args()


dataset_name = opts.dataset
data_update_scenario = opts.scenario
output_dir = os.path.join(opts.base_path, 'bcwi_nlp_outputs', opts.output_dir)
dataset_dir = os.path.join(opts.base_path, 'data', dataset_name)
seeds = [int(seed) for seed in opts.seeds]
data_type = opts.data_type
bucket_name = opts.bucket_name
load_from_s3 = opts.load_from_s3

if bucket_name is not None or load_from_s3 is not None:
    import boto3
    s3 = boto3.client('s3')
    from utils import upload_directory_to_s3

pt_model_name = 'roberta-base'
load_tokenizer(pt_model_name)

hparam = dict(MASSIVE=dict(e=16, lr=6e-5),
              banking77=dict(e=16, lr=6e-5),
              ag_news=dict(e=8, lr=6e-5))[dataset_name]

def main():
    for seed in seeds:
        method_dir = os.path.join(output_dir, dataset_name, str(seed), data_update_scenario, 'old_model')
        os.makedirs(method_dir, exist_ok=True)

        print('seed', seed)
        print('dataset_name', dataset_name)
        print('data_update_scenario', data_update_scenario)
        print('pretrained_model_name', pt_model_name)

        assert (data_type == "old" or data_type == "updated")
        old_dataset_files = os.path.join(dataset_dir, data_update_scenario, data_type, '{}.jsonl')
        old_dataset, old_dataset_info = load_dataset(['train', 'dev', 'test'], old_dataset_files)


        print('train_dataset', old_dataset['train'])
        print('dev_dataset', old_dataset['dev'])
        print('test_dataset', old_dataset['test'])

        num_labels = len(old_dataset_info['labels'])

        setup_seed(seed)
        if load_from_s3:
            # TODO finish this function
            print (f"Loading from S3 {load_from_s3}")
            # create old_model_dir
            os.makedirs("old_model_dir", exist_ok=True)
            # load stuff from s3
            with open("old_model_dir/config.json", "wb") as f:
                s3.download_fileobj(bucket_name, os.path.join(load_from_s3, "config.json"), f)
                print("Downloaded old_model_dir/config.json")
            with open("old_model_dir/hparams.json", "wb") as f:
                s3.download_fileobj(bucket_name, os.path.join(load_from_s3, "hparams.json"), f)
                print("Downloaded old_model_dir/hparams.json")
            with open("old_model_dir/pytorch_model.bin", "wb") as f:
                s3.download_fileobj(bucket_name, os.path.join(load_from_s3, "pytorch_model.bin"), f)
                print("Downloaded old_model_dir/pytorch_model.bin")

            model = RobertaForSequenceClassification.from_pretrained("old_model_dir", num_labels=num_labels)
            print (f"Initialized RoBERTA from s3 {load_from_s3}")
        else:
            print ("Initializing new RoBERTa classifier")
            model = RobertaForSequenceClassification.from_pretrained(pt_model_name, num_labels=num_labels)
        init_classifier_weight = model.classifier.out_proj.weight.detach().cpu().clone()
        init_classifier_bias = model.classifier.out_proj.bias.detach().cpu().clone()

        config = get_config(hparam, seed)

        model_dir = os.path.join(method_dir, 'model')
        #writer = get_writer(model_dir)
        #model = train(model, config, None, old_dataset, writer, log_batch=True, fp16=fp16)
        model = train(model, config, None, old_dataset, log_batch=True, fp16=fp16)

        #writer.close()
        save_hparams_to_dir(config, model_dir)
        save_model_to_dir(model, model_dir)

        new_dataset_files = os.path.join(dataset_dir, data_update_scenario, 'updated', '{}.jsonl')
        new_dataset, new_dataset_info = load_dataset(['train', 'dev', 'test'], new_dataset_files)
        old_instance_ids = dict(train={e['id'] for e in old_dataset['train']},
                                dev={e['id'] for e in old_dataset['dev']},
                                test={e['id'] for e in old_dataset['test']})
        new_instance_ids = dict(train={e['id'] for e in new_dataset['train']} - old_instance_ids['train'],
                                dev={e['id'] for e in new_dataset['dev']} - old_instance_ids['dev'],
                                test={e['id'] for e in new_dataset['test']} - old_instance_ids['test'])

        metrics, logits = get_metrics(['train', 'dev', 'test'], model, new_dataset, config)

        for split in ['train', 'dev', 'test']:
            instance_ids = logits[split][0]
            predictions = np.argmax(logits[split][1], axis=-1)
            labels = logits[split][2]
            assert len(instance_ids) == len(predictions)
            assert len(instance_ids) == len(labels)
            instance_id_to_idx = {instance_id: idx for idx, instance_id in enumerate(instance_ids)}

            acc_all = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy'] * 100

            indices = [instance_id_to_idx[instance_id] for instance_id in old_instance_ids[split]]
            predictions_old = np.take(predictions, indices, axis=0)
            labels_old = np.take(labels, indices, axis=0)
            acc_old = accuracy_metric.compute(predictions=predictions_old, references=labels_old)['accuracy'] * 100

            indices = [instance_id_to_idx[instance_id] for instance_id in new_instance_ids[split]]
            predictions_new = np.take(predictions, indices, axis=0)
            labels_new = np.take(labels, indices, axis=0)
            acc_new = accuracy_metric.compute(predictions=predictions_new, references=labels_new)['accuracy'] * 100

            metrics[split]['ACC'] = acc_all
            metrics[split]['ACC_OLD'] = acc_old
            metrics[split]['ACC_NEW'] = acc_new

        print(metrics)
        save_metrics(method_dir, {str(hparam): metrics})
        save_logits(method_dir, {str(hparam): logits})

        info = dict()
        info['classifier_init'] = dict(old_model=(init_classifier_weight, init_classifier_bias))
        info['old_instance_ids'] = old_instance_ids
        info['new_instance_ids'] = new_instance_ids
        info['pt_model_name'] = pt_model_name

        with open(os.path.join(method_dir, 'infos.pickle'), 'wb') as f:
            pickle.dump(info, f)

        # if args.bucket_name is not None save the method_dir in the bucket
        if bucket_name is not None:
            print (f"bucket name {bucket_name}")
            print (f"dir {method_dir}")
            upload_directory_to_s3(s3, bucket_name, "", method_dir)
            print (f"uploaded {method_dir} to s3 bucket name {bucket_name}")

        print('train_metrics', metrics['train'])
        print('dev_metrics', metrics['dev'])
        print('test_metrics', metrics['test'])
        print('')

    print('finished old models')


if __name__ == '__main__':
    main()


