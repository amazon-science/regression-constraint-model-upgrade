# Code and files related to backward-compatible updates in Natural Language Processing

The details will be released in the upcoming paper. If you find these files useful consider citing the paper

```
@article{Schumann2023BCWI,
  title={Backward Compatibility During Data Updates by Weight Interpolation
},
  author={Raphael Schumann and Elman Mansimov and Yi-An Lai and Nikolaos Pappas and Xibin Gao and Yi Zhang},
  journal={ArXiv},
  year={2023},
}


```

# Reproduce Results

## Requirements
```
python==3.8
torch==1.10.0
transformers==4.19.2
```

## Other Information
The experiments in the paper are all run with ten different random seeds: 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1010. Add this to the seeds argument to fully reproduce the numbers in the paper

The choices for scenarios are: add_data, add_classes

The choices for datasets are MASSIVE, banking77, ag_news

## Prerequisites

```
#Train old model and new model
python train.py --dataset MASSIVE --scenario add_data --seeds 1111 2222 --output_dir v1
python run_experiments.py --datasets MASSIVE --scenarios add_data --exp_group new_model_add_data --seeds 1111 2222 --output_dir v1
```


```
# Show results for old and new model
python results/print_results.py --dataset MASSIVE --scenario add_data --exp_group old_model --seeds 1111 2222 --output_dir v1
python results/print_results.py --dataset MASSIVE --scenario add_data --exp_group new_model_add_data --seeds 1111 2222 --output_dir v1
```

## Reproduce BCWI Results
```
python bcwi/run_bcwi.py --datasets MASSIVE --scenarios add_data --variant plain --seeds 1111 2222 --output_dir v1
python results/print_bcwi.py --dataset MASSIVE --scenario add_data --variant plain --seeds 1111 2222 --output_dir v1
```

# Reproduce SoupBCWI Results
```
python run_experiments.py --datasets MASSIVE --scenarios add_data --exp_group ensemble_add_data --seeds 1111 --output_dir v1
python bcwi/run_bcwi.py --datasets MASSIVE --scenarios add_data --variant multi --seeds 1111 --output_dir v1
python results/print_bcwi.py --dataset MASSIVE --scenario add_data --variant multi --seeds 1111 --output_dir v1
```

# Reproduce fisherBCWI Results
```
python methods/create_fisher_information_matrix.py --datasets MASSIVE --scenarios add_data --seeds 1111 2222 --output_dir v1
python bcwi/run_bcwi.py --datasets MASSIVE --scenarios add_data --variant fisher --seeds 1111 2222 --output_dir v1
python results/print_bcwi.py --dataset MASSIVE --scenario add_data --variant fisher --seeds 1111 2222 --output_dir v1
```

# Reproduce Target Model
```
python run_experiments.py --datasets MASSIVE --scenarios add_data --exp_group target_model_add_data --seeds 1111 2222 --output_dir v1
python results/print_results.py --dataset MASSIVE --scenario add_data --exp_group target_model_add_data --seeds 1111 2222 --output_dir v1
```

# Reproduce Baselines
The baseline experiment groups are: bitfit_add_data, bitfit_add_classes, ia3, mixout, pre_wd, ewc, distillation 
```
python run_experiments.py --datasets MASSIVE --scenarios add_data --exp_group mixout --seeds 1111 2222 --output_dir v1
python results/print_results.py --dataset MASSIVE --scenario add_data --exp_group mixout --seeds 1111 2222 --output_dir v1