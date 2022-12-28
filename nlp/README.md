# Files related to backward-compatible updates in Natural Language Processing

## MASSIVE

 MASSIVE [FitzGerald et al., 2022](https://github.com/alexa/massive) is a natural language understanding
dataset with 60 intents covering basic domains of a
virtual assistant. 

We use a subset of that dataset to simulate the data update scenario. Please take a look at `MASSIVE/old` and `MASSIVE/updated` folders for more information. `barchart` contains the high-level summary of dataset.\

### Train

``python train.py --dataset MASSIVE --scenario add_data --seeds 1111 --output_dir v1``
