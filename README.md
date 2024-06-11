# SVFT: Singular Vector guided Fine Tuning

[Optimized version coming soon]

### Installing Required Packages

```bash
pip install -r requirements.txt
```

### Setting up Commonsense Reasoning
Once the requirements are installed, download the eval datasets i.e the "dataset" folder from https://github.com/AGI-Edgerunners/LLM-Adapters into the LLM-Adapters directory.

```
./run_commonsense.sh
```
Is configured to run Gemma-2B models on Commonesense-15K dataset.

Evaluation is done by running,
```
python3 multi_dataset_eval.py
```

### Setting up Mathematical Reasoning
First, download the MetaMathQA dataset into the ```data/train``` directory. Then download the MetaMathQA-40K dataset
```
cd ./data/train

wget https://huggingface.co/datasets/meta-math/MetaMathQA-40K/resolve/main/MetaMathQA-40K.json
```
To run experiments on Pythia models,
```
./run_pythia.sh
```
For other models, run,
```
./run_math.sh
```
which is currently configured to run Gemma-2B with SVFT.
```run_math.sh``` also contains an example to run evaluation on GSM-8K and MetaMath-40K.

### Vision Experiments

For the vision experiments, see the ReadMe file in the vision experiments folder

## Citation
```
@misc{lingam2024svft,
      title={SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors}, 
      author={Vijay Lingam and Atula Tejaswi and Aditya Vavre and Aneesh Shetty and Gautham Krishna Gudur and Joydeep Ghosh and Alex Dimakis and Eunsol Choi and Aleksandar Bojchevski and Sujay Sanghavi},
      year={2024},
      eprint={2405.19597},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

