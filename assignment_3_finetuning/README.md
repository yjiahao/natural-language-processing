## Running Instructions

After cloning repository, from root, change directory.

```bash
cd assignment_3_finetuning
```

### Downloading data

Download data from kaggle at this link: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

- Add the `all-data.csv` file into a folder `data/` (you may create the data folder) at the same hierarchy as the notebooks.

### Setting up environment variables

Create a .env file according to the .env.example file found in this repository, and add in your credentials.

### Virtual environment

Create virtual environment and activate it.

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install requirements

```bash
pip install -r requirements.txt
```

### Run notebooks

Now, you may navigate to `data_processing.ipynb` or `finetune_BERT_LoRA.ipynb` or `finetune_BERT.ipynb` to run the code in the notebooks.

### Running main finetuning file

To run the full finetuning pipeline, you may use the following command as an example:

```bash
cd src

python main.py \
    --figure_output_dir ../figures \
    --apply_lora true \
    --hf_output_dir financial_classifier
```

`--figure_output_dir`: Where to output figures from evaluation (confusion matrix, ROC curve, Precision-Recall curve).
`--apply_lora`: Whether to apply finetuning with LoRA or not. true or false.
`--hf_output_dir`: Repository to save finetuned model to. Optional argument, will not perform saving if not specified.