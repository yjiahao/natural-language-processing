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

Now, you may navigate to `data_processing.ipynb` or `` to run the code in the notebooks.