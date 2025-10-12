import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score, precision_recall_fscore_support, classification_report, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from dotenv import load_dotenv
import os
os.environ["WANDB_DISABLED"] = "true"

from huggingface_hub import login

import evaluate

from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import logging

from peft import get_peft_model, LoraConfig, TaskType

import time

import torch

import argparse

def create_output_figure_dir(figure_output_dir):
    os.makedirs(figure_output_dir, exist_ok=True)

def process_dataset(dataset, tokenizer):
    # helper function to tokenize the text
    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    # apply tokenization
    dataset = dataset.map(tokenize, batched=True)
    # set format to torch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset

def load_model(apply_lora):
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_PATH, num_labels=3)
    if not apply_lora:
        print("Loaded model for full finetuning")
        return model
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        print("Loaded model for finetuning with LoRA")
        model.print_trainable_parameters()
        return model

def finetune_model(base_model, train_dataset, val_dataset, metric):
    # helper to compute metrics during training
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # convert the logits to their predicted class
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average='weighted')
    
    # define training arguments
    training_args = TrainingArguments(
        output_dir=HF_OUTPUT_DIR,
        learning_rate=1e-5,
        logging_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        report_to=None
    )

    # define trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # time the training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training took {end_time - start_time} seconds.")

    return trainer

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='2f', cmap='Blues')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # save figure to output directory
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")

def plot_roc(labels, probabilities, save_path):
    # roc curve for all 3 classes
    # one vs all ROC curve
    labels_onehot = label_binarize(labels, classes=[0, 1, 2])

    plt.figure(figsize=(15, 10))
    for i in range(3):
        fpr, tpr, _ = roc_curve(labels_onehot[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0,1], [0,1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend()
    
    # save figure to output directory
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")

def plot_pr_curve(labels, probabilities, save_path):
    labels_onehot = label_binarize(labels, classes=[0, 1, 2])
    n_classes = labels_onehot.shape[1]

    # Compute Precision-Recall for each class
    plt.figure(figsize=(12, 8))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(labels_onehot[:, i], probabilities[:, i])
        ap = average_precision_score(labels_onehot[:, i], probabilities[:, i], average='weighted')
        plt.plot(recall, precision,
                label=f'Class {i} (Average Precision = {ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('One-vs-Rest Precision-Recall Curve')
    plt.legend()

    # save figure to output directory
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to {save_path}")

def evaluate_model(trainer, dataset):
    print("Validation performance:")
    trainer.evaluate(dataset['validation'])

    print("Test performance:")
    trainer.evaluate(dataset['test'])

    # get the true labels and probabilities
    predictions = trainer.predict(dataset["test"])
    logits = predictions.predictions
    labels = predictions.label_ids
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = np.argmax(logits, axis=1)

    # create and save confusion matrix
    plot_confusion_matrix(labels, preds, f"{FIGURE_OUTPUT_DIR}/confusion_matrix.png")

    # create and save ROC curve
    plot_roc(labels, probabilities, f"{FIGURE_OUTPUT_DIR}/ROC_curve.png")

    # create and save precision-recall curve
    plot_pr_curve(labels, probabilities, f"{FIGURE_OUTPUT_DIR}/precision_recall_curve.png")

    # generate classification report for test dataset
    report = classification_report(labels, preds)
    print("Classification report for test set:")
    print(report)

# main function to run
def main():
    # create output directory if not exists
    create_output_figure_dir(FIGURE_OUTPUT_DIR)

    # load base model and tokenizer
    model = load_model(apply_lora=APPLY_LORA)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    # load dataset
    dataset = load_dataset(DATASET_PATH)
    dataset = process_dataset(dataset, tokenizer)

    # set up finetuning metric
    metric = evaluate.load("f1")

    # finetune model
    trainer = finetune_model(model, dataset['train'], dataset['validation'], metric)

    # evaluate model on validation and test set
    evaluate_model(trainer, dataset)

    # push model to huggingface if user specified hf_output_dir
    if HF_OUTPUT_DIR:
        trainer.push_to_hub()
        print(f"Pushed trained model to huggingface repository: {HF_OUTPUT_DIR}")
    
    print("Training and evaluation complete.")

if __name__ == '__main__':
    # load environment variables
    load_dotenv()
    login(os.getenv('HUGGINGFACE_API_KEY'))

    # parse terminal arguments
    parser = argparse.ArgumentParser(description="Finetuning BERT")
    parser.add_argument("--figure_output_dir", help="Directory to output plots to", required=True)
    # add optional argument to push to huggingface repository if user specifies it
    parser.add_argument("--hf_output_dir", help="Huggingface repository to push the trained model to", default=None)
    parser.add_argument("--apply_lora", help="Whether to apply LoRA or not to finetuning", choices=["true", "false"], required=True)

    args = parser.parse_args()

    BASE_MODEL_PATH = "google-bert/bert-base-cased"
    DATASET_PATH = "Jiahao123/FinancialPhraseBank_processed"
    APPLY_LORA = args.apply_lora.lower() == "true" # whether to apply LoRA or not
    HF_OUTPUT_DIR = args.hf_output_dir
    FIGURE_OUTPUT_DIR = args.figure_output_dir

    # run main function
    main()