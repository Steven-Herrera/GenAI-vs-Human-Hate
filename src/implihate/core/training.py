"""Train HateBERT"""

import os
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datasets import load_dataset
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


def create_binary_dataset():
    """Imports the Implicit Hate and Toxigen datasets. Labels all Implicit Hate data as 0 for
    human generated and all data from Toxigen as 1 for AI generated. Only the implicit hate
    generated by the ALICE technique is used.

    Returns:
        ai_df (pd.DataFrame): Human and AI generated hate
    """
    # load implicit hate dataset
    cols = ["text", "label"]
    implicit_df = pd.read_csv(
        "../../../data/implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv", sep="\t"
    )
    implicit_df_v2 = implicit_df[implicit_df["class"] == "implicit_hate"]
    implicit_df_v3 = implicit_df_v2.rename(columns={"post": "text"})
    implicit_df_v4 = implicit_df_v3[["text"]]
    implicit_df_v4["label"] = 0

    implicit_df_v4["label"] = implicit_df_v4["label"].astype(int)
    implicit_df_v4["text"] = implicit_df_v4["text"].astype(str)
    implicit_df_v4 = implicit_df_v4[cols]

    # load toxigen
    toxigen_data = load_dataset("toxigen/toxigen-data", name="train")
    toxigen_df = pd.DataFrame.from_dict(toxigen_data["train"])
    alice_df = toxigen_df[toxigen_df["generation_method"] == "ALICE"]

    alice_hate_df = alice_df[alice_df["prompt_label"] == 1]
    alice_df_v2 = alice_hate_df.rename(
        columns={"generation": "text", "prompt_label": "label"}
    )

    alice_df_v2["label"] = alice_df_v2["label"].astype(int)
    alice_df_v2["text"] = alice_df_v2["text"].astype(str)

    alice_df_v2 = alice_df_v2[cols]

    # binary dataset
    ai_df = pd.concat([implicit_df_v4, alice_df_v2])
    return ai_df


class CustomHateBERTModel(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, dropout_rate=0.3):
        super(CustomHateBERTModel, self).__init__()
        # Load the pre-trained HateBERT model
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size

        # Add custom layers for classification
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Forward pass through HateBERT
        encoder_outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = self.pooling(last_hidden_state.permute(0, 2, 1)).squeeze(-1)

        # Apply dropout and dense layer
        dropped_output = self.dropout(pooled_output)
        logits = self.fc(dropped_output)
        return logits


# Save checkpoint function
def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_f1,
    best_f1,
    save_dir="checkpoints",
    model_name="model",
    filename=None,
):
    if val_f1 > best_f1:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if filename is None:
            filename = f"{model_name}_best_epoch_{epoch}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            },
            save_path,
        )
        print(f"Improved F1-Score! Checkpoint saved at: {save_path}")
        return val_f1
    return best_f1


# Early stopping function
def early_stopping(epochs_without_improvement, patience):
    return epochs_without_improvement > patience


# Metric: F1 score
def f1_score_metric(y_true, y_pred):
    y_pred_classes = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred_classes, average="macro")


# Dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Create dataset function
def create_text_classification_dataset(df, tokenizer_name, max_len, train_size=0.8):
    texts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, train_size=train_size, stratify=labels
    )

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_len
    )
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_len)

    return train_dataset, val_dataset


# Train model function
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc, total_f1 = 0, 0, 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_score(labels.cpu(), torch.argmax(outputs, dim=1).cpu())
        total_f1 += f1_score_metric(labels, outputs)

    return (
        total_loss / len(data_loader),
        total_acc / len(data_loader),
        total_f1 / len(data_loader),
    )


# Validate model function
def validate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_f1 = 0, 0, 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += accuracy_score(
                labels.cpu(), torch.argmax(outputs, dim=1).cpu()
            )
            total_f1 += f1_score_metric(labels, outputs)

    return (
        total_loss / len(data_loader),
        total_acc / len(data_loader),
        total_f1 / len(data_loader),
    )


def _hsd_label_to_int(label):
    """Converts the labels found in the HSD_final_merged dataset into integers for training a model

    Args:
        label (str): Either Human or AI

    Returns:
        integer_label (int): 0 for human and 1 for AI

    Raises:
        ValueError: Label isn't Human or AI
    """
    if "Human":
        integer_label = 0
    elif "AI":
        integer_label = 1
    else:
        raise ValueError(f"label should be Human or AI. Got: {label}")
    return integer_label


def load_hsd_dataset():
    """Load Yueru's version of the merged datasets

    Returns:
        hsd_df (pd.DataFrame): A dataframe with the texts and labels

    Raises:
        AssertionError: hsd_df columns are not exactly text and label
    """
    cols = ["text", "label"]
    df = pd.read_csv("../../../data/HSD_final_merged.csv")
    df["Label"] = df["Label"].apply(lambda x: _hsd_label_to_int(x))
    df = df.rename(columns={"Label": "label"})
    hsd_df = df[cols]
    hsd_cols = list(hsd_df.columns)
    assert hsd_cols == cols, hsd_cols
    return hsd_df


# Main function
def main():
    # Configurations
    MAX_LEN = 128
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 10
    PATIENCE = 3
    TRAIN_SIZE = 0.8
    PRETRAINED_MODEL_NAME = "GroNLP/hateBERT"
    model_name = PRETRAINED_MODEL_NAME.split("/")[-1]
    NUM_CLASSES = 2
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    GPU_ID = 4

    # Initialize mlflow
    mlflow.set_tracking_uri("https://dagshub.com/YOUR_USERNAME/YOUR_REPO_NAME.mlflow")
    mlflow.set_experiment("HateBERT Training")

    with mlflow.start_run():
        # Load data
        df = load_hsd_dataset()  # Assume this function returns a labeled dataframe
        train_dataset, val_dataset = create_text_classification_dataset(
            df, PRETRAINED_MODEL_NAME, MAX_LEN, TRAIN_SIZE
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = CustomHateBERTModel(PRETRAINED_MODEL_NAME, NUM_CLASSES)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # TensorBoard writer
        writer = SummaryWriter(log_dir=f"{model_name}_{LOG_DIR}")

        best_f1 = 0
        epochs_without_improvement = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc, train_f1 = train_model(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, val_f1 = validate_model(
                model, val_loader, criterion, device
            )

            print(f"Epoch {epoch}/{EPOCHS}")
            print(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}"
            )
            print(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )

            # Log metrics
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("F1/train", train_f1, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val", val_f1, epoch)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            # Save checkpoint
            best_f1 = save_checkpoint(
                model,
                optimizer,
                epoch,
                val_f1,
                best_f1,
                CHECKPOINT_DIR,
                model_name=model_name,
            )

            # Early stopping
            if val_f1 > best_f1:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if early_stopping(epochs_without_improvement, PATIENCE):
                print("Early stopping triggered.")
                break

        # Log hyperparameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("model name", PRETRAINED_MODEL_NAME)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("Learning Rate", LEARNING_RATE)
        mlflow.log_param("Max len", MAX_LEN)
        mlflow.log_param("Train size", TRAIN_SIZE)
