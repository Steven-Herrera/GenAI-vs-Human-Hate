"""Train HateBERT"""

import os
from dotenv import load_dotenv
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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# loads environment variables from a .env file to allow mlflow to send
# metrics to the DagsHub server
load_dotenv()


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
def create_text_classification_dataset(
    df, tokenizer_name, max_len, train_size=0.6, val_size=0.2
):
    texts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_len = tokenizer.model_max_length

    if train_size + val_size >= 1.0:
        raise ValueError("The sum of train_size and val_size must be less than 1.0")

    test_size = 1.0 - (train_size + val_size)

    train_texts, remaining_texts, train_labels, remaining_labels = train_test_split(
        texts, labels, train_size=train_size, stratify=labels
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        remaining_texts,
        remaining_labels,
        train_size=val_size / (val_size + test_size),
        stratify=remaining_labels,
    )

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_len
    )
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_len)
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, max_len
    )

    return train_dataset, val_dataset, test_dataset


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


def model_preds(model, data_loader, device):
    model.eval()

    y_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            y_pred = model(input_ids, attention_mask)
            y_pred = torch.argmax(y_pred, dim=1).cpu()

            y_preds.extend(y_pred.numpy().tolist())

    return y_preds


def _hsd_label_to_int(label):
    """Converts the labels found in the HSD_final_merged dataset into integers for training a model

    Args:
        label (str): Either Human or AI

    Returns:
        integer_label (int): 0 for human and 1 for AI

    Raises:
        ValueError: Label isn't Human or AI
    """
    if label == "Human":
        integer_label = 0
    elif label == "AI":
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


def load_experiment3_dataset(filepath="../../../data/final_hsd_1217.csv"):
    """
    Loads the 'final_hsd_1217.csv' dataset. This dataset contains ONLY AI non-hate, AI hate, and Human hate.
    Experiment 3:
        Train a model on AI non-hate and AI hate

    Args:
        filepath (str): Filepath to the final_hsd_1217.csv file

    Returns:
        df3 (pd.DataFrame): Only AI non-hate and AI hate
    """
    cols = ["text", "label"]
    df = pd.read_csv(filepath)

    ai_nonhate_df = df[(df["Source"] == "AI") & (df["Label"] == 0)]
    ai_hate_df = df[(df["Source"] == "AI") & (df["Label"] == 1)]

    nonhate_size = ai_nonhate_df.shape[0]
    hate_size = ai_hate_df.shape[0]

    min_size = min(nonhate_size, hate_size)

    ai_nonhate_df_v2 = ai_nonhate_df.head(min_size).rename(columns={"Label": "label"})
    ai_hate_df_v2 = ai_hate_df.head(min_size).rename(columns={"Label": "label"})

    df3 = pd.concat([ai_nonhate_df_v2[cols], ai_hate_df_v2[cols]])
    return df3


def load_experiment4_dataset(filepath="../../../data/final_hsd_1217.csv"):
    """
    Loads the 'final_hsd_1217.csv' dataset. This dataset contains ONLY AI non-hate, AI hate, and Human hate.
    Experiment 4:
        Train a model on AI non-hate and human hate

    Args:
        filepath (str): Filepath to the final_hsd_1217.csv file

    Returns:
        df4 (pd.DataFrame): Only AI non-hate and human hate
    """
    cols = ["text", "label"]
    df = pd.read_csv(filepath)

    ai_nonhate_df = df[(df["Source"] == "AI") & (df["Label"] == 0)]
    human_hate_df = df[(df["Source"] == "Human") & (df["Label"] == 1)]

    nonhate_size = ai_nonhate_df.shape[0]
    hate_size = human_hate_df.shape[0]

    min_size = min(nonhate_size, hate_size)

    ai_nonhate_df_v2 = ai_nonhate_df.head(min_size).rename(columns={"Label": "label"})
    human_hate_df_v2 = human_hate_df.head(min_size).rename(columns={"Label": "label"})

    df4 = pd.concat([ai_nonhate_df_v2[cols], human_hate_df_v2[cols]])
    return df4


def load_experiment5_dataset(filepath="../../../data/final_hsd_1217.csv"):
    """
    Loads the 'final_hsd_1217.csv' dataset. This dataset contains ONLY AI non-hate, AI hate, and Human hate.
    The entire dataset is 50/50 hate and non-hate. Of the Hate labels, 50% are human and 50% are AI.

    Experiment 5:
        Train a model on AI non-hate and BOTH human hate and AI hate

    Args:
        filepath (str): Filepath to the final_hsd_1217.csv file

    Returns:
        df5 (pd.DataFrame): AI non-hate and AI/human hate
    """
    cols = ["text", "label"]
    df = pd.read_csv(filepath)

    ai_nonhate_df = df[(df["Source"] == "AI") & (df["Label"] == 0)]
    ai_hate_df = df[(df["Source"] == "AI") & (df["Label"] == 1)]
    human_hate_df = df[(df["Source"] == "Human") & (df["Label"] == 1)]

    nonhate_size = ai_nonhate_df.shape[0]
    half_size = int(nonhate_size / 2)

    ai_nonhate_df_v2 = ai_nonhate_df.rename(columns={"Label": "label"})
    human_hate_df_v2 = human_hate_df.head(half_size).rename(columns={"Label": "label"})
    ai_hate_df_v2 = ai_hate_df.head(half_size).rename(columns={"Label": "label"})

    df5 = pd.concat(
        [ai_nonhate_df_v2[cols], ai_hate_df_v2[cols], human_hate_df_v2[cols]]
    )
    return df5


def lst_to_df(texts, labels, filepath):
    """Creates a pandas dataframe from two lists

    Args:
        texts (List[str]): Text
        labels (List[int]): Labels
        filepath (str): filepath to save csv

    Returns:
        df (pd.DataFrame): Pandas dataframe
    """
    data = {"text": texts, "label": labels}
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return df


# Main function
def main():
    # Configurations
    ITER_NUM = 0
    MAX_LEN = 512
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 16
    EPOCHS = 30
    PATIENCE = 5
    TRAIN_SIZE = 0.6
    VAL_SIZE = .2
    PRETRAINED_MODEL_NAME = "tomh/toxigen_roberta"
    model_name = PRETRAINED_MODEL_NAME.split("/")[-1]
    NUM_CLASSES = 2
    CHECKPOINT_DIR = f"checkpoints_hsd_{model_name}_{ITER_NUM}"
    LOG_DIR = f"logs_hsd_{model_name}_{ITER_NUM}"
    GPU_ID = 4
    

    print("Setting up mlflow")
    # Initialize mlflow
    mlflow.set_tracking_uri(
        "https://dagshub.com/Steven-Herrera/GenAI-vs-Human-Hate.mlflow"
    )
    mlflow.set_experiment("HateBERT Training")

    with mlflow.start_run():
        # Load data
        print("loading dataset from csv")
        df = load_hsd_dataset()  # Assume this function returns a labeled dataframe
        print("creating train/val/test split")
        train_dataset, val_dataset, test_dataset = create_text_classification_dataset(
            df, PRETRAINED_MODEL_NAME, MAX_LEN, TRAIN_SIZE, VAL_SIZE
        )

        _ = lst_to_df(
            train_dataset.texts, train_dataset.labels, f"{model_name}_train.csv"
        )
        _ = lst_to_df(val_dataset.texts, val_dataset.labels, f"{model_name}_val.csv")
        _ = lst_to_df(test_dataset.texts, test_dataset.labels, f"{model_name}_test.csv")

        mlflow.log_artifact(f"{model_name}_train.csv", "train_datasets")
        mlflow.log_artifact(f"{model_name}_val.csv", "val_datasets")
        mlflow.log_artifact(f"{model_name}_test.csv", "test_datasets")

        print(f"creating train/val/test loaders with batch size: {BATCH_SIZE}")
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print("Instantiating model")
        # Initialize model, optimizer, and loss function
        model = CustomHateBERTModel(PRETRAINED_MODEL_NAME, NUM_CLASSES)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # TensorBoard writer
        writer = SummaryWriter(log_dir=f"{model_name}_{LOG_DIR}")

        best_f1 = 0
        best_model_state = None
        epochs_without_improvement = 0

        print(f"Training loop with patience of {PATIENCE} epochs")
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

            # Early Stopping
            if val_f1 > best_f1:
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1
            if early_stopping(epochs_without_improvement, PATIENCE):
                print("Early stopping triggered.")
                break

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

        # Log hyperparameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("model name", PRETRAINED_MODEL_NAME)
        mlflow.log_param("patience", PATIENCE)
        mlflow.log_param("Learning Rate", LEARNING_RATE)
        mlflow.log_param("Max len", MAX_LEN)
        mlflow.log_param("Train size", TRAIN_SIZE)

        test_loss, test_acc, test_f1 = validate_model(
            model, test_loader, criterion, device
        )

        print(
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1-Score: {test_f1:.4f}"
        )

        writer.add_scalar("test_loss", test_loss)
        writer.add_scalar("test_accuracy", test_acc)
        writer.add_scalar("test_f1", test_f1)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        # Log final model to DagsHub
        print("Logging best model to MLflow")
        model.load_state_dict(best_model_state)
        best_model_state_path = f"{model_name}_model.pt"
        torch.save(best_model_state, best_model_state_path)

        # Infer model signature
        example_input = next(iter(test_loader))[0].to(device)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=f"{model_name}_model",
            conda_env=mlflow.pytorch.get_default_conda_env(),
            input_example=example_input,
            signature=mlflow.models.infer_signature(
                example_input.cpu().numpy(), model(example_input).cpu().detach().numpy()
            ),
        )

        print(f"Model {model_name}_model logged successfully!")

        y_pred = model_preds(model, test_loader, device)
        y_true = test_dataset.labels
        test_cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(test_cm, annot=True, fmt="d")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.savefig(f"{model_name}_test_cm.png")
        mlflow.log_artifact(f"{model_name}_test_cm.png")


if __name__ == "__main__":
    main()
