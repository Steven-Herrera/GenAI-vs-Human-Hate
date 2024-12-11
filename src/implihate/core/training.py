"""Train HateBERT"""

import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from datasets import load_dataset

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score


def save_checkpoint(
    model, optimizer, epoch, f1_score, best_f1, save_dir="checkpoints", filename=None
):
    """Saves a model checkpoint if F1-score improves."""
    if f1_score > best_f1:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if filename is None:
            filename = f"model_best_epoch_{epoch}.pt"
        save_path = os.path.join(save_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1_score": f1_score,
            },
            save_path,
        )
        print(f"Improved F1-Score! Checkpoint saved at: {save_path}")
        return f1_score  # Update best F1-score
    return best_f1


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


# Metric: Accuracy function
def accuracy(preds, labels):
    _, predictions = torch.max(preds, dim=1)
    return (predictions == labels).sum().item() / labels.size(0)


def f1_score_metric(y_true, y_pred, average="weighted"):
    """Calculates the F1 score for predictions and true labels.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted probabilities or logits.

    Returns:
        float: The F1 score.
    """
    y_pred_classes = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred_classes, average=average)


# Training loop
def train_model(model, data_loader, optimizer, criterion, gpu_id=4):
    """Training loop for a PyTorch model, including accuracy and F1 score computation.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to train on (CPU or GPU).

    Returns:
        tuple: Average loss, accuracy, and F1 score for the epoch.
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    total_loss = 0
    total_acc = 0
    total_f1 = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        acc = (torch.argmax(outputs, dim=1) == labels).sum().item() / labels.size(0)
        total_acc += acc

        # Compute F1 score
        f1 = f1_score_metric(labels, outputs)
        total_f1 += f1

    return (
        total_loss / len(data_loader),
        total_acc / len(data_loader),
        total_f1 / len(data_loader),
    )


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification using a Hugging Face tokenizer.

    Attributes:
        texts (list): A list of input text strings.
        labels (list): A list of integer labels corresponding to the input texts.
        tokenizer (AutoTokenizer): Tokenizer from the Hugging Face Transformers library.
        max_len (int): Maximum length of tokenized input sequences.
    """

    def __init__(self, texts, labels, tokenizer, max_len):
        """Initializes the dataset.

        Args:
            texts (list): A list of strings containing the input texts.
            labels (list): A list of strings or integers containing the labels.
            tokenizer (AutoTokenizer): Tokenizer from the Hugging Face Transformers library.
            max_len (int): Maximum length of tokenized input sequences.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Fetches a single data point at the given index.

        Args:
            idx (int): Index of the data point to fetch.

        Returns:
            dict: A dictionary containing input IDs, attention masks, and the label.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
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


def create_text_classification_dataset(df, tokenizer_name, max_len):
    """Creates a PyTorch dataset from a pandas dataframe for text classification.

    Args:
        df (pd.DataFrame): DataFrame containing two columns: the first column has the text, and the second column has the labels.
        label_mapping (dict): A dictionary mapping string labels to integer class indices.
        tokenizer_name (str): Name of the pre-trained tokenizer (e.g., "GroNLP/hateBERT").
        max_len (int): Maximum length of tokenized input sequences.

    Returns:
        TextClassificationDataset: A PyTorch dataset ready for training.
    """

    # Extract texts and map string labels to integers
    texts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create and return the dataset
    return TextClassificationDataset(texts, labels, tokenizer, max_len)


def main(sub_df=False):
    """Train a binary model with checkpointing and metrics logging."""
    # Configurations
    FRAC = 0.005
    MAX_LEN = 128
    LEARNING_RATE = 0.001
    BATCH_SIZE = 2
    EPOCHS = 3
    CHECKPOINT_INTERVAL = 2  # Save checkpoint every 2 epochs
    pretrained_model_name = "GroNLP/hateBERT"
    NUM_CLASSES = 2
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    GPU_ID = 4
    model = CustomHateBERTModel(pretrained_model_name, NUM_CLASSES)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print("Creating dataset")
    ai_df = create_binary_dataset()
    if sub_df:
        ai_df = ai_df.sample(frac=FRAC)

    print("Creating PyTorch dataset")
    text_ds = create_text_classification_dataset(
        ai_df, tokenizer_name=pretrained_model_name, max_len=MAX_LEN
    )

    print("Creating data loader")
    data_loader = DataLoader(text_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Training configuration
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)

    print("Starting training loop")
    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1score = train_model(
            model, data_loader, optimizer, criterion, GPU_ID
        )
        print(f"Epoch {epoch}/{EPOCHS}")
        print(
            f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1-Score: {train_f1score:.4f}"
        )

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("F1-Score/train", train_f1score, epoch)

        # Save checkpoints every CHECKPOINT_INTERVAL epochs
        if epoch % CHECKPOINT_INTERVAL == 0:
            # save_checkpoint(model, optimizer, epoch, save_dir=CHECKPOINT_DIR)
            best_f1 = save_checkpoint(
                model, optimizer, epoch, train_f1score, best_f1, save_dir=CHECKPOINT_DIR
            )

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main(sub_df=False)
