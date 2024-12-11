"""A host of utility functions to help with miscellaneous tasks

Functions:
    words: counts the words in a sentence or returns the words
    get_hate_only_words: Words that exist in the hate corpus but not in the non-hate corpus
    get_vocab_size: Calculates |V| for hate or non-hate corpus
    doc_vocab_size: Calculates |V| for a single text
    pos_counts: Counts the POS in a string
    create_pos_columns: Creates a dataframe for the POS counts
    pos_word_counter: Counts of words of a given POS
"""

import nltk
import spacy
import pandas as pd
from collections import Counter
from tqdm.notebook import tqdm


def words(sentence, count=True):
    """Uses nltk to tokenize then count the number of words
    in a sentence

    Args:
        sentence (str): AI generated text
        count (bool): Whether to return the word count instead of words

    Returns:
        words (List[str]): The tokenized words
        word_count (int): Number of words in the text
    """
    words = nltk.word_tokenize(sentence)
    if count:
        word_count = len(words)
        return word_count
    else:
        return words


def get_hate_only_words(
    df, text_col="text", label_col="label", map={"non-hate": 0, "hate": 1}
):
    """
    Returns a dictionary of words that exist in hate texts but not in non-hate texts.
    The dictionary also contains the word counts of those words.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing text and labels.
        text_col (str): The name of the column containing text data.
        label_col (str): The name of the column containing labels ('hate' or 'non-hate').

    Returns:
        hate_only_words (Dict[str, int]): A dictionary with words as keys and their counts in hate texts as values.
    """
    # Separate hate and non-hate texts
    hate_texts = df[df[label_col] == map["hate"]][text_col].str.cat(sep=" ")
    non_hate_texts = df[df[label_col] == map["non-hate"]][text_col].str.cat(sep=" ")

    # Tokenize the texts using nltk.word_tokenize
    hate_words = Counter(nltk.word_tokenize(hate_texts))
    non_hate_words = set(nltk.word_tokenize(non_hate_texts))

    # Extract words exclusive to hate texts
    hate_only_words = Counter(
        {
            word: count
            for word, count in hate_words.items()
            if word not in non_hate_words
        }
    )
    hate_only_words = Counter(dict(hate_only_words.most_common()))
    # Return a sorted Counter object
    return hate_only_words


def get_vocab_size(
    df, text_col="sentence", label_col="label", label=0, return_vocab=False
):
    """Concatenate all of the documents in the text column that have a given label, tokenize the corpus,
    get the unique words, then calculate |V|.

    Args:
        df (pd.DataFrame): Contains the sentences and labels
        text_col (str): The column with sentences
        label_col (str): Column with labels
        return_vocab (bool): Whether to return the vocabulary V

    Returns:
        vocab_size (int): Number of words in V
        vocab (List[str]): The unique words
    """
    texts = df[df[label_col] == label][text_col].str.cat(sep=" ")
    vocab = set(nltk.word_tokenize(texts))
    vocab_size = len(vocab)
    if return_vocab:
        return (vocab_size, vocab)
    else:
        return vocab_size


def doc_vocab_size(sentence):
    """Calculates the number of unique words in a sentence

    Args:
        sentence (str): A sentence from the corpus

    Returns:
        vocab_size (int): The number of unique words in the sentence
    """
    vocab = set(nltk.word_tokenize(sentence))
    vocab_size = len(vocab)
    return vocab_size


def pos_counts(text, nlp):
    """
    Accepts a string, tags parts of speech using SpaCy,
    and returns a dictionary of all possible POS tags with their counts.

    Args:
        text (str): The input text to analyze.
        nlp (spacy.lang.en.English): A spacy language model

    Returns:
        dict: A dictionary with POS tags as keys and their counts as values.
    """

    # Create an empty dictionary for all possible POS tags with initial count 0
    pos_dict = {pos: 0 for pos in spacy.parts_of_speech.IDS.keys()}

    # Perform POS tagging
    doc = nlp(text)
    # Count each POS in the text
    for token in doc:
        pos_dict[token.pos_] += 1

    return pos_dict


def create_pos_columns(df, nlp, text_col="generation"):
    """Uses a spacy model to tag the parts of speech, counts the various POS, and creates a new df

    Args:
        df (pd.DataFrame): Contains a text column
        nlp (spacy.lang.en.English): A spacy POS tagger
        text_col (str): Column name with the text to tag

    Returns:
        df (pd.DataFrame): Contains original dataframe now with POS counts
    """
    df = df.reset_index()
    df["pos_counts"] = df[text_col].progress_apply(lambda x: pos_counts(x, nlp))
    pos_df = pd.json_normalize(df["pos_counts"])
    df = pd.concat([df, pos_df], axis=1)
    df = df.drop(columns=["pos_counts"])
    return df


def pos_word_counter(df, text_column, pos_name, nlp, group=None):
    """
    Perform POS tagging on each text in a DataFrame column, count tokens with a specific POS, and return a Counter.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text data.
        pos_name (str): The name of the target part-of-speech (e.g., "PRON", "VERB").
        nlp (spacy.language.Language): A SpaCy language model for POS tagging.

    Returns:
        pos_counter: A Counter dictionary with words of the specified POS and their counts in the corpus.
    """
    # Initialize the Counter
    pos_counter = Counter()
    if group is not None:
        df = df[df["group"] == group]

    # Process each text in the column
    for text in tqdm(df[text_column]):
        # Parse the text using SpaCy
        doc = nlp(text)

        # Count tokens with the specified POS
        for token in doc:
            if token.pos_ == pos_name:
                if token.pos_ in pos_counter:
                    pos_counter[token.text.casefold()] += (
                        1  # Convert to lowercase to normalize
                    )
                else:
                    pos_counter[token.text.casefold()] = 1

    return pos_counter
