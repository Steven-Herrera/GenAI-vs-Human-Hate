"""Configurations for downloading data"""

TOXIGEN_PATH = "../../../data/"
# specifies training data (250k) or human annotated data (27.45k)
TOXIGEN_NAME = "train"  # "annotated"
# downloading from huggingface requires an authorization token
HF_AUTH_TOKEN = ""

IMPLICIT_HATE_PATH = "../../../data"
IMPLICIT_HATE_URL = "https://www.dropbox.com/scl/fi/b1r7uukaj4zezzwsw72ox/implicit-hate-corpus.zip?rlkey=r9ylsgpt5mky9syip8rjvn46l&e=1&st=g4y1s03a&dl=0"
