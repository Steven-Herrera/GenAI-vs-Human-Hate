"""Module for downloading datasets"""

import wget
import zipfile
from datasets import load_dataset

import loader_configs


def get_toxigen_data():
    # This is probably unfinished and not needed
    """Downloads the TOXIGEN data from HuggingFace and stores it
    in the data directory"""

    tg = load_dataset(
        "skg/toxigen-data",
        name=loader_configs.TOXIGEN_NAME,
        use_auth_token=True,
        cache_dir=loader_configs.TOXIGEN_PATH,
    )
    return tg


def get_implicit_hate_corpus():
    zip_filepath = wget.download(
        loader_configs.IMPLICIT_HATE_URL, out=loader_configs.IMPLICIT_HATE_PATH
    )
    # zip_filepath = os.path.join(loader_configs.IMPLICIT_HATE_PATH, filename)

    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(loader_configs.IMPLICIT_HATE_PATH)


if __name__ == "__main__":
    get_implicit_hate_corpus()
