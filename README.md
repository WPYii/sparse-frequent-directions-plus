# Frequent Directions Plus
Implementation and experiments for **Frequent Directions (FD)**, **Sparse Frequent Directions (SFD)**, and future improvements in matrix sketching.

## Getting Started

### 1. Dataset
Dataset: 20 Newsgroups. The dataset can be downloaded from http://qwone.com/~jason/20Newsgroups/. Please download the file 20news-bydate.tar.gz. Only train dataset is used

### 2. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc

Check that Poetry was installed successfully:
poetry --version
```

### 3. Configuration 
Configure the path in `application.yaml`.
Entry point: run the experiment with
python main.py