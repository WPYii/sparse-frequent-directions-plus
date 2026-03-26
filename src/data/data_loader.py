from pathlib import Path
from typing import List
from tqdm import tqdm


class DataLoader:
    def __init__(self, config):
        self.train_dir = Path(config.get("data", "decompressed", "train"))
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")

    def load_documents(self):
        documents: List[str] = []
        documents.extend(self._read_folder(self.train_dir))
        return documents

    def _read_folder(self, base_dir):
        docs: List[str] = []
        files = []
        for category_dir in base_dir.iterdir():
            if category_dir.is_dir():
                for file_path in category_dir.iterdir():
                    if file_path.is_file():
                        files.append(file_path)

        for file_path in tqdm(files, desc=f"Loading {base_dir.name}"):
            text = file_path.read_text(encoding="latin1", errors="ignore")
            docs.append(text)

        return docs