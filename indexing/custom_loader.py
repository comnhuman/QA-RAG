import os
import tempfile
import subprocess
from pathlib import Path
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import BSHTMLLoader

class HWPLoader(BSHTMLLoader):
    def __init__(self, file_path: str) -> None:
        self.original_file_path = file_path
        
        xml_text = subprocess.check_output(
            ["hwp5proc", "xml", file_path],
            text=True,
            stderr=subprocess.DEVNULL,
        )

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".xml", mode="w", encoding="utf-8"
        ) as tmp:
            tmp.write(xml_text)
            self._temp_path = tmp.name

        super().__init__(file_path=self._temp_path, bs_kwargs={"features": "xml"})

    def load(self):
        docs = super().load()

        try:
            os.remove(self._temp_path)
        except FileNotFoundError:
            pass

        return docs

    def lazy_load(self) -> Iterator[Document]:
        for doc in super().lazy_load():
            doc.metadata["source"] = f"{Path(self.original_file_path)}"
            yield doc