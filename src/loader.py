# src/loader.py

import fitz  # PyMuPDF
import os
import requests
from typing import List

class DocumentLoader:
    """
    Loads a PDF document (downloads it if necessary) and extracts text page by page.
    """

    def __init__(self, file_path: str, download_url: str = None):
        self.file_path = file_path
        self.download_url = download_url

    def download_pdf(self):
        """
        Downloads the PDF file if it does not exist locally.
        """
        if not os.path.exists(self.file_path):
            if self.download_url:
                print(f"[INFO] File {self.file_path} not found. Downloading...")
                response = requests.get(self.download_url)
                if response.status_code == 200:
                    with open(self.file_path, "wb") as f:
                        f.write(response.content)
                    print(f"[INFO] File downloaded and saved as {self.file_path}.")
                else:
                    raise Exception(f"Failed to download file. Status code: {response.status_code}")
            else:
                raise FileNotFoundError(f"File {self.file_path} not found and no download URL provided.")

    def load(self) -> List[str]:
        """
        Loads the document and returns a list of text chunks, one per page.
        """
        try:
            self.download_pdf()  # Make sure file is available

            doc = fitz.open(self.file_path)
            pages_text = [page.get_text() for page in doc]
            doc.close()
            return pages_text
        except Exception as e:
            print(f"[ERROR] Failed to load document: {e}")
            return []
