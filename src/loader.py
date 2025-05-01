# src/loader.py
import os
import requests
import fitz  # PyMuPDF
from tqdm.auto import tqdm

class DocumentLoader:
    """
    Loads a PDF document either from a local file or downloads it from a given URL.
    Parses each page to extract text and calculate various statistics.
    """

    def __init__(self, file_path, download_url=None):
        """
        Initializes the DocumentLoader.

        Args:
            file_path (str): Path to the local PDF file.
            download_url (str, optional): URL to download the PDF if not found locally.
        """
        self.file_path = file_path
        self.download_url = download_url

    def download(self):
        """
        Downloads the PDF from the URL if it doesn't exist locally.
        """
        if not os.path.exists(self.file_path):
            if self.download_url:
                print("[INFO] Downloading PDF...")
                response = requests.get(self.download_url)
                if response.status_code == 200:
                    with open(self.file_path, "wb") as f:
                        f.write(response.content)
                    print(f"[INFO] Downloaded and saved to {self.file_path}")
                else:
                    raise Exception("Failed to download the file")
            else:
                raise FileNotFoundError(f"File {self.file_path} not found and no URL provided.")

    def load(self):
        """
        Loads and parses the PDF into a list of dictionaries, each representing a page.

        Returns:
            list: A list of dictionaries containing page content and metadata.
        """
        self.download()
        doc = fitz.open(self.file_path)
        pages = []

        # Iterate through each page and extract structured data
        for page_number, page in tqdm(enumerate(doc), desc="Reading PDF"):
            text = page.get_text().replace("\n", " ").strip()
            pages.append({
                "page_number": page_number - 41,  # Adjust for nutrition PDF's actual numbering
                "page_char_count": len(text),
                "page_word_count": len(text.split()),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # Approximate token count
                "text": text
            })

        return pages