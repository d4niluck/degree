from typing import Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
import re
import fitz # type: ignore
import logging
import requests

from .schema import Document, Page


type FileReader = PDFReader | DOCXReader


class Reader:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.readers = self._init_readers()
        
    def _init_readers(self):
        pdf_reader = PDFReader(self.logger)
        docx_reader = DOCXReader(self.logger)
        readers = {
            'pdf': pdf_reader,
            'doc': docx_reader,
            'docx': docx_reader,
        }
        self.ocr_reader = OCRReader(self.logger)
        return readers
    
    def read(self, file_path: str, forced_ocr: bool = False) -> Optional[Document]:
        document = None
        extension = self.get_file_extension(file_path)
        if forced_ocr:
            assert extension == 'pdf' # TODO расширить для других расширений
            document = self.ocr_reader.read(file_path)
            return document
        else:
            file_reader = self.get_file_reader(extension)
            document = file_reader.read(file_path)
        return document
        
    def get_file_reader(self, extension: str) -> Optional[FileReader]:
        file_reader = self.readers.get(extension)
        if file_reader:
            self.logger.debug(f'Reader: {file_reader.__class__.__name__}')
        else:
            self.logger.error(f'Unexpected file extension: {extension}')
        return file_reader

    def get_file_extension(self, file_name: str) -> str:
        extension = file_name.split('.')[-1]
        return extension


class BaseReader:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def read(self, file_path: str) -> Optional[Document]:
        raise NotImplementedError()


class OCRReader(BaseReader):
    def __init__(self, logger: logging.Logger):
        self.ocr_model = None
        self.logger = logger
    
    def read(self, file_path: str) -> Optional[Document]:
        self._init_ocr_model()
        pages = []
        with fitz.open(file_path) as doc:
            for i in tqdm(range(doc.page_count)):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=150, colorspace=fitz.csRGB, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = self._ocr(img)
                pages.append(Page(i+1, text))
        document = Document(file_path, pages)
        return document

    def _ocr(self, img: Image) -> str:
        try:
            np_img = np.array(img)
            raw = self.ocr_model.predict(np_img)
            text = "\n".join(raw[0]['rec_texts']).strip()
        except Exception as e:
            self.logger.error(f'OCR img error {e}')
            text = ''
        return text

    def _init_ocr_model(self) -> None:
        if not self.ocr_model:
            try:
                from paddleocr import PaddleOCR
            except Exception as e:
                raise ImportError('Need paddleocr module -> pip install paddleocr')
            self.logger.info('Init OCR model')
            self.ocr_model = PaddleOCR(
                ocr_version="PP-OCRv5",
                lang="ru",
                use_textline_orientation=False,
            )


class PDFReader(BaseReader):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._ws = re.compile(r"\s+")
        self.pages_ratio_threshold = 0.6
        
    def read(self, file_path: str) -> Optional[Document]:
        document = None
        if not document:
            document = self.read_digital_pdf(file_path)
        if not document:
            document = self.read_scanned_pdf(file_path)
        if not document:
            self.logger.error(f"Couldn't read the file {file_path}")
        else:
            self.logger.debug(f"Successfully read {file_path}")
        return document
            
    def read_scanned_pdf(self, file_path: str) -> Optional[Document]:
        raise NotImplementedError()
        
    def read_digital_pdf(self, file_path: str) -> Optional[Document]:
        pages = []
        text_pages = 0
        with fitz.open(file_path) as doc:
            for i in range(doc.page_count):
                page = doc.load_page(i)
                text = self._norm_text(page.get_text("text") or "")
                pages.append(Page(i+1, text))
                if len(text) >= 50:
                    text_pages += 1
        
        total_pages = len(pages)
        text_pages_ratio = (text_pages / total_pages) if total_pages else 0.0
        self.logger.debug(f'Text pages ratio: {text_pages_ratio}')
        
        if text_pages_ratio >= self.pages_ratio_threshold:
            document = Document(file_path, pages)
            return document
        else:
            self.logger.info('Need OCR for this pdf')
            return None
        
    def _norm_text(self, s: str) -> str:
        return self._ws.sub(" ", s).strip()


class DOCXReader(BaseReader):
    def __init__(self, logger: logging.Logger):
        ...
    
    def read(self, file_path: str) -> Optional[Document]:
        raise NotImplementedError()


class HTTPReader:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8002",
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def read(self, file_path: str, forced_ocr: bool = False) -> Optional[Document]:
        response = self.session.post(
            f"{self.base_url}/reader/read",
            json={
                "file_path": file_path,
                "forced_ocr": forced_ocr,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        pages = [Page(number=page["number"], text=page["text"]) for page in payload["pages"]]
        return Document(
            source_path=payload["source_path"],
            pages=pages,
            doc_id=payload.get("doc_id"),
        )