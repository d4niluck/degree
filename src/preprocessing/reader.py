from typing import Optional, List
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from tqdm import tqdm
import re
import fitz # type: ignore
import logging

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
        if forced_ocr:
            document = self.ocr_reader.read(file_path)
            return document
        else:
            extension = self.get_file_extension(file_path)
            file_reader = self.get_file_reader(extension)
            document = file_reader.read(file_path)
        return document
        
    def get_file_reader(self, extension: str) -> Optional[FileReader]:
        file_reader = self.readers.get(extension)
        if file_reader:
            self.logger.info(f'Reader: {file_reader.__class__.__name__}')
        else:
            self.logger.error(f'Unexpected file extension: {extension}')
        return file_reader

    def get_file_extension(self, file_name: str) -> str:
        extension = file_name.split('.')[-1]
        return extension


class OCRReader:
    def __init__(self, logger: logging.Logger):
        self.ocr_model = None
        self.logger = logger
    
    def read(self, file_path: str) -> Optional[Document]:
        self._init_ocr_model()
        doc = fitz.open(file_path)
        pages = []
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
            self.logger.info('Init OCR model')
            self.ocr_model = PaddleOCR(
                ocr_version="PP-OCRv5",
                lang="ru",
                use_textline_orientation=False,
            )


class PDFReader:
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
            self.logger.error(f"Successfully read {file_path}")
        return document
            
    def read_scanned_pdf(self, file_path: str) -> Optional[Document]:
        raise NotImplementedError()
        
    def read_digital_pdf(self, file_path: str) -> Optional[Document]:
        doc = fitz.open(file_path)
        pages = []
        text_pages = 0
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = self._norm_text(page.get_text("text") or "")
            pages.append(Page(i+1, text))
            if len(text) >= 50:
                text_pages += 1
        
        text_pages_ratio = (text_pages / doc.page_count) if doc.page_count else 0.0
        self.logger.info(f'Text pages ratio: {text_pages_ratio}')
        
        if text_pages_ratio >= self.pages_ratio_threshold:
            document = Document(file_path, pages)
            return document
        else:
            self.logger.info('Need OCR for this pdf')
            return None
        
    def _norm_text(self, s: str) -> str:
        return self._ws.sub(" ", s).strip()


class DOCXReader:
    def __init__(self, logger: logging.Logger):
        ...
    
    def read(self, file_path: str) -> Optional[Document]:
        raise NotImplementedError()
