import pytest
from typing import Optional
from src.preprocessing import Reader, Document
from pathlib import Path
import logging
import os


logger = logging.getLogger()
DATA_DIR = Path('data/raw_data')
if not DATA_DIR.exists() or not DATA_DIR.is_dir():
    raise ValueError()


@pytest.mark.parametrize(
    ('file_path'),
    [
        (str(DATA_DIR / '4293726411.pdf')),
        (str(DATA_DIR / '4293784693.pdf')),
        (str(DATA_DIR / '4293796604.pdf')),
    ]
)
def test_reader_with_pdf(file_path: str):
    reader = Reader(logger)
    text = reader.read(file_path)
    
    if not isinstance(text, Optional[Document]):
        raise 'Result must be Document type'
    

@pytest.mark.parametrize(
    ('file_path'),
    [
        (str(DATA_DIR / 'ocr_test.pdf')),
    ]
)
def test_reader_with_ocr(file_path: str):
    reader = Reader(logger)
    text = reader.read(file_path, forced_ocr=True)
    
    if not isinstance(text, Optional[Document]):
        raise 'Result must be Document type'
    