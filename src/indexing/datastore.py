import logging
import re
import os
import shutil
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import json

from ..preprocessing import Document, Page


class DataStore:
    def __init__(self, store_dir: str, logger: logging.Logger):
        self.logger = logger
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.logger.debug(f"Data store directory is ready: {self.store_dir}")

    def add(self, document: Document) -> None:
        doc_id = self._get_doc_id(document)
        document.doc_id = doc_id
        save_path = self._create_doc_path(doc_id)
        if not save_path:
            self.logger.info(f'File {doc_id}.json already exists')
        else:
            self._save_document(document, save_path)            
        
    def read(self, path: Optional[str] = None, doc_id: Optional[str] = None) -> Optional[Document]:
        if path:
            document = self._read_json(path)
        elif doc_id:
            path = self.doc_id2path(doc_id)
            document = self._read_json(path)
        else:
            self.logger.error('Need file path or document id')
            document = None
        return self._dict_to_document(document)
    
    def delete(self, path: Optional[str] = None, doc_id: Optional[str] = None) -> None:
        if path:
            self._delete_document(path)
        elif doc_id:
            path = self.doc_id2path(doc_id)
            self._delete_document(path)
        else:
            self.logger.error('Need file path or document id')            

    def clear(self) -> None:
        for name in os.listdir(self.store_dir):
            path = os.path.join(self.store_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        self.logger.info(f"DataStore data removed from {self.store_dir}")
    
    def doc_id2path(self, doc_id: str) -> Optional[str]:
        path = os.path.join(self.store_dir, f'{doc_id}.json')
        return path if os.path.exists(path) else None
    
    def path2doc_id(self, path: str) -> Optional[str]:
        doc_id = None
        if path:
            if os.path.exists(path):
                doc_id = path.split('/')[-1].split('.')[0]
            else:
                path = os.path.join(self.store_dir, path)
                if os.path.exists(path):
                    doc_id = path.split('/')[-1].split('.')[0]
        return doc_id
    
    def get_list_doc_id(self) -> List[str]:
        doc_ids = []
        for path in self.get_list_doc_path():
            document = self.read(path)
            if not document.doc_id:
                self.logger.error(f'File {path} without doc_id')
            else:
                doc_ids.append(document.doc_id)
        self.logger.debug(f'{len(doc_ids)} files were read')
        return doc_ids
        
    def get_list_doc_path(self) -> List[str]:
        files_names = os.listdir(self.store_dir)
        paths = [os.path.join(self.store_dir, file_name) for file_name in files_names]
        self.logger.debug(f'{len(paths)} files were found')
        return paths

    def info(self) -> None:
        paths = self.get_list_doc_path()
        total_bytes = sum(os.path.getsize(path) for path in paths if os.path.exists(path))
        total_mb = total_bytes / (1024 * 1024)
        print(f'{len(paths)} files, {total_mb:.2f} MB total')

    def _dict_to_document(self, document: Optional[Dict[str, Any]]) -> Optional[Document]:
        if not document:
            return None
        source_path = document.get('source_path')
        doc_id = document.get('doc_id')
        if not 'pages' in document:
            self.logger.error(f'Document source_path={source_path} doc_id={doc_id} without pages')
            pages = []
        else:
            pages = document['pages']
            pages = [Page(number=page['number'], text=page['text']) for page in pages]
        return Document(
            source_path=source_path,
            doc_id=doc_id,
            pages=pages
        )
         
    def _save_document(self, document: Document, path: str) -> None:
        try:
            payload = asdict(document)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            self.logger.debug(f'Successfully save {path}')
        except Exception as e:
            self.logger.error(f'Save {path} error:\n{e}')
            
    def _read_json(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.logger.debug(f'Successfully read {path}')
            return payload
        except Exception as e:
            self.logger.error('Read {path} error:\n{e}')

    def _delete_document(self, path: str) -> None:
        if path and os.path.exists(path):
            os.remove(path)
            self.logger.info(f'File {path} deleted')
        else:
            self.logger.error(f'File {path} not exists')

    def _create_doc_path(self, doc_id: str) -> Optional[str]:
        path = os.path.join(self.store_dir, f'{doc_id}.json')
        return path if not os.path.exists(path) else None
        
    def _get_doc_id(self, document: Document) -> str:
        pages = document.pages
        doc_id = None
        idx = 0
        while not doc_id and idx < len(pages):
            text = pages[idx].text
            if len(text):
                payload = text[:1000].encode()
                hash = hashlib.sha256(payload)
                doc_id = hash.hexdigest()
                return doc_id
            idx += 1
        doc_id = uuid.uuid4().hex
        return doc_id
