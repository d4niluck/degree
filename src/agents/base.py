from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def answer(self, question: str) -> str:
        raise NotImplementedError
