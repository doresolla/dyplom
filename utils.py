
from typing import Optional, Callable


def _emit(callback: Optional[Callable[[str], None]], text: str) -> None:
    if callback is not None:
        callback(text)