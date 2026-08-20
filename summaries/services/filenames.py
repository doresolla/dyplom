from __future__ import annotations

import re


_WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
}


def make_safe_filename_stem(title: str, fallback: str = 'summary', max_length: int = 120) -> str:
    """
    Делает безопасную основу имени файла из пользовательского названия конспекта.

    Сохраняет кириллицу, пробелы и точки, чтобы файл назывался понятно:
        "Лекция 1. Вероятность" -> "Лекция 1. Вероятность"

    Заменяет только символы, которые ломают пути в Windows/Linux/macOS:
        < > : " / \\ | ? * и управляющие символы.
    """
    value = (title or '').strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', value)
    value = re.sub(r'\s+', ' ', value).strip()
    value = value.strip(' .')

    if not value:
        value = fallback

    if value.upper() in _WINDOWS_RESERVED_NAMES:
        value = f'{value}_file'

    if len(value) > max_length:
        value = value[:max_length].rstrip(' ._')

    return value or fallback
