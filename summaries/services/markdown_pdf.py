from __future__ import annotations

import html
import os
import re
import traceback
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse


_IMAGE_LINK_RE = re.compile(r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)')
_HTML_IMG_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])(?P<src>[^"\']+)(["\'][^>]*>)', re.IGNORECASE)
_HEADING_RE = re.compile(r'^(?P<level>#{1,6})\s+(?P<text>.+?)\s*$')
_UNORDERED_RE = re.compile(r'^\s*[-*+]\s+(?P<text>.+?)\s*$')
_ORDERED_RE = re.compile(r'^\s*\d+[.)]\s+(?P<text>.+?)\s*$')


def generate_pdf_from_markdown(
    markdown_path: Path,
    pdf_path: Path,
    title: str = '',
    log: Callable[[str], None] | None = None,
) -> Path:
    """
    Преобразует markdown-конспект в PDF и сохраняет изображения внутри PDF.

    Основной способ — WeasyPrint, потому что он лучше всего сохраняет HTML/CSS-верстку.
    Но на Windows импорт WeasyPrint часто падает не из-за pip-пакета, а из-за системных
    библиотек Pango/Cairo/GDK-PixBuf. Поэтому добавлен запасной вариант на ReportLab:
    он проще по оформлению, зато не требует GTK/MSYS2 и позволяет не ронять весь пайплайн.
    """
    markdown_path = Path(markdown_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if log:
        log(f'[pdf] исходный markdown: {markdown_path}')

    try:
        return _generate_pdf_with_weasyprint(
            markdown_path=markdown_path,
            pdf_path=pdf_path,
            title=title,
            log=log,
        )
    except Exception as exc:
        if log:
            log('[pdf] WeasyPrint недоступен или завершился с ошибкой. Использую fallback ReportLab.')
            log(f'[pdf] причина WeasyPrint: {type(exc).__name__}: {exc}')
            log(traceback.format_exc())

    try:
        return _generate_pdf_with_reportlab(
            markdown_path=markdown_path,
            pdf_path=pdf_path,
            title=title,
            log=log,
        )
    except Exception as exc:
        raise RuntimeError(
            'Не удалось сформировать PDF ни через WeasyPrint, ни через ReportLab. '
            f'Последняя ошибка: {type(exc).__name__}: {exc}'
        ) from exc


def _generate_pdf_with_weasyprint(
    markdown_path: Path,
    pdf_path: Path,
    title: str = '',
    log: Callable[[str], None] | None = None,
) -> Path:
    import markdown
    from weasyprint import CSS, HTML

    md_text = markdown_path.read_text(encoding='utf-8')
    md_text = _normalize_markdown_image_links(md_text, markdown_path.parent, log=log)

    body_html = markdown.markdown(
        md_text,
        extensions=['extra', 'toc', 'tables', 'nl2br', 'sane_lists'],
        output_format='html5',
    )
    body_html = _normalize_html_image_links(body_html, markdown_path.parent, log=log)

    document_html = _build_pdf_html(body_html=body_html, title=title or markdown_path.stem)

    css = CSS(string=_pdf_css())
    HTML(string=document_html, base_url=markdown_path.parent.as_uri()).write_pdf(
        target=str(pdf_path),
        stylesheets=[css],
    )

    if log:
        log(f'[pdf] PDF-конспект сохранен через WeasyPrint: {pdf_path}')

    return pdf_path


def _generate_pdf_with_reportlab(
    markdown_path: Path,
    pdf_path: Path,
    title: str = '',
    log: Callable[[str], None] | None = None,
) -> Path:
    """
    Упрощенный генератор PDF без WeasyPrint.

    Поддерживает основные элементы, которые обычно есть в конспекте:
    - заголовки #, ##, ###;
    - обычные абзацы;
    - маркированные и нумерованные списки;
    - fenced code blocks;
    - markdown-изображения ![](path) с локальными путями.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer

    font_regular, font_bold = _register_reportlab_fonts()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='RuBody',
        parent=styles['BodyText'],
        fontName=font_regular,
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name='RuTitle',
        parent=styles['Title'],
        fontName=font_bold,
        fontSize=20,
        leading=24,
        spaceAfter=12,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name='RuH1',
        parent=styles['Heading1'],
        fontName=font_bold,
        fontSize=18,
        leading=22,
        spaceBefore=8,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='RuH2',
        parent=styles['Heading2'],
        fontName=font_bold,
        fontSize=15,
        leading=19,
        spaceBefore=8,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name='RuH3',
        parent=styles['Heading3'],
        fontName=font_bold,
        fontSize=12.5,
        leading=16,
        spaceBefore=6,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name='RuCode',
        parent=styles['Code'],
        fontName=font_regular,
        fontSize=8.5,
        leading=11,
        backColor=colors.whitesmoke,
        borderColor=colors.lightgrey,
        borderWidth=0.3,
        borderPadding=5,
        spaceAfter=8,
    ))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title=title or markdown_path.stem,
        author='Django lecture notes app',
    )

    story = []
    if title:
        story.append(Paragraph(_clean_inline_markdown(title), styles['RuTitle']))
        story.append(Spacer(1, 4 * mm))

    story.extend(_markdown_to_reportlab_flowables(
        markdown_text=markdown_path.read_text(encoding='utf-8'),
        base_dir=markdown_path.parent,
        styles=styles,
        log=log,
    ))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont(font_regular, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 9 * mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    if log:
        log(f'[pdf] PDF-конспект сохранен через ReportLab: {pdf_path}')

    return pdf_path


def _markdown_to_reportlab_flowables(markdown_text: str, base_dir: Path, styles, log=None):
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, Preformatted, Spacer

    story = []
    paragraph_buffer: list[str] = []
    list_buffer: list[tuple[str, str]] = []
    code_buffer: list[str] = []
    in_code = False

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            text = ' '.join(part.strip() for part in paragraph_buffer if part.strip())
            if text:
                story.append(Paragraph(_clean_inline_markdown(text), styles['RuBody']))
            paragraph_buffer = []

    def flush_list():
        nonlocal list_buffer
        if list_buffer:
            items = [ListItem(Paragraph(_clean_inline_markdown(text), styles['RuBody'])) for _, text in list_buffer]
            bullet_type = 'bullet' if list_buffer[0][0] == 'ul' else '1'
            story.append(ListFlowable(items, bulletType=bullet_type, leftIndent=14 * mm))
            list_buffer = []

    def flush_code():
        nonlocal code_buffer
        if code_buffer:
            story.append(Preformatted('\n'.join(code_buffer), styles['RuCode']))
            code_buffer = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip('\n')

        if line.strip().startswith('```'):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                flush_list()
                in_code = True
            continue

        if in_code:
            code_buffer.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        image_match = _IMAGE_LINK_RE.search(line.strip())
        if image_match:
            flush_paragraph()
            flush_list()
            alt = image_match.group('alt').strip()
            src = image_match.group('src').strip()
            image_path = _resolve_image_path(src, base_dir)
            if image_path and image_path.exists():
                img = _reportlab_image(image_path, max_width=170 * mm, max_height=140 * mm, log=log)
                if img is not None:
                    story.append(img)
                    story.append(Spacer(1, 3 * mm))
                    if alt:
                        story.append(Paragraph(_clean_inline_markdown(alt), styles['RuBody']))
            elif log:
                log(f'[pdf] предупреждение: изображение не найдено для ReportLab: {src}')
            rest = _IMAGE_LINK_RE.sub('', line).strip()
            if rest:
                paragraph_buffer.append(rest)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group('level'))
            text = heading_match.group('text')
            if level == 1:
                style = styles['RuH1']
            elif level == 2:
                style = styles['RuH2']
            else:
                style = styles['RuH3']
            story.append(Paragraph(_clean_inline_markdown(text), style))
            continue

        ul_match = _UNORDERED_RE.match(line)
        if ul_match:
            flush_paragraph()
            if list_buffer and list_buffer[0][0] != 'ul':
                flush_list()
            list_buffer.append(('ul', ul_match.group('text')))
            continue

        ol_match = _ORDERED_RE.match(line)
        if ol_match:
            flush_paragraph()
            if list_buffer and list_buffer[0][0] != 'ol':
                flush_list()
            list_buffer.append(('ol', ol_match.group('text')))
            continue

        paragraph_buffer.append(line)

    flush_code()
    flush_paragraph()
    flush_list()
    return story


def _register_reportlab_fonts() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates_regular: list[Path] = []
    candidates_bold: list[Path] = []

    windir = os.environ.get('WINDIR') or os.environ.get('SystemRoot')
    if windir:
        fonts_dir = Path(windir) / 'Fonts'
        candidates_regular.extend([fonts_dir / 'arial.ttf', fonts_dir / 'calibri.ttf', fonts_dir / 'segoeui.ttf'])
        candidates_bold.extend([fonts_dir / 'arialbd.ttf', fonts_dir / 'calibrib.ttf', fonts_dir / 'segoeuib.ttf'])

    candidates_regular.extend([
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'),
        Path('/Library/Fonts/Arial Unicode.ttf'),
        Path('/Library/Fonts/Arial.ttf'),
    ])
    candidates_bold.extend([
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
        Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'),
        Path('/Library/Fonts/Arial Bold.ttf'),
    ])

    regular_path = next((path for path in candidates_regular if path.exists()), None)
    bold_path = next((path for path in candidates_bold if path.exists()), None)

    if regular_path:
        pdfmetrics.registerFont(TTFont('LectureNotesRegular', str(regular_path)))
        regular_name = 'LectureNotesRegular'
    else:
        regular_name = 'Helvetica'

    if bold_path:
        pdfmetrics.registerFont(TTFont('LectureNotesBold', str(bold_path)))
        bold_name = 'LectureNotesBold'
    elif regular_path:
        bold_name = regular_name
    else:
        bold_name = 'Helvetica-Bold'

    return regular_name, bold_name


def _reportlab_image(image_path: Path, max_width, max_height, log=None):
    from PIL import Image as PILImage
    from reportlab.platypus import Image

    try:
        with PILImage.open(image_path) as pil_img:
            width_px, height_px = pil_img.size
        if width_px <= 0 or height_px <= 0:
            return None
        scale = min(max_width / width_px, max_height / height_px, 1.0)
        return Image(str(image_path), width=width_px * scale, height=height_px * scale)
    except Exception as exc:
        if log:
            log(f'[pdf] предупреждение: не удалось вставить изображение {image_path}: {exc}')
        return None


def _clean_inline_markdown(text: str) -> str:
    """Минимальная очистка inline markdown для ReportLab Paragraph."""
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font name="Courier">\1</font>', text)
    text = re.sub(r'\[(?P<label>[^\]]+)\]\((?P<url>[^)]+)\)', r'\g<label>', text)
    return text


def _resolve_image_path(src: str, base_dir: Path) -> Path | None:
    src = src.strip().strip('<>').strip().strip('"\'')
    if not src:
        return None

    parsed = urlparse(src)
    if parsed.scheme == 'file':
        return Path(unquote(parsed.path))
    if parsed.scheme in {'http', 'https', 'data'}:
        return None

    # Markdown допускает ![](path "title"). Берем только путь.
    if ' "' in src:
        src = src.split(' "', 1)[0]
    elif " '" in src:
        src = src.split(" '", 1)[0]

    raw_path = unquote(src).replace('\\', '/')
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / raw_path
    return path.resolve()


def _normalize_markdown_image_links(
    md_text: str,
    base_dir: Path,
    log: Callable[[str], None] | None = None,
) -> str:
    def repl(match: re.Match[str]) -> str:
        alt = match.group('alt')
        src = match.group('src').strip()
        normalized_src = _normalize_image_src(src, base_dir, log=log)
        return f'![{alt}]({normalized_src})'

    return _IMAGE_LINK_RE.sub(repl, md_text)


def _normalize_html_image_links(
    html_text: str,
    base_dir: Path,
    log: Callable[[str], None] | None = None,
) -> str:
    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        src = match.group('src')
        suffix = match.group(3)
        normalized_src = html.escape(_normalize_image_src(src, base_dir, log=log), quote=True)
        return f'{prefix}{normalized_src}{suffix}'

    return _HTML_IMG_RE.sub(repl, html_text)


def _normalize_image_src(
    src: str,
    base_dir: Path,
    log: Callable[[str], None] | None = None,
) -> str:
    """Преобразует локальный путь изображения в file:// URI, пригодный для WeasyPrint."""
    src = src.strip()
    if not src:
        return src

    src_without_angle = src.strip('<>')
    parsed = urlparse(src_without_angle)
    if parsed.scheme in {'http', 'https', 'data', 'file'}:
        return src_without_angle

    raw_path = unquote(src_without_angle).strip().strip('"\'')
    if ' "' in raw_path:
        raw_path = raw_path.split(' "', 1)[0]
    elif " '" in raw_path:
        raw_path = raw_path.split(" '", 1)[0]
    raw_path = raw_path.replace('\\', '/')

    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = base_dir / raw_path

    if image_path.exists():
        return image_path.resolve().as_uri()

    if log:
        log(f'[pdf] предупреждение: изображение не найдено: {image_path}')
    return src_without_angle


def _build_pdf_html(body_html: str, title: str) -> str:
    safe_title = html.escape(title)
    return f'''<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>{safe_title}</title>
</head>
<body>
    <main class="document">
        {body_html}
    </main>
</body>
</html>'''


def _pdf_css() -> str:
    return '''
@page {
    size: A4;
    margin: 18mm 16mm 20mm 16mm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
        color: #6b7280;
    }
}
* {
    box-sizing: border-box;
}
body {
    margin: 0;
    font-family: "DejaVu Sans", "Arial", sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #111827;
}
h1, h2, h3, h4 {
    line-height: 1.25;
    color: #111827;
    page-break-after: avoid;
}
h1 {
    font-size: 22pt;
    margin: 0 0 12pt;
}
h2 {
    font-size: 16pt;
    margin: 18pt 0 8pt;
    border-bottom: 1px solid #d1d5db;
    padding-bottom: 4pt;
}
h3 {
    font-size: 13pt;
    margin: 14pt 0 6pt;
}
p {
    margin: 0 0 8pt;
}
ul, ol {
    margin-top: 4pt;
    margin-bottom: 8pt;
}
li {
    margin-bottom: 3pt;
}
img {
    display: block;
    max-width: 100%;
    max-height: 150mm;
    height: auto;
    margin: 8pt auto 12pt;
    border: 1px solid #e5e7eb;
    border-radius: 6pt;
    page-break-inside: avoid;
}
figure {
    page-break-inside: avoid;
}
code, pre {
    font-family: "DejaVu Sans Mono", "Consolas", monospace;
    background: #f3f4f6;
}
code {
    padding: 1pt 3pt;
    border-radius: 3pt;
}
pre {
    padding: 8pt;
    border-radius: 6pt;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8pt 0 12pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #d1d5db;
    padding: 5pt 6pt;
    vertical-align: top;
}
th {
    background: #f3f4f6;
}
blockquote {
    margin: 8pt 0;
    padding: 6pt 10pt;
    border-left: 3pt solid #9ca3af;
    background: #f9fafb;
}
'''
