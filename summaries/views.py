from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.core.files import File
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.safestring import mark_safe
from pathlib import Path
import markdown

from .forms import LectureNoteUploadForm, RatingForm, UserRegistrationForm
from .models import LectureNote, Notification, Rating
from .tasks import process_lecture_note
from .services.markdown_pdf import generate_pdf_from_markdown
from .services.filenames import make_safe_filename_stem


def register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Регистрация завершена. Теперь можно загрузить первую видеолекцию.')
            return redirect('dashboard')
    else:
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form': form})


def public_notes(request):
    query = request.GET.get('q', '').strip()
    notes = (
        LectureNote.objects.filter(status=LectureNote.Status.DONE, is_public=True)
        .select_related('owner')
        .annotate(avg_score=Avg('ratings__score'), ratings_total=Count('ratings'))
    )
    if query:
        notes = notes.filter(Q(title__icontains=query) | Q(summary_text__icontains=query))

    paginator = Paginator(notes, 10)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'summaries/public_notes.html', {'page_obj': page_obj, 'query': query})


@login_required
def dashboard(request):
    notes = request.user.lecture_notes.all().annotate(avg_score=Avg('ratings__score'), ratings_total=Count('ratings'))
    return render(request, 'summaries/dashboard.html', {'notes': notes})


@login_required
def upload_note(request):
    if request.method == 'POST':
        form = LectureNoteUploadForm(request.POST, request.FILES)
        if form.is_valid():
            note = form.save(commit=False)
            note.owner = request.user
            note.is_public = True
            note.status = LectureNote.Status.QUEUED
            note.progress = 0
            note.status_message = 'Задача поставлена в очередь'
            note.save()
            process_lecture_note.delay(note.id)
            messages.success(request, 'Видео загружено. Обработка запущена в фоне.')
            return redirect('dashboard')
    else:
        form = LectureNoteUploadForm(initial={'llm_model': 'qwen'})
    return render(request, 'summaries/upload_note.html', {'form': form})


def note_detail(request, pk: int):
    note = get_object_or_404(
        LectureNote.objects.select_related('owner').annotate(avg_score=Avg('ratings__score'), ratings_total=Count('ratings')),
        pk=pk,
    )

    if not note.is_public and (not request.user.is_authenticated or note.owner_id != request.user.id):
        messages.error(request, 'Нет доступа к этому конспекту.')
        return redirect('public_notes')

    if note.status == LectureNote.Status.DONE and note.summary_file and not note.summary_pdf_file:
        try:
            _ensure_pdf_file_for_note(note)
            note.refresh_from_db()
        except Exception as exc:
            messages.warning(request, f'PDF-конспект пока не удалось сформировать: {exc}')

    user_rating = None
    form = None
    if request.user.is_authenticated:
        user_rating = Rating.objects.filter(note=note, user=request.user).first()
        if request.method == 'POST' and note.status == LectureNote.Status.DONE:
            form = RatingForm(request.POST, instance=user_rating)
            if form.is_valid():
                rating = form.save(commit=False)
                rating.note = note
                rating.user = request.user
                rating.save()
                messages.success(request, 'Оценка сохранена.')
                return redirect('note_detail', pk=note.pk)
        else:
            form = RatingForm(instance=user_rating)

    summary_source = note.summary_text or _read_summary_file(note)
    html_summary = ''
    if summary_source:
        html_summary = mark_safe(markdown.markdown(summary_source, extensions=['extra', 'toc', 'tables', 'nl2br']))

    return render(
        request,
        'summaries/note_detail.html',
        {
            'note': note,
            'html_summary': html_summary,
            'rating_form': form,
            'user_rating': user_rating,
            'ratings': note.ratings.select_related('user')[:20],
        },
    )


def _read_summary_file(note: LectureNote) -> str:
    """Возвращает markdown-текст конспекта, даже если поле summary_text по какой-то причине пустое."""
    if not note.summary_file:
        return ''
    try:
        with note.summary_file.open('rb') as f:
            return f.read().decode('utf-8')
    except Exception:
        return ''


def _ensure_pdf_file_for_note(note: LectureNote) -> None:
    """Ленивая генерация PDF для старых конспектов, созданных до добавления PDF-поля."""
    source_markdown = _get_original_markdown_path(note)
    if source_markdown is None or not source_markdown.exists():
        raise RuntimeError('не найден markdown-файл для конвертации')

    if note.result_dir:
        output_dir = Path(note.result_dir)
    else:
        output_dir = source_markdown.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_basename = make_safe_filename_stem(note.title, fallback=f'note_{note.id}')
    pdf_path = output_dir / f'{summary_basename}.pdf'

    def log(message: str) -> None:
        note.logs.create(message=message)

    generate_pdf_from_markdown(
        markdown_path=source_markdown,
        pdf_path=pdf_path,
        title=f'Конспект: {note.title}',
        log=log,
    )

    with pdf_path.open('rb') as f:
        note.summary_pdf_file.save(f'summaries/{note.id}/{pdf_path.name}', File(f), save=True)


def _get_original_markdown_path(note: LectureNote) -> Path | None:
    meta_path = (note.pipeline_meta or {}).get('summary_path')
    if meta_path:
        path = Path(meta_path)
        if path.exists():
            return path

    try:
        if note.summary_file and note.summary_file.path:
            return Path(note.summary_file.path)
    except Exception:
        return None

    return None


@login_required
def notifications_list(request):
    notifications = request.user.notifications.select_related('note')
    if request.method == 'POST':
        notifications.filter(is_read=False).update(is_read=True)
        messages.success(request, 'Уведомления отмечены как прочитанные.')
        return redirect('notifications')
    return render(request, 'summaries/notifications.html', {'notifications': notifications})


@login_required
def processing_logs(request, pk: int):
    note = get_object_or_404(LectureNote, pk=pk, owner=request.user)
    return render(request, 'summaries/processing_logs.html', {'note': note, 'logs': note.logs.all()})
