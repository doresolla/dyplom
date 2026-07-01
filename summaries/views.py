from __future__ import annotations

import shutil
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import PasswordChangeView
from django.core.files import File
from django.db.models import Count, Q
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.views.decorators.http import require_POST

from .forms import AccountEmailForm, AccountPasswordChangeForm, LectureNoteTagsForm, LectureNoteUploadForm, ReviewForm, UserRegistrationForm
from .models import LectureNote, Review, Tag
from .services.filenames import make_safe_filename_stem
from .services.markdown_pdf import generate_pdf_from_markdown
from .tasks import process_lecture_note


def home(request):
    return redirect('dashboard' if request.user.is_authenticated else 'login')


def register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    form = UserRegistrationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, 'Аккаунт создан. Загрузите первую видеолекцию.')
        return redirect('dashboard')
    return render(request, 'registration/register.html', {'form': form})


def _private_library(user):
    """Единственный пользовательский источник конспектов: личная библиотека владельца."""
    return user.lecture_notes.prefetch_related('tags')


def _owned_note_or_404(request, pk: int):
    return get_object_or_404(_private_library(request.user), pk=pk)


def _library_filters(request, notes):
    selected_tag_id = request.GET.get('tag', '').strip()
    query = request.GET.get('q', '').strip()
    if selected_tag_id.isdigit():
        notes = notes.filter(tags__id=int(selected_tag_id))
    else:
        selected_tag_id = ''
    if query:
        notes = notes.filter(Q(title__icontains=query) | Q(tags__name__icontains=query))
    return notes.distinct(), selected_tag_id, query


@login_required
def dashboard(request):
    library = _private_library(request.user)
    stats = library.aggregate(
        total=Count('id'),
        ready=Count('id', filter=Q(status=LectureNote.Status.DONE)),
        processing=Count('id', filter=Q(status__in=[LectureNote.Status.QUEUED, LectureNote.Status.PROCESSING])),
        favorite=Count('id', filter=Q(is_favorite=True)),
    )
    notes, selected_tag_id, query = _library_filters(request, library)
    return render(request, 'summaries/dashboard.html', {
        'notes': notes,
        'stats': stats,
        'available_tags': Tag.objects.filter(is_active=True).order_by('name'),
        'selected_tag_id': selected_tag_id,
        'query': query,
    })


@login_required
def favorites(request):
    notes, selected_tag_id, query = _library_filters(request, _private_library(request.user).filter(is_favorite=True))
    return render(request, 'summaries/favorites.html', {
        'notes': notes,
        'available_tags': Tag.objects.filter(is_active=True).order_by('name'),
        'selected_tag_id': selected_tag_id,
        'query': query,
    })


@login_required
def upload_note(request):
    form = LectureNoteUploadForm(request.POST or None, request.FILES or None, initial={'llm_model': 'qwen'})
    if request.method == 'POST' and form.is_valid():
        note = form.save(commit=False)
        note.owner = request.user
        note.video_file_original_name = request.FILES['video_file'].name
        note.status = LectureNote.Status.QUEUED
        note.progress = 0
        note.status_message = 'Задача поставлена в очередь'
        note.save()
        form.save_m2m()
        process_lecture_note.delay(note.id)
        messages.success(request, 'Видео принято. Обработка началась в фоне, а исходный файл будет удалён после её завершения.')
        return redirect('dashboard')
    return render(request, 'summaries/upload_note.html', {'form': form})


@login_required
def note_detail(request, pk: int):
    note = _owned_note_or_404(request, pk)

    if note.status == LectureNote.Status.DONE and note.summary_file and not note.summary_pdf_file:
        try:
            _ensure_pdf_file_for_note(note)
            note.refresh_from_db()
        except Exception as exc:
            messages.warning(request, f'PDF-конспект пока не удалось сформировать: {exc}')

    review = Review.objects.filter(note=note, user=request.user).first()
    return render(request, 'summaries/note_detail.html', {
        'note': note,
        'tags_form': LectureNoteTagsForm(instance=note),
        'review': review,
        'review_form': ReviewForm(instance=review),
    })


@login_required
@require_POST
def update_note_tags(request, pk: int):
    note = _owned_note_or_404(request, pk)
    form = LectureNoteTagsForm(request.POST, instance=note)
    if form.is_valid():
        form.save()
        messages.success(request, 'Теги сохранены.')
    else:
        messages.error(request, 'Не удалось сохранить теги.')
    return redirect('note_detail', pk=note.pk)


@login_required
@require_POST
def toggle_favorite(request, pk: int):
    note = _owned_note_or_404(request, pk)
    note.is_favorite = not note.is_favorite
    note.save(update_fields=['is_favorite', 'updated_at'])
    messages.success(request, 'Конспект добавлен в избранное.' if note.is_favorite else 'Конспект удалён из избранного.')
    destination = request.POST.get('destination')
    if destination == 'favorites':
        return redirect('favorites')
    if destination == 'dashboard':
        return redirect('dashboard')
    return redirect('note_detail', pk=note.pk)


@login_required
@require_POST
def save_review(request, pk: int):
    note = _owned_note_or_404(request, pk)
    if note.status != LectureNote.Status.DONE:
        messages.error(request, 'Отзыв можно оставить только после завершения обработки.')
        return redirect('note_detail', pk=note.pk)
    current = Review.objects.filter(note=note, user=request.user).first()
    form = ReviewForm(request.POST, instance=current)
    if form.is_valid():
        review = form.save(commit=False)
        review.note = note
        review.user = request.user
        review.save()
        messages.success(request, 'Отзыв сохранён.' if current is None else 'Отзыв обновлён.')
    else:
        messages.error(request, 'Проверьте данные отзыва.')
    return redirect('note_detail', pk=note.pk)


@login_required
@require_POST
def delete_review(request, pk: int):
    note = _owned_note_or_404(request, pk)
    deleted, _ = Review.objects.filter(note=note, user=request.user).delete()
    messages.success(request, 'Отзыв удалён.' if deleted else 'Отзыв уже отсутствует.')
    return redirect('note_detail', pk=note.pk)


@login_required
def delete_note(request, pk: int):
    note = _owned_note_or_404(request, pk)
    if note.status in {LectureNote.Status.QUEUED, LectureNote.Status.PROCESSING}:
        messages.warning(request, 'Нельзя удалить конспект, пока выполняется обработка. Дождитесь завершения или ошибки.')
        return redirect('note_detail', pk=note.pk)
    if request.method == 'POST':
        title = note.title
        _delete_note_artifacts(note)
        note.delete()
        messages.success(request, f'Конспект «{title}» удалён.')
        return redirect('dashboard')
    return render(request, 'summaries/note_confirm_delete.html', {'note': note})


@login_required
def note_file(request, pk: int, file_kind: str):
    note = _owned_note_or_404(request, pk)
    file_map = {
        'pdf': ('summary_pdf_file', 'application/pdf'),
        'markdown': ('summary_file', 'text/markdown; charset=utf-8'),
        'transcript': ('transcript_file', 'text/plain; charset=utf-8'),
    }
    if file_kind not in file_map:
        raise Http404('Неизвестный тип файла')
    field_name, content_type = file_map[file_kind]
    file_field = getattr(note, field_name)
    if not file_field:
        raise Http404('Файл не найден')
    try:
        return FileResponse(
            file_field.open('rb'),
            content_type=content_type,
            as_attachment=request.GET.get('download') == '1' or file_kind != 'pdf',
            filename=Path(file_field.name).name,
        )
    except FileNotFoundError as exc:
        raise Http404('Файл отсутствует на диске') from exc


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
    note = _owned_note_or_404(request, pk)
    return render(request, 'summaries/processing_logs.html', {'note': note, 'logs': note.logs.all()})


@login_required
def account_profile(request):
    form = AccountEmailForm(request.POST or None, instance=request.user)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, 'Email для уведомлений обновлён.')
        return redirect('account_profile')
    return render(request, 'registration/profile.html', {
        'form': form,
        'note_count': request.user.lecture_notes.count(),
        'favorite_count': request.user.lecture_notes.filter(is_favorite=True).count(),
    })


class AccountPasswordChangeView(LoginRequiredMixin, PasswordChangeView):
    form_class = AccountPasswordChangeForm
    template_name = 'registration/password_change.html'
    success_url = reverse_lazy('account_profile')

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, 'Пароль успешно изменён.')
        return response


def _delete_note_artifacts(note: LectureNote) -> None:
    for field_name in ('video_file', 'summary_file', 'summary_pdf_file', 'transcript_file'):
        stored_file = getattr(note, field_name)
        if stored_file:
            stored_file.delete(save=False)
    if not note.result_dir:
        return
    result_path = Path(note.result_dir).resolve()
    private_runs_root = (Path(settings.MEDIA_ROOT) / 'runs' / str(note.owner_id)).resolve()
    try:
        result_path.relative_to(private_runs_root)
    except ValueError:
        return
    if result_path.is_dir():
        shutil.rmtree(result_path, ignore_errors=True)


def _ensure_pdf_file_for_note(note: LectureNote) -> None:
    source_markdown = _get_original_markdown_path(note)
    if source_markdown is None or not source_markdown.exists():
        raise RuntimeError('не найден markdown-файл для конвертации')
    output_dir = Path(note.result_dir) if note.result_dir else source_markdown.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_basename = make_safe_filename_stem(note.title, fallback=f'note_{note.id}')
    pdf_path = output_dir / f'{summary_basename}.pdf'
    generate_pdf_from_markdown(
        markdown_path=source_markdown,
        pdf_path=pdf_path,
        title=f'Конспект: {note.title}',
        log=lambda message: note.logs.create(message=message),
    )
    with pdf_path.open('rb') as file_object:
        note.summary_pdf_file.save(f'summaries/{note.id}/{pdf_path.name}', File(file_object), save=True)


def _get_original_markdown_path(note: LectureNote) -> Path | None:
    meta_path = (note.pipeline_meta or {}).get('summary_path')
    if meta_path and Path(meta_path).exists():
        return Path(meta_path)
    try:
        return Path(note.summary_file.path) if note.summary_file else None
    except Exception:
        return None
