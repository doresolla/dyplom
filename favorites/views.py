from django.shortcuts import redirect, render, get_object_or_404
from website.models import Summary
from .favorites import Favorites

def favorites_add(request, summary_id):
    favorites = Favorites(request)
    favorites.add(summary_id)
    return redirect('favorites:favorites_detail')

def favorites_remove(request, summary_id):
    favorites = Favorites(request)
    favorites.remove(summary_id)
    return redirect('favorites:favorites_detail')

def favorites_detail(request):
    favorites = Favorites(request)
    summaries = favorites.get_summaries()

    for summary in summaries:
        summary.transcript_text = summary.audio.get_transcription_text()
        summary.summary_text = summary.get_file_text()

    return render(request, 'favorites_detail.html', {
        'summaries': summaries,
        'favorites': favorites,
    })
