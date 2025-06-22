
from django.conf import settings
from website.models import Summary

class Favorites:
    def __init__(self, request):
        self.session = request.session
        fav = self.session.get(settings.FAVORITES_SESSION_ID)
        if not fav:
            fav = self.session[settings.FAVORITES_SESSION_ID] = []
        self.fav = fav

    def add(self, note_id):
        if note_id not in self.fav:
            self.fav.append(note_id)
            self.save()

    def remove(self, note_id):
        if note_id in self.fav:
            self.fav.remove(note_id)
            self.save()

    def save(self):
        self.session[settings.FAVORITES_SESSION_ID] = self.fav
        self.session.modified = True

    def clear(self):
        self.session[settings.FAVORITES_SESSION_ID] = []
        self.session.modified = True

    def get_notes(self):
        return Summary.objects.filter(note_id__in=self.fav)

    def __len__(self):
        return len(self.fav)
