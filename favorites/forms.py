from django import forms

class FavoritesAddForm(forms.Form):
    summary_id = forms.IntegerField(widget=forms.HiddenInput)