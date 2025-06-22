from django import forms
from .models import ProcessingRequest

class ProcessingRequestForm(forms.ModelForm):
    class Meta:
        model = ProcessingRequest
        fields = ['algorithm', 'format', 'ratio']
        widgets = {
            'ratio': forms.NumberInput(attrs={'step': 0.1, 'min': 0.1, 'max': 1.0}),
        }
