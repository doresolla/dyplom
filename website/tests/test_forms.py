from website.forms import UserRegistrationForm, VideoUploadForm
from django.test import TestCase
class UserRegistrationFormTest(TestCase):
    def test_valid_data(self):
        form_data = {
            'email': 'user1@example.com',
            'username': 'user1',
            'phone_number': '12345',
            'password': 'pass12345',
            'confirm_password': 'pass12345'
        }
        form = UserRegistrationForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_password_mismatch(self):
        form_data = {
            'email': 'user2@example.com',
            'username': 'user2',
            'phone_number': '12345',
            'password': 'pass123',
            'confirm_password': 'otherpass'
        }
        form = UserRegistrationForm(data=form_data)
        self.assertFalse(form.is_valid())

class VideoUploadFormTest(TestCase):
    def test_validation_empty(self):
        form = VideoUploadForm(data={})
        self.assertFalse(form.is_valid())
