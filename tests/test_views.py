from django.test import TestCase
from django.urls import reverse

class RegisterUserViewTest(TestCase):
    def test_register_user_get(self):
        response = self.client.get(reverse('register_user'))
        self.assertEqual(response.status_code, 200)
