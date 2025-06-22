from django.test import TestCase
from website.text import Text
import os

class TextProcessingTest(TestCase):
    def test_tokenize(self):
        test_file = 'tests/test_data/sample.txt'
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Это пример текста. Здесь два предложения.")
        text_obj = Text(abs_name=test_file, video_id=1)
        text_obj.tokenize()
        self.assertGreater(len(text_obj.sentences), 0)
