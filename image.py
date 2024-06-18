## coding='utf-8'
import subprocess

# def extract_image(video):
#     width = 1920
#     height = 1080
#     # filename, ext = os.path.splitext(self.abs_filename)
#     print("Имя видео файла", video + '.mp4')
#     s = "eq(pict_type\\,PICT_TYPE_I)"
#     command = ["ffmpeg", "-y", "-i", video + '.mp4', "-vsync", "0", "-vf", f"select={s}", f"-s",
#                f"{width}x{height}", "-f", "image2", f"{video}-%03d.jpeg"]
#     try:
#         result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         print(result.stdout)
#         print(result.stderr)
#         print("Команда успешно выполнена")
#     except subprocess.CalledProcessError as e:
#         print(f"Ошибка выполнения команды: {e}")
video_file = "C:/Users/dondu/Documents/GitHub/dyplom/КАК_РАБОТАЕТ_ПРОЦЕССОР_ОСНОВЫ_ПРОГРАММИРОВАНИЯ/КАК_РАБОТАЕТ_ПРОЦЕССОР_ОСНОВЫ_ПРОГРАММИРОВАНИЯ.mp4"
output_file = "C:/Users/dondu/Documents/GitHub/dyplom/КАК_РАБОТАЕТ_ПРОЦЕССОР_ОСНОВЫ_ПРОГРАММИРОВАНИЯ-%03d.jpeg"


# Запуск команды с использованием subprocess.run
try:
    command = [
    "ffmpeg",
    "-y",
    "-i", video_file,
    "-vsync", "0",
    "-vf", "select=eq(pict_type\\,PICT_TYPE_I)",
    "-s", "1920x1080",
    "-f", "image2",
    output_file
]
    # extract_image('C:\\Users\\dondu\\Documents\\GitHub\\dyplom\\КАК_РАБОТАЕТ_ПРОЦЕССОР_ОСНОВЫ_ПРОГРАММИРОВАНИЯ')
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    print("Вывод команды:\n", result.stdout)
    print("Ошибки команды:\n", result.stderr)
    print("Видео успешно обработано")
except subprocess.CalledProcessError as e:
    print(f"Произошла ошибка при выполнении команды: {e}")
