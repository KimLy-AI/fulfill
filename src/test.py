from PIL import Image
import os, pathlib

filename = "(ÁO TRẮNG)Rod Wave Last Lap Tour 2024 Shirt And Sweatshirt R_JniEvvwa.png"
base = pathlib.Path(__file__).parent / "src/data/images/Design"
file_path = (base / filename).resolve()

print("CWD:", os.getcwd())
print("Checking:", file_path)
print("Exists?", file_path.is_file())

if not file_path.is_file():
    raise FileNotFoundError(f"File not found: {file_path}")

# # Nếu đường dẫn quá dài trên Windows
# file_to_open = str(file_path)
# if os.name == "nt" and len(file_to_open) > 250:
#     file_to_open = r"\\?\" + file_to_open

img = Image.open(file_to_open)
