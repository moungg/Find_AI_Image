from PIL import Image, ExifTags

def get_exif_info(image_path):
    img = Image.open(image_path)
    exif_data = img.getexif()
    if not exif_data:
        return None
    return {ExifTags.TAGS.get(tag): value for tag, value in exif_data.items() if tag in ExifTags.TAGS}

# 사용 예시
print("실제 이미지 EXIF:")
print(get_exif_info("test_image.jpeg"))

print("\nAI 이미지 EXIF:")
print(get_exif_info("ai_test_image.png"))
