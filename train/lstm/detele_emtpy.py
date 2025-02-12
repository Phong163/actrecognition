import os

output_folder = r"C:\Users\OS\Desktop\Act_recognize\datasets\Standing"

# Duyệt qua tất cả các file trong thư mục output
for file in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file)

    # Kiểm tra nếu là file (không phải thư mục)
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()  # Đọc nội dung file và loại bỏ khoảng trắng

        # Nếu file rỗng hoặc chỉ chứa dòng trống → xóa
        if not content:
            os.remove(file_path)
            print(f"🗑️ Đã xóa file rỗng: {file}")
        else:
            print(f"✅ File hợp lệ: {file}")
