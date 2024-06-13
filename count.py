import os
import matplotlib.pyplot as plt

def count_images(directory):
    # Tạo một từ điển để lưu số lượng ảnh của mỗi người
    image_counts = {}

    # Lặp qua tất cả các tệp trong thư mục
    for filename in os.listdir(directory):
        # Kiểm tra nếu là tệp hình ảnh
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Tách tên tệp để lấy tên người
            person_name = filename.split('_')[0]

            # Kiểm tra xem người này đã được thêm vào từ điển chưa
            if person_name not in image_counts:
                image_counts[person_name] = 1
            else:
                image_counts[person_name] += 1

    # Vẽ biểu đồ
    labels = list(image_counts.keys())
    counts = list(image_counts.values())

    plt.bar(labels, counts, color='blue')
    plt.xlabel('Người')
    plt.ylabel('Số lượng ảnh')
    plt.title('Số lượng ảnh của mỗi người')
    plt.show()

# Thay đổi đường dẫn dưới đây thành đường dẫn thực tế của thư mục chứa ảnh của bạn
directory_path = "photos"
count_images(directory_path)
