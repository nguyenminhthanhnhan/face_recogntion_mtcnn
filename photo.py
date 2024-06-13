import cv2
import os

def capture_and_save_face_photos(output_dir, student_id, total_photos):
    # Mở camera
    cap = cv2.VideoCapture(0)

    # Kiểm tra xem camera có được mở không
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    # Số thứ tự ban đầu
    photo_counter = 1

    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()

        # Hiển thị frame
        cv2.imshow('Capture Face', frame)

        # Chờ phím 'ESC' để thoát, 'SPACE' để chụp ảnh, 'q' để kết thúc
        key = cv2.waitKey(1)
        if key == 27:  # Phím 'ESC'
            break
        elif key == 32:  # Phím 'SPACE'
            # Lưu ảnh vào thư mục photos
            file_name = f"{student_id}_{photo_counter}.jpg"
            file_path = os.path.join(output_dir, file_name)
            cv2.imwrite(file_path, frame)
            print(f"Ảnh đã được lưu tại {file_path}")

            # Tăng số thứ tự
            photo_counter += 1

            # Kiểm tra điều kiện thoát sau khi chụp đủ số ảnh
            if photo_counter > total_photos:
                break

    # Đóng camera và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Thư mục lưu ảnh
    output_directory = "photos"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Mã số sinh viên
    student_id = "2004609"

    # Số lượng ảnh cần chụp
    total_photos_to_capture = 500

    # Chụp ảnh và lưu
    capture_and_save_face_photos(output_directory, student_id, total_photos_to_capture)
   