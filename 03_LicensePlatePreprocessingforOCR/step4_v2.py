import cv2
import numpy as np


def crop_plate_border(img, pad=2, debug=False):
    """
    Remove white borders around license plate content.
    
    Phương pháp: Flood fill từ 4 góc để loại bỏ border trắng,
    sau đó tìm bounding box của phần text còn lại.
    Chính xác và hiệu quả với ảnh binary đã preprocess.

    Args:
        img (np.ndarray): BGR or grayscale image (thường là binary sau morph)
        pad (int): padding sau khi crop
        debug (bool): return debug visualization

    Returns:
        cropped image (chỉ chứa nội dung text, không có border)
    """
    # Chuyển sang grayscale nếu là ảnh màu
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    
    # Tạo bản sao để flood fill
    gray_filled = gray.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # Flood fill từ 4 góc để loại bỏ border trắng
    # Góc trên trái
    if gray[0, 0] > 128:  # Nếu là màu trắng (border)
        cv2.floodFill(gray_filled, mask, (0, 0), 0, loDiff=10, upDiff=10, flags=8)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)  # Reset mask
    
    # Góc trên phải
    if gray[0, w-1] > 128:
        cv2.floodFill(gray_filled, mask, (w-1, 0), 0, loDiff=10, upDiff=10, flags=8)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # Góc dưới trái
    if gray[h-1, 0] > 128:
        cv2.floodFill(gray_filled, mask, (0, h-1), 0, loDiff=10, upDiff=10, flags=8)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # Góc dưới phải
    if gray[h-1, w-1] > 128:
        cv2.floodFill(gray_filled, mask, (w-1, h-1), 0, loDiff=10, upDiff=10, flags=8)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # Flood fill từ các cạnh (để loại bỏ border dọc/ngang)
    # Cạnh trên
    for x in range(0, w, max(1, w//20)):
        if gray[0, x] > 128:
            cv2.floodFill(gray_filled, mask, (x, 0), 0, loDiff=10, upDiff=10, flags=8)
    
    # Cạnh dưới
    for x in range(0, w, max(1, w//20)):
        if gray[h-1, x] > 128:
            cv2.floodFill(gray_filled, mask, (x, h-1), 0, loDiff=10, upDiff=10, flags=8)
    
    # Cạnh trái
    for y in range(0, h, max(1, h//20)):
        if gray[y, 0] > 128:
            cv2.floodFill(gray_filled, mask, (0, y), 0, loDiff=10, upDiff=10, flags=8)
    
    # Cạnh phải
    for y in range(0, h, max(1, h//20)):
        if gray[y, w-1] > 128:
            cv2.floodFill(gray_filled, mask, (w-1, y), 0, loDiff=10, upDiff=10, flags=8)
    
    # ===== TÌM CONTOUR CỦA BIỂN SỐ =====
    # Tìm contour từ phần text còn lại (màu trắng)
    contours, _ = cv2.findContours(gray_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Không tìm thấy contour, trả về ảnh gốc
        return img
    
    # Chọn contour lớn nhất (biển số)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Lấy bounding rect của contour
    x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
    
    # Thêm padding
    top = max(0, y - pad)
    bottom = min(h, y + h_rect + pad)
    left = max(0, x - pad)
    right = min(w, x + w_rect + pad)
    
    # Kiểm tra hợp lệ
    if top >= bottom or left >= right:
        return img
    
    # Crop ảnh gốc (giữ nguyên màu nếu là BGR)
    if len(img.shape) == 3:
        crop = img[top:bottom, left:right]
    else:
        crop = img[top:bottom, left:right]
    
    if debug:
        dbg = img.copy()
        if len(dbg.shape) == 2:
            dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)
        
        # Vẽ contour (màu xanh lá, độ dày 2)
        cv2.drawContours(dbg, [largest_contour], -1, (0, 255, 0), 2)
        
        # Vẽ bounding rect (màu đỏ, độ dày 2)
        cv2.rectangle(dbg, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # Vẽ điểm góc của bounding rect (màu xanh dương)
        cv2.circle(dbg, (left, top), 5, (255, 0, 0), -1)
        cv2.circle(dbg, (right, bottom), 5, (255, 0, 0), -1)
        
        # Vẽ điểm góc của contour (màu vàng)
        for point in largest_contour:
            cv2.circle(dbg, tuple(point[0]), 3, (0, 255, 255), -1)
        
        # Thêm text thông tin
        contour_area = cv2.contourArea(largest_contour)
        rect_area = w_rect * h_rect
        info_text = [
            f"Contour area: {contour_area:.0f}",
            f"Rect area: {rect_area:.0f}",
            f"Points: {len(largest_contour)}",
            f"Crop: ({left},{top}) to ({right},{bottom})"
        ]
        
        y_offset = 20
        for i, text in enumerate(info_text):
            cv2.putText(dbg, text, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return crop, dbg
    
    return crop


if __name__ == "__main__":
    # img_path = "./img/multi_plates.png"
    img_path = "./outputs/preprocess/06_final/plate_0_06_final.jpg"
    img = cv2.imread(img_path)
    assert img is not None, "Cannot read input image"
    print(f"Image shape: {img.shape}")

    crop, debug_img = crop_plate_border(img, debug=True)
    cv2.imwrite("./outputs/crop_border.jpg", crop)
    cv2.imwrite("./outputs/crop_border_debug.jpg", debug_img)

    print("Done!")