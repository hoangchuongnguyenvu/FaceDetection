import os
import cv2
import numpy as np
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Hàm để tải và xử lý ảnh
def load_data(face_files, non_face_files):
    data = []
    labels = []
    
    # Tải ảnh gương mặt
    for uploaded_file in face_files:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (24, 24))
        data.append(img.flatten())
        labels.append(1)  # Nhãn cho ảnh gương mặt

    # Tải ảnh không phải gương mặt
    for uploaded_file in non_face_files:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (24, 24))
        data.append(img.flatten())
        labels.append(0)  # Nhãn cho ảnh không phải gương mặt

    return np.array(data), np.array(labels)

# Hàm để dự đoán
def predict_image(image, model):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (24, 24))
    image = image.flatten().reshape(1, -1)
    prediction = model.predict(image)
    return prediction[0]

# Hàm phát hiện khuôn mặt trong ảnh
def detect_and_recognize_face(image, model=None, show_boxes=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (24, 24)).flatten().reshape(1, -1)
        
        if model is not None:
            label = model.predict(face_resized)
        else:
            label = ["Unknown"]
        
        if show_boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, f'Person: {label[0]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return image

# Thiết lập tiêu đề ứng dụng

# Define sections for the sidebar
sections = {
    "Thị giác máy tính": [
        "Training và testing Dataset cho trước để phát hiện khuôn mặt",
        "Phát hiện khuôn mặt ở hình ảnh",
        "Phát hiện khuôn mặt ở video"
    ],
    "Nhiệm vụ liên quan đến thị giác": [
        "Phát hiện đối tượng",
        "Phân loại hình ảnh",
        "Phân đoạn hình ảnh",
        "Phân đoạn tương tác",
        "Nhận dạng cử chỉ",
        "Phát hiện điểm mốc trên bàn tay",
        "Phát hiện khuôn mặt"
    ],
    "Việc cần làm liên quan đến văn bản": [
        "Phân loại văn bản",
        "Nhúng văn bản"
    ]
}

# Set up sidebar
st.sidebar.title("Lọc")
selected_section = st.sidebar.selectbox("Chọn một đề mục", list(sections.keys()))

# Sub-sections based on the selected main section
selected_sub_section = st.sidebar.radio("Chọn một ứng dụng", sections[selected_section])

# Display content dynamically based on the selection
st.title(f"{selected_sub_section}")

# Phần xử lý cho "Training và testing Dataset cho trước để phát hiện khuôn mặt"
if selected_sub_section == "Training và testing Dataset cho trước để phát hiện khuôn mặt":
    st.subheader("Tải lên ảnh gương mặt")
    face_files = st.file_uploader("Tải lên ảnh gương mặt (nhấn Ctrl để chọn nhiều ảnh)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    st.subheader("Tải lên ảnh không phải gương mặt")
    non_face_files = st.file_uploader("Tải lên ảnh không phải gương mặt (nhấn Ctrl để chọn nhiều ảnh)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Huấn luyện mô hình"):
        if face_files and non_face_files:
            data, labels = load_data(face_files, non_face_files)

            # Chia tập dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

            # Huấn luyện mô hình KNN
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)

            # Lưu mô hình vào st.session_state
            st.session_state.knn = knn

            # Đánh giá mô hình
            accuracy = knn.score(X_test, y_test)
            st.success(f'Mô hình đã được huấn luyện thành công với độ chính xác: {accuracy * 100:.2f}%')
        else:
            st.error("Vui lòng tải lên ảnh gương mặt và ảnh không phải gương mặt.")

    uploaded_file = st.file_uploader("Tải lên ảnh mới để dự đoán", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc ảnh từ tệp tải lên
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Kiểm tra xem mô hình đã được huấn luyện chưa
        if 'knn' in st.session_state:
            # Dự đoán ảnh
            prediction = predict_image(image, st.session_state.knn)

            if prediction == 1:
                st.image(image, caption="Đây là một bức ảnh gương mặt.", use_column_width=True)
            else:
                st.image(image, caption="Đây không phải là bức ảnh gương mặt.", use_column_width=True)
        else:
            st.warning("Vui lòng huấn luyện mô hình trước khi dự đoán.")

# Phát hiện khuôn mặt ở hình ảnh
elif selected_sub_section == "Phát hiện khuôn mặt ở hình ảnh":
    uploaded_image = st.file_uploader("Tải lên hình ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Add a checkbox to toggle bounding boxes
        show_boxes = st.checkbox('Hiển thị viền quanh khuôn mặt', value=True)

        # Detect and recognize face in the uploaded image
        if 'knn' in st.session_state:
            processed_image = detect_and_recognize_face(image, st.session_state.knn, show_boxes)
        else:
            processed_image = detect_and_recognize_face(image, show_boxes=show_boxes)

        # Convert BGR to RGB for display in Streamlit
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Display the processed image
        st.image(processed_image, channels="RGB", use_column_width=True)

# Phát hiện khuôn mặt ở video
elif selected_sub_section == "Phát hiện khuôn mặt ở video":
    start_webcam = st.button('Bắt đầu webcam')
    
    if start_webcam:
        cap = cv2.VideoCapture(0)  # Mở webcam (ID 0 là webcam mặc định)
        stframe = st.empty()  # Khung để hiển thị luồng video

        stop_webcam = False
        stop_button = st.button('Dừng webcam')

        while not stop_webcam and cap.isOpened():
            ret, frame = cap.read()  # Đọc khung hình từ webcam
            if ret:
                # Phát hiện và nhận diện khuôn mặt trong từng khung hình
                if 'knn' in st.session_state:
                    processed_frame = detect_and_recognize_face(frame, st.session_state.knn)
                else:
                    processed_frame = detect_and_recognize_face(frame)

                # Chuyển đổi BGR sang RGB để hiển thị trên Streamlit
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Hiển thị khung hình đã xử lý
                stframe.image(processed_frame, channels="RGB", use_column_width=True)

                # Kiểm tra nếu người dùng nhấn nút "Dừng webcam"
                if stop_button:
                    stop_webcam = True

        cap.release()  # Giải phóng webcam sau khi dừng
        cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV nếu có
