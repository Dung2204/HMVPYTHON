import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import struct
import altair
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import mlflow
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from collections import Counter



@st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
def get_sampled_pixels(images, sample_size=100_000):
    return np.random.choice(images.flatten(), sample_size, replace=False)

@st.cache_data  # Cache danh sách ảnh ngẫu nhiên
def get_random_indices(num_images, total_images):
    return np.random.randint(0, total_images, size=num_images)

# Cấu hình Streamlit
st.set_page_config(page_title="Phân loại ảnh", layout="wide")
# Định nghĩa hàm để đọc file .idx
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]
mlflow_username = st.secrets["MLFLOW_TRACKING_USERNAME"]
mlflow_password = st.secrets["MLFLOW_TRACKING_PASSWORD"]

# Thiết lập biến môi trường
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

# Thiết lập MLflow
mlflow.set_tracking_uri(mlflow_tracking_uri)



# Định nghĩa đường dẫn đến các file MNIST
# dataset_path = r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh4"
dataset_path = os.path.dirname(os.path.abspath(__file__)) 
train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

# Tải dữ liệu
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# Giao diện Streamlit
st.title("📸 Phân loại ảnh MNIST với Streamlit")

with st.expander("🖼️ Dữ liệu ban đầu", expanded=True):
    st.subheader("📌***1.Thông tin về bộ dữ liệu MNIST***")
    st.markdown(
        '''
        **MNIST** là phiên bản được chỉnh sửa từ bộ dữ liệu **NIST gốc** của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
        Bộ dữ liệu ban đầu gồm các chữ số viết tay từ **nhân viên bưu điện** và **học sinh trung học**.  

        Các nhà nghiên cứu **Yann LeCun, Corinna Cortes, và Christopher Burges** đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành **MNIST**  
        để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
        '''
    )
    # Đặc điểm của bộ dữ liệu
    st.subheader("📌***2. Đặc điểm của bộ dữ liệu***")
    st.markdown(
        '''
        - **Số lượng ảnh:** 70.000 ảnh chữ số viết tay  
        - **Kích thước ảnh:** Mỗi ảnh có kích thước 28x28 pixel  
        - **Cường độ điểm ảnh:** Từ 0 (màu đen) đến 255 (màu trắng)  
        - **Dữ liệu nhãn:** Mỗi ảnh đi kèm với một nhãn số từ 0 đến 9  
        '''
    )
    st.write(f"🔍 Số lượng ảnh huấn luyện: `{train_images.shape[0]}`")
    st.write(f"🔍 Số lượng ảnh kiểm tra: `{test_images.shape[0]}`")


    st.subheader("📌**3. Hiển thị số lượng mẫu của từng chữ số từ 0 đến 9 trong tập huấn luyện**")
    label_counts = pd.Series(train_labels).value_counts().sort_index()

    # # Hiển thị biểu đồ cột
    # st.subheader("📊 Biểu đồ số lượng mẫu của từng chữ số")
    # st.bar_chart(label_counts)

    # Hiển thị bảng dữ liệu dưới biểu đồ
    st.subheader("📋 Số lượng mẫu cho từng chữ số")
    df_counts = pd.DataFrame({"Chữ số": label_counts.index, "Số lượng mẫu": label_counts.values})
    st.dataframe(df_counts)


    st.subheader("📌***4. Chọn ngẫu nhiên 10 ảnh từ tập huấn luyện để hiển thị***")
    num_images = 10
    random_indices = random.sample(range(len(train_images)), num_images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for ax, idx in zip(axes, random_indices):
        ax.imshow(train_images[idx], cmap='gray')
        ax.axis("off")
        ax.set_title(f"Label: {train_labels[idx]}")

    st.pyplot(fig)

    st.subheader("📌***5. Kiểm tra hình dạng của tập dữ liệu***")
        # Kiểm tra hình dạng của tập dữ liệu
    st.write("🔍 Hình dạng tập huấn luyện:", train_images.shape)
    st.write("🔍 Hình dạng tập kiểm tra:", test_images.shape)

    st.subheader("📌***6. Kiểm tra xem có giá trị không phù hợp trong phạm vi không***")

    # Kiểm tra xem có giá trị pixel nào ngoài phạm vi 0-255 không
    if (train_images.min() < 0) or (train_images.max() > 255):
        st.error("⚠️ Cảnh báo: Có giá trị pixel ngoài phạm vi 0-255!")
    else:
        st.success("✅ Dữ liệu pixel hợp lệ (0 - 255).")



    st.subheader("📌***7. Chuẩn hóa dữ liệu (đưa giá trị pixel về khoảng 0-1)***")
    # Chuẩn hóa dữ liệu
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Hiển thị thông báo sau khi chuẩn hóa
    st.success("✅ Dữ liệu đã được chuẩn hóa về khoảng [0,1].")

    # Hiển thị bảng dữ liệu đã chuẩn hóa (dạng số)
    num_samples = 5  # Số lượng mẫu hiển thị
    df_normalized = pd.DataFrame(train_images[:num_samples].reshape(num_samples, -1))  

    st.subheader("📌 **Bảng dữ liệu sau khi chuẩn hóa**")
    st.dataframe(df_normalized)

    
    sample_size = 10_000  
    pixel_sample = np.random.choice(train_images.flatten(), sample_size, replace=False)

    st.subheader("📊 **Phân bố giá trị pixel sau khi chuẩn hóa**")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Vẽ histogram tối ưu hơn
    ax.hist(pixel_sample, bins=30, color="blue", edgecolor="black")
    ax.set_title("Phân bố giá trị pixel sau khi chuẩn hóa", fontsize=12)
    ax.set_xlabel("Giá trị pixel (0-1)")
    ax.set_ylabel("Tần suất")

    st.pyplot(fig)
    st.markdown(
    """
    **🔍 Giải thích:**

        1️⃣ Phần lớn pixel có giá trị gần 0: 
        - Cột cao nhất nằm ở giá trị pixel ~ 0 cho thấy nhiều điểm ảnh trong tập dữ liệu có màu rất tối (đen).  
        - Điều này phổ biến trong các tập dữ liệu grayscale như **MNIST** hoặc **Fashion-MNIST**.  

        2️⃣ Một lượng nhỏ pixel có giá trị gần 1:
        - Một số điểm ảnh có giá trị pixel gần **1** (màu trắng), nhưng số lượng ít hơn nhiều so với pixel tối.  

        3️⃣ Rất ít pixel có giá trị trung bình (0.2 - 0.8):
        - Phân bố này cho thấy hình ảnh trong tập dữ liệu có độ tương phản cao.  
        - Phần lớn pixel là **đen** hoặc **trắng**, ít điểm ảnh có sắc độ trung bình (xám).  
    """
    )



with st.expander("🖼️ XỬ LÝ DỮ LIỆU", expanded=True):
    st.subheader("📌***8. Xử lý dữ liệu và chuẩn bị huấn luyện***")
    # Kiểm tra nếu dữ liệu đã được load
    if 'train_images' in globals() and 'train_labels' in globals() and 'test_images' in globals():
        # Chuyển đổi dữ liệu thành vector 1 chiều
        X_train = train_images.reshape(train_images.shape[0], -1)
        X_test = test_images.reshape(test_images.shape[0], -1)
        y_test = test_labels
        # Cho phép người dùng chọn tỷ lệ validation
        val_size = st.slider("🔹 Chọn tỷ lệ tập validation (%)", min_value=10, max_value=50, value=20, step=5) / 100

        # Chia tập train thành train/validation theo tỷ lệ đã chọn
        X_train, X_val, y_train, y_val = train_test_split(X_train, train_labels, test_size=val_size, random_state=42)

        st.write("✅ Dữ liệu đã được xử lý và chia tách.")
        st.write(f"🔹 Kích thước tập huấn luyện: `{X_train.shape}`")
        st.write(f"🔹 Kích thước tập validation: `{X_val.shape}`")
        st.write(f"🔹 Kích thước tập kiểm tra: `{X_test.shape}`")

            # Biểu đồ phân phối nhãn dữ liệu
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=list(Counter(y_train).keys()), y=list(Counter(y_train).values()), palette="Blues", ax=ax)
        ax.set_title("Phân phối nhãn trong tập huấn luyện")
        ax.set_xlabel("Nhãn")
        ax.set_ylabel("Số lượng")
        st.pyplot(fig)

        st.markdown(
        """
            ### 📊 Mô tả biểu đồ  
            Biểu đồ cột hiển thị **phân phối nhãn** trong tập huấn luyện.  
            - **Trục hoành (x-axis):** Biểu diễn các nhãn (labels) từ `0` đến `9`.  
            - **Trục tung (y-axis):** Thể hiện **số lượng mẫu dữ liệu** tương ứng với mỗi nhãn.  

            ### 🔍 Giải thích  
            - Biểu đồ giúp ta quan sát số lượng mẫu của từng nhãn trong tập huấn luyện.  
            - Mỗi thanh (cột) có màu sắc khác nhau: **xanh nhạt đến xanh đậm**, đại diện cho số lượng dữ liệu của từng nhãn.  
            - Một số nhãn có số lượng mẫu nhiều hơn hoặc ít hơn, điều này có thể gây ảnh hưởng đến độ chính xác của mô hình nếu dữ liệu không cân bằng.  
        """
        )
    else:
        st.error("🚨 Dữ liệu chưa được nạp. Hãy đảm bảo `train_images`, `train_labels` và `test_images` đã được tải trước khi chạy.")
    

with st.expander("🖼️ Kỹ thuật phân cụm", expanded=True):
    st.subheader("📌***9. Phân cụm dữ liệu***")

    if 'X_train' in globals() and 'X_val' in globals() and 'X_test' in globals():
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Giảm chiều bằng PCA (2D) để trực quan hóa
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)

        # Chọn phương pháp phân cụm
        clustering_method = st.selectbox("🔹 Chọn phương pháp phân cụm:", ["K-means", "DBSCAN"])

        if clustering_method == "K-means":
            k = st.slider("🔸 Số cụm (K-means)", min_value=2, max_value=20, value=10)
            st.markdown(
            """ 
                    🔹 **Số cụm (K):**  
                    - Xác định số lượng nhóm mà thuật toán sẽ chia dữ liệu vào.  
                    - Giá trị hợp lý: `2` đến `20`.  
                    - Lưu ý:  
                        - Chọn **quá nhỏ** có thể dẫn đến nhóm không đủ tốt.  
                        - Chọn **quá lớn** có thể làm mất ý nghĩa.  
                    """
            )
            st.markdown("&nbsp;" * 3, unsafe_allow_html=True)  # Tạo khoảng trống


            init_method = st.selectbox("🔸 Phương pháp khởi tạo", ["k-means++", "random"])
            st.markdown(
            """ 
                    🔹 **Phương pháp khởi tạo (`init` method)**  
                    - `"k-means++"`: Chọn các điểm trung tâm ban đầu thông minh hơn, giúp hội tụ nhanh hơn.  
                    - `"random"`: Chọn ngẫu nhiên các điểm trung tâm, có thể không tối ưu.  
                    - Khuyến nghị: `"k-means++"` (thường tốt hơn).  
            """
            )
            st.markdown("&nbsp;" * 33, unsafe_allow_html=True)  # Tạo khoảng trống


            max_iter = st.slider("🔸 Số vòng lặp tối đa", min_value=100, max_value=500, value=300, step=50)
            st.markdown(
            """ 
                    🔹 **Số vòng lặp tối đa (`max_iter`)**  
                    - Xác định số lần cập nhật trung tâm cụm trước khi thuật toán dừng.  
                    - Giá trị hợp lý: `100` đến `500`.  
                    - Lưu ý:  
                        - Số vòng lặp lớn giúp thuật toán hội tụ tốt hơn.  
                        - Nhưng cũng tăng thời gian tính toán, có thể gây chậm trễ nếu dữ liệu lớn.  
            """
            )

            if st.button("🚀 Chạy K-means"):
                with mlflow.start_run():
                    kmeans = KMeans(n_clusters=k, init=init_method, max_iter=max_iter, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_train_pca)

                    mlflow.log_param("algorithm", "K-means")
                    mlflow.log_param("k", k)
                    mlflow.log_param("init_method", init_method)
                    mlflow.log_param("max_iter", max_iter)    

                    # Log kết quả: Inertia (tổng bình phương khoảng cách)
                    mlflow.log_metric("inertia", kmeans.inertia_)
                    # Log mô hình K-means
                    mlflow.sklearn.log_model(kmeans, "kmeans_model")


                    # Vẽ biểu đồ phân cụm
                    fig, ax = plt.subplots(figsize=(6, 4))
                    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=labels, cmap='tab10', alpha=0.5)
                    ax.set_title(f"K-means với K={k}")
                    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                    ax.add_artist(legend1)
                    st.pyplot(fig)

                    fig.savefig("kmeans_clusters.png")
                    mlflow.log_artifact("kmeans_clusters.png")

                    st.markdown(
                    """
                    ### 📌 Giải thích biểu đồ phân cụm   
                    - **Mỗi chấm trên đồ thị** 🟢🔵🟣: Đại diện cho một mẫu dữ liệu trong tập huấn luyện (ở đây có thể là dữ liệu MNIST hoặc một tập dữ liệu khác).  
                    - **Màu sắc** 🎨:  
                        - Các màu sắc tượng trưng cho các cụm dữ liệu được tạo ra bởi thuật toán K-Means với K bằng số cụm được chọn.  
                        - Các điểm có cùng màu được nhóm lại vào cùng một cụm do K-Means phân cụm dựa trên khoảng cách trong không gian hai chiều.  
                    - **Trục X và Y** 📉:  
                        - Đây là hai thành phần chính (principal components) được tạo ra bằng phương pháp PCA (Principal Component Analysis).  
                        - PCA giúp giảm chiều dữ liệu từ nhiều chiều xuống 2 chiều để trực quan hóa.  
                        - Giá trị trên trục X và Y có thể lên đến khoảng ±30, phản ánh sự phân bố dữ liệu sau khi PCA được áp dụng.  
                    - **Chú thích (legend)** 🏷️: Hiển thị các cụm được tạo ra.  

                    """
                    )
                mlflow.end_run()

        elif clustering_method == "DBSCAN":
            eps = st.slider("🔸 Epsilon (DBSCAN)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            st.markdown(
            """ 
                🔹 **Epsilon (`eps`)**  
                    - Xác định bán kính tối đa để xem một điểm có thuộc cùng cụm hay không.  
                    - Giá trị hợp lý: `0.1` đến `5.0`.  
                    - Lưu ý:  
                        - Nếu `eps` **quá nhỏ**, nhiều cụm nhỏ hoặc không có cụm nào hình thành.  
                        - Nếu `eps` **quá lớn**, có thể gộp quá nhiều điểm vào một cụm, làm mất ý nghĩa phân cụm.  
                """
            )
            st.markdown("&nbsp;" * 33, unsafe_allow_html=True)  # Tạo khoảng trống

            
            max_iter = st.slider("🔸 Số vòng lặp tối đa", min_value=100, max_value=500, value=300, step=50)
            st.markdown(
            """ 
                    🔹 **Số vòng lặp tối đa (`max_iter`)**  
                    - Xác định số lần cập nhật trung tâm cụm trước khi thuật toán dừng.  
                    - Giá trị hợp lý: `100` đến `500`.  
                    - Lưu ý:  
                        - Số vòng lặp lớn giúp thuật toán hội tụ tốt hơn.  
                        - Nhưng cũng tăng thời gian tính toán, có thể gây chậm trễ nếu dữ liệu lớn.  
            """
            )
            st.markdown("&nbsp;" * 33, unsafe_allow_html=True)  # Tạo khoảng trống

            #3
            min_samples = st.slider("🔸 Min Samples (DBSCAN)", min_value=1, max_value=20, value=5)
            st.markdown(
                    """ 
                    🔹 **Min Samples (`min_samples`)**  
                    - Xác định số lượng điểm lân cận tối thiểu để tạo thành một cụm hợp lệ.  
                    - Giá trị hợp lý: `1` đến `20`.  
                    - Lưu ý:  
                        - Nếu `min_samples` **quá nhỏ**, có thể tạo ra nhiều cụm nhiễu.  
                        - Nếu `min_samples` **quá lớn**, có thể bỏ sót các cụm nhỏ, gây mất thông tin quan trọng.  
                    """
            )
            st.markdown("&nbsp;" * 33, unsafe_allow_html=True)  # Tạo khoảng trống

                #4
            metric = st.selectbox("🔸 Khoảng cách (Metric)", ["euclidean", "manhattan", "cosine"])
            st.markdown(
                    """ 
                    🔹 **Metric (Khoảng cách)**  
                    - Cách đo khoảng cách giữa các điểm dữ liệu trong thuật toán DBSCAN.  
                    - Các tùy chọn phổ biến:  
                        - `"euclidean"`: Khoảng cách Euclid (mặc định, phổ biến nhất).  
                        - `"manhattan"`: Khoảng cách theo đường phố (tổng khoảng cách theo từng trục).  
                        - `"cosine"`: Đo độ tương đồng theo góc giữa hai vector (thường dùng cho dữ liệu văn bản hoặc không có tỷ lệ cố định).  
                    - Lưu ý:  
                        - `"euclidean"` thường hoạt động tốt khi dữ liệu đã chuẩn hóa.  
                        - `"manhattan"` phù hợp hơn khi dữ liệu có các trục quan trọng rõ ràng.  
                        - `"cosine"` thích hợp khi làm việc với dữ liệu không liên quan đến khoảng cách tuyệt đối, như văn bản hoặc dữ liệu nhị phân.  
                    """
            )



            if st.button("🚀 Chạy DBSCAN"):
                with mlflow.start_run():
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                    labels = dbscan.fit_predict(X_train_pca)

                    mlflow.log_param("algorithm", "DBSCAN")
                    mlflow.log_param("eps", eps)
                    mlflow.log_param("min_samples", min_samples)
                    mlflow.log_param("metric", metric)

                    # Log số lượng cụm tìm được (không tính noise)
                    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    mlflow.log_metric("num_clusters", num_clusters)

                    # Log mô hình DBSCAN (Lưu ý: DBSCAN không có model serialization như KMeans)
                    mlflow.sklearn.log_model(dbscan, "dbscan_model")

                    # Vẽ biểu đồ phân cụm
                    fig, ax = plt.subplots(figsize=(6, 4))
                    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=labels, cmap='tab10', alpha=0.5)
                    ax.set_title(f"DBSCAN với eps={eps}, min_samples={min_samples}")
                    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                    ax.add_artist(legend1)
                    st.pyplot(fig)

                    fig.savefig("dbscan_clusters.png")
                    mlflow.log_artifact("dbscan_clusters.png")
                    
                    st.markdown("""
                    ### 📌 Giải thích biểu đồ phân cụm  
                    - **Mỗi chấm trên đồ thị** 🟢🔵🟣:  
                    - Mỗi chấm trên đồ thị biểu diễn một điểm dữ liệu, được tô màu theo cụm mà thuật toán xác định.  
                    - Trục X và Y là không gian giảm chiều (có thể bằng PCA hoặc t-SNE).  

                    - **Màu sắc** 🎨:  
                    - Mỗi màu tượng trưng cho một cụm dữ liệu khác nhau.  
                    - Vì có quá nhiều màu khác nhau, điều này cho thấy thuật toán đã chia dữ liệu thành quá nhiều cụm.  

                    - **Trục X và Y** 📉:  
                    - Trục X và Y dao động từ -10 đến khoảng 30, phản ánh sự phân bố dữ liệu.  
                    - Điều này gợi ý rằng dữ liệu gốc có thể đã được giảm chiều trước khi phân cụm.  

                    - **Chú thích (legend)** 🏷️:  
                    - Các nhãn cụm cho thấy thuật toán DBSCAN đã tìm thấy rất nhiều cụm khác nhau.  
                    - Điều này có thể là do tham số `eps` quá nhỏ, khiến thuật toán coi nhiều điểm dữ liệu riêng lẻ là một cụm riêng biệt.  
                    """)
                mlflow.end_run()
    else:
        st.error("🚨 Dữ liệu chưa được xử lý! Hãy đảm bảo bạn đã chạy phần tiền xử lý dữ liệu trước khi thực hiện phân cụm.")



with st.expander("🖼️ Đánh giá hiệu suất mô hình phân cụm", expanded=True):
    st.subheader("📌***10. Đánh giá hiệu suất mô hình phân cụm***")
    if clustering_method == "K-means" and 'labels' in locals():
        silhouette_avg = silhouette_score(X_train_pca, labels)
        dbi_score = davies_bouldin_score(X_train_pca, labels)

        st.markdown("### 📊 Đánh giá mô hình K-means")
        st.write(f"✅ **Silhouette Score**: {silhouette_avg:.4f}")
        st.write(f"✅ **Davies-Bouldin Index**: {dbi_score:.4f}")

        # Vẽ biểu đồ Silhouette Score
        fig, ax = plt.subplots(figsize=(6, 4))
        sample_silhouette_values = silhouette_samples(X_train_pca, labels)
        y_lower = 10

        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

        ax.set_title("Biểu đồ Silhouette Score - K-means")
        ax.set_xlabel("Silhouette Score")
        ax.set_ylabel("Cụm")
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", label="Giá trị trung bình")
        ax.legend()

        st.pyplot(fig)

            # Giải thích về biểu đồ
        st.markdown("""
        **📌 Giải thích biểu đồ Silhouette Score**:
            - **Trục hoành**: Silhouette Score (từ -1 đến 1).
            - **Trục tung**: Các cụm được phát hiện.
            - **Dải màu**: Độ rộng biểu thị số lượng điểm trong từng cụm.
            - **Đường đứt đỏ**: Trung bình Silhouette Score của toàn bộ dữ liệu.
            - **Nếu giá trị Silhouette Score âm**: có thể một số điểm bị phân cụm sai.
        """)

    elif clustering_method == "DBSCAN" and 'labels' in locals():
        unique_labels = set(labels)
        if len(unique_labels) > 1:  # Tránh lỗi khi chỉ có 1 cụm hoặc toàn bộ điểm bị coi là nhiễu (-1)
            silhouette_avg = silhouette_score(X_train_pca, labels)
            dbi_score = davies_bouldin_score(X_train_pca, labels)

            st.markdown("### 📊 Đánh giá mô hình DBSCAN")
            st.write(f"✅ **Silhouette Score**: {silhouette_avg:.4f}")
            st.write(f"✅ **Davies-Bouldin Index**: {dbi_score:.4f}")

            # Vẽ biểu đồ Silhouette Score
            fig, ax = plt.subplots(figsize=(6, 4))
            sample_silhouette_values = silhouette_samples(X_train_pca, labels)
            y_lower = 10

            for i in unique_labels:
                if i == -1:  # Bỏ qua nhiễu
                    continue
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10

            ax.set_title("Biểu đồ Silhouette Score - DBSCAN")
            ax.set_xlabel("Silhouette Score")
            ax.set_ylabel("Cụm")
            ax.axvline(x=silhouette_avg, color="red", linestyle="--", label="Giá trị trung bình")
            ax.legend()

            st.pyplot(fig)

            # Giải thích chi tiết về biểu đồ Silhouette Score - DBSCAN
            st.markdown("""
                **📌 Giải thích biểu đồ Silhouette Score (DBSCAN)**:    
                - **Trục tung (Cụm - Cluster ID)**: Mỗi cụm được hiển thị với một dải màu.
                - **Trục hoành (Silhouette Score)**: Giá trị càng gần **1** thì phân cụm càng tốt, gần **0** là chồng chéo, âm là phân cụm kém.
                - **Đường đỏ nét đứt**: Silhouette Score trung bình của toàn bộ cụm.
                
                🔍 **Về các đường đen trong biểu đồ**:
                - Đây là các điểm nhiễu (outliers) mà DBSCAN không thể gán vào cụm nào.
                - Trong DBSCAN, các điểm nhiễu được gán nhãn `-1`, nhưng không được hiển thị trên biểu đồ.
                - Tuy nhiên, một số điểm nhiễu có thể vẫn xuất hiện như **các vệt đen dọc**, do chúng có Silhouette Score gần giống nhau nhưng không thuộc bất kỳ cụm nào.
                - Điều này xảy ra khi:
                - Số lượng điểm nhiễu lớn.
                - Silhouette Score của nhiễu không ổn định, khiến nhiều điểm có giá trị gần nhau.
                - Cụm có chất lượng kém, tức là thuật toán đang nhận diện rất nhiều điểm là nhiễu thay vì cụm rõ ràng.
            """)
        else:
            st.warning("⚠️ DBSCAN chỉ tìm thấy 1 cụm hoặc tất cả điểm bị coi là nhiễu. Hãy thử điều chỉnh `eps` và `min_samples`.")
            
st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
with st.expander("🖼️ Đánh giá hiệu suất mô hình phân cụm", expanded=True):
    # st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print("🎯 Kiểm tra trên DagsHub: https://dagshub.com/Dung2204/Minst-mlflow.mlflow")


# # # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh4"