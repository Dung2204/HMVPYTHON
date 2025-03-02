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
from mlflow.tracking import MlflowClient
def run_ClusteringMinst_app():
    @st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
    def get_sampled_pixels(images, sample_size=100_000):
        return np.random.choice(images.flatten(), sample_size, replace=False)

    @st.cache_data  # Cache danh sách ảnh ngẫu nhiên
    def get_random_indices(num_images, total_images):
        return np.random.randint(0, total_images, size=num_images)

    # Cấu hình Streamlit
    # st.set_page_config(page_title="Phân loại ảnh", layout="wide")
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
    
    # Thiết lập MLflow (Đặt sau khi mlflow_tracking_uri đã có giá trị)
    mlflow.set_tracking_uri(mlflow_tracking_uri)



    # Định nghĩa đường dẫn đến các file MNIST
    # dataset_path = r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App"
    dataset_path = os.path.dirname(os.path.abspath(__file__)) 
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    if "train_images" not in st.session_state:
            st.session_state.train_images = load_mnist_images(train_images_path)
            st.session_state.train_labels = load_mnist_labels(train_labels_path)
            st.session_state.test_images = load_mnist_images(test_images_path)
            st.session_state.test_labels = load_mnist_labels(test_labels_path)
    # Tải dữ liệu
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    

    # Giao diện Streamlit
    st.title("📸 MINST Clustering")
    tabs = st.tabs([
            "Tập dữ liệu",
            "Xử lí dữ liệu",
            "Thông tin",
            "Huấn luyện mô hình",
            "Đánh giá mô hình",
            "Thông tin & Mlflow",
    ])
    tab_info, tab_load,tab_note, tab_preprocess, tab_split ,tab_mlflow= tabs
    with tab_info:
        with st.expander("**Thông tin dữ liệu**", expanded=True):
            st.markdown(
                '''
                **MNIST** là phiên bản được chỉnh sửa từ bộ dữ liệu **NIST gốc** của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
                Bộ dữ liệu ban đầu gồm các chữ số viết tay từ **nhân viên bưu điện** và **học sinh trung học**.  

                Các nhà nghiên cứu **Yann LeCun, Corinna Cortes, và Christopher Burges** đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành **MNIST**  
                để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
                '''
            )
            # Đặc điểm của bộ dữ liệu
        with st.expander("**Đặc điểm của bộ dữ liệu**", expanded=True):
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


        with st.expander("**Hiển thị số lượng mẫu của từng chữ số từ 0 đến 9 trong tập huấn luyện**", expanded=True):
            label_counts = pd.Series(train_labels).value_counts().sort_index()
            # Hiển thị bảng dữ liệu dưới biểu đồ
            df_counts = pd.DataFrame({"Chữ số": label_counts.index, "Số lượng mẫu": label_counts.values})
            st.dataframe(df_counts)
            num_images = 10
            random_indices = random.sample(range(len(train_images)), num_images)
            fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
            st.write("**Một số ảnh ví dụ:**")
            for ax, idx in zip(axes, random_indices):
                ax.imshow(train_images[idx], cmap='gray')
                ax.axis("off")
                ax.set_title(f"Label: {train_labels[idx]}")

            st.pyplot(fig)

        with st.expander("**Kiểm tra hình dạng của tập dữ liệu**", expanded=True):    
                # Kiểm tra hình dạng của tập dữ liệu
            st.write("🔍 Hình dạng tập huấn luyện:", train_images.shape)
            st.write("🔍 Hình dạng tập kiểm tra:", test_images.shape)
            # Kiểm tra xem có giá trị pixel nào ngoài phạm vi 0-255 không
            if (train_images.min() < 0) or (train_images.max() > 255):
                st.error("⚠️ Cảnh báo: Có giá trị pixel ngoài phạm vi 0-255!")
            else:
                st.success("✅ Dữ liệu pixel hợp lệ (0 - 255).")

            # Chuẩn hóa dữ liệu
            train_images = train_images.astype("float32") / 255.0
            test_images = test_images.astype("float32") / 255.0

            # Hiển thị thông báo sau khi chuẩn hóa
            st.success("✅ Dữ liệu đã được chuẩn hóa về khoảng [0,1].")

            # Hiển thị bảng dữ liệu đã chuẩn hóa (dạng số)
            num_samples = 5  # Số lượng mẫu hiển thị
            df_normalized = pd.DataFrame(train_images[:num_samples].reshape(num_samples, -1))  

            st.write("**Bảng dữ liệu sau khi chuẩn hóa**")
            st.dataframe(df_normalized)

            
            sample_size = 10_000  
            pixel_sample = np.random.choice(train_images.flatten(), sample_size, replace=False)



    with tab_load:
        with st.expander("**Xử lý dữ liệu**", expanded=True):    
            # Chuyển đổi dữ liệu thành vector 1 chiều
            X_train = st.session_state.train_images.reshape(st.session_state.train_images.shape[0], -1)
            X_test = st.session_state.test_images.reshape(st.session_state.test_images.shape[0], -1)
            y_train = train_labels
            y_test = test_labels

            # Chọn tỷ lệ tập validation
            val_size = st.slider("🔹 **Chọn tỷ lệ tập validation (%)**", min_value=10, max_value=50, value=20, step=5) / 100

            # Chọn tỷ lệ tập kiểm tra (test)
            test_size = st.slider("🔹 **Chọn tỷ lệ tập kiểm tra (%)**", min_value=10, max_value=40, value=20, step=5) / 100

            # Chia tập train ban đầu thành train + validation + test
            X_train, X_temp, y_train, y_temp = train_test_split(X_train, y_train, test_size=(val_size + test_size), random_state=42)

            # Tiếp tục chia X_temp thành validation và test theo tỷ lệ đã chọn
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (val_size + test_size)), random_state=42)

            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.X_test = X_test


            st.write("✅ Dữ liệu đã được xử lý và chia tách.")
            st.write(f"🔹 **Kích thước tập huấn luyện**: `{X_train.shape}`")
            st.write(f"🔹 **Kích thước tập validation**: `{X_val.shape}`")
            st.write(f"🔹 **Kích thước tập kiểm tra**: `{X_test.shape}`")

            # Biểu đồ phân phối nhãn trong tập huấn luyện
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=list(Counter(y_train).keys()), y=list(Counter(y_train).values()), palette="Blues", ax=ax)
            ax.set_title("Phân phối nhãn trong tập huấn luyện")
            ax.set_xlabel("Nhãn")
            ax.set_ylabel("Số lượng")
            st.pyplot(fig)

    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):
            # Chọn mô hình
            model_option1 = st.selectbox("Chọn mô hình", ["K-Means", "DBSCAN"])
            
            if model_option1 == "K-Means":
                st.markdown("## 🔹 K-Means Clustering")
                st.markdown("---")

                st.markdown("**Khái niệm**")
                st.write("""
                - **K-Means** là thuật toán phân cụm chia dữ liệu thành $K$ nhóm dựa trên khoảng cách giữa các điểm và tâm cụm.
                - Thuật toán tìm cách **tối thiểu hóa tổng bình phương khoảng cách** giữa các điểm dữ liệu và tâm cụm (**WCSS**).
                """)

                st.markdown("### 🔄 **Quy trình hoạt động**")
                st.write("""
                1. Chọn ngẫu nhiên $K$ tâm cụm ban đầu.
                2. Gán mỗi điểm vào cụm gần nhất dựa trên khoảng cách.
                3. Cập nhật lại tâm cụm bằng trung bình của các điểm trong cụm.
                4. Lặp lại cho đến khi thuật toán hội tụ.
                """)

                st.markdown("### ⚙️ **Các tham số chính**")
                st.write("""
                - **Số cụm $K$**: Số lượng cụm cần phân loại.
                - **init method**:
                    - `"random"`: Chọn tâm cụm ngẫu nhiên.
                    - `"k-means++"`: Chọn tâm cụm thông minh hơn để tăng tốc hội tụ.
                - **max_iter**: Số lần lặp tối đa trước khi thuật toán dừng.
                """)
                st.markdown("### 🤔 **Tại sao cần K-Means++?**")
                st.write("""
                - Trong **K-Means**, nếu chọn tâm cụm ban đầu **ngẫu nhiên không tốt**, thuật toán có thể:
                    - Hội tụ chậm hoặc mắc kẹt vào nghiệm kém tối ưu.
                    - Tạo cụm không cân đối.
                    - Nhạy cảm với outlier.
                - **K-Means++** giúp chọn tâm cụm **một cách thông minh hơn**, giúp thuật toán **hội tụ nhanh hơn và ổn định hơn**.
                - **K-Means++** là một cải tiến của thuật toán K-Means nhằm chọn tâm cụm ban đầu một cách thông minh hơn, giúp tăng độ ổn định và giảm nguy cơ hội tụ vào nghiệm kém tối ưu.
                """)

                st.markdown("### ✅ **Điều kiện hội tụ**")
                st.write("""
                - Tâm cụm không thay đổi giữa các vòng lặp.
                - Gán cụm của các điểm không thay đổi.
                - Số lần lặp đạt `max_iter`.
                - Tổng bình phương khoảng cách không giảm đáng kể.
                """)

                st.markdown("### 👍 **Ưu điểm**")
                st.write("""
                - Dễ hiểu, dễ triển khai.
                - Hiệu suất cao với dữ liệu lớn.
                """)

                st.markdown("### ⚠️ **Nhược điểm**")
                st.write("""
                - Cần chọn trước số cụm $K$.
                - Nhạy cảm với nhiễu và outlier.
                - Không hiệu quả nếu cụm có hình dạng bất thường.
                """)

            elif model_option1 == "DBSCAN":
                st.markdown("## 🔹 DBSCAN (Density-Based Clustering)")
                st.markdown("---")

                st.markdown("**Khái niệm**")
                st.write("""
                - DBSCAN là thuật toán phân cụm dựa trên mật độ, hoạt động tốt với dữ liệu có hình dạng cụm phức tạp.
                - Không cần chọn số cụm trước, mà dựa vào mật độ dữ liệu.
                """)

                st.markdown("### 🔄 **Quy trình hoạt động**")
                st.write("""
                1. Chọn một điểm dữ liệu chưa được phân cụm.
                2. Nếu trong bán kính `eps` có ít nhất `min_samples` điểm, thuật toán tạo một cụm mới.
                3. Mở rộng cụm bằng cách tìm điểm gần kề có mật độ cao.
                4. Điểm nào không thuộc cụm nào được coi là **nhiễu (outlier)**.
                """)

                st.markdown("### ⚙️ **Các tham số chính**")
                st.write("""
                - **Epsilon (`eps`)**: Bán kính lân cận.
                - **min_samples**: Số điểm tối thiểu để tạo cụm.
                - **max_iter**: Số lần lặp tối đa trước khi thuật toán dừng.
                - **metric**: Cách tính khoảng cách giữa các điểm:
                    - `"euclidean"`: Khoảng cách Euclid.
                    - `"manhattan"`: Khoảng cách Manhattan.
                    - `"cosine"`: Khoảng cách Cosine.
                """)

                st.markdown("### 👍 **Ưu điểm**")
                st.write("""
                - Không cần chọn số cụm trước.
                - Xử lý tốt dữ liệu có hình dạng cụm phức tạp.
                - Tự động phát hiện outlier.
                """)

                st.markdown("### ⚠️ **Nhược điểm**")
                st.write("""
                - Nhạy cảm với giá trị `eps` và `min_samples`.
                - Hiệu suất giảm với dữ liệu có mật độ không đồng nhất.
                - Không hiệu quả với dữ liệu có số chiều cao.
                """)


    with tab_preprocess:
        with st.expander("**Kỹ thuật phân cụm**", expanded=True):    
            st.write("***Phân cụm dữ liệu***")

            if "X_train" in st.session_state and "X_val" in st.session_state and "X_test" in st.session_state:
                # Lấy dữ liệu từ session_state
                X_train = st.session_state.X_train
                X_val = st.session_state.X_val
                X_test = st.session_state.X_test
                # Chuẩn hóa dữ liệu
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

                    init_method = st.selectbox("🔸 Phương pháp khởi tạo", ["k-means++", "random"])
                    max_iter = st.slider("🔸 Số vòng lặp tối đa", min_value=100, max_value=500, value=300, step=50)
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
                    max_iter = st.slider("🔸 Số vòng lặp tối đa", min_value=100, max_value=500, value=300, step=50)
                    min_samples = st.slider("🔸 Min Samples (DBSCAN)", min_value=1, max_value=20, value=5)
                    metric = st.selectbox("🔸 Khoảng cách (Metric)", ["euclidean", "manhattan", "cosine"])

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
                            ffig, ax = plt.subplots(figsize=(6, 4))
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
            

    with tab_split:
        with st.expander(" Đánh giá hiệu suất mô hình phân cụm", expanded=True):
            if "clustering_method" not in st.session_state:
                st.session_state.clustering_method = "K-means"  # Giá trị mặc định
                clustering_method = st.session_state.clustering_method  # Lấy giá trị từ session_state
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
                st.markdown("**Giải thích biểu đồ Silhouette Score**")
                st.write("""
                - **Trục hoành**: Silhouette Score (từ -1 đến 1).
                - **Trục tung**: Các cụm được phát hiện.
                - **Dải màu**: Độ rộng biểu thị số lượng điểm trong từng cụm.
                - **Đường đứt đỏ**: Trung bình Silhouette Score của toàn bộ dữ liệu.
                - **Silhouette Score âm**: Có thể một số điểm bị phân cụm sai.
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
                    st.markdown("**Giải thích biểu đồ Silhouette Score (DBSCAN)**")
                    st.write("""
                    - **Trục tung (Cụm - Cluster ID)**: Mỗi cụm được hiển thị với một dải màu.
                    - **Trục hoành (Silhouette Score)**: Giá trị càng gần **1** thì phân cụm càng tốt, gần **0** là chồng chéo, âm là phân cụm kém.
                    - **Đường đỏ nét đứt**: Silhouette Score trung bình của toàn bộ cụm.
                    """)

                    st.markdown("🔍 **Về các đường đen trong biểu đồ**")

                    st.write("""
                    - Đây là các điểm nhiễu (outliers) mà DBSCAN không thể gán vào cụm nào.
                    - Trong DBSCAN, các điểm nhiễu được gán nhãn `-1`, nhưng không được hiển thị trên biểu đồ.
                    - Tuy nhiên, một số điểm nhiễu có thể vẫn xuất hiện như **các vệt đen dọc**, do chúng có Silhouette Score gần giống nhau nhưng không thuộc bất kỳ cụm nào.

                    **Điều này xảy ra khi:**
                    - Số lượng điểm nhiễu lớn.
                    - Silhouette Score của nhiễu không ổn định, khiến nhiều điểm có giá trị gần nhau.
                    - Cụm có chất lượng kém, tức là thuật toán đang nhận diện rất nhiều điểm là nhiễu thay vì cụm rõ ràng.
                    """)
                else:
                    st.warning("⚠️ DBSCAN chỉ tìm thấy 1 cụm hoặc tất cả điểm bị coi là nhiễu. Hãy thử điều chỉnh `eps` và `min_samples`.")

    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "MyExperiment"
    
            # Kiểm tra nếu experiment đã tồn tại
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
                st.success(f"Experiment mới được tạo với ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                st.info(f"Đang sử dụng experiment ID: {experiment_id}")
    
            mlflow.set_experiment(experiment_name)
    
            # Truy vấn các run trong experiment
            runs = client.search_runs(experiment_ids=[experiment_id])
    
            # 1) Chọn và đổi tên Run Name
            st.subheader("Đổi tên Run")
            if runs:
                run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
                               for run in runs}
                selected_run_id_for_rename = st.selectbox("Chọn Run để đổi tên:", 
                                                          options=list(run_options.keys()), 
                                                          format_func=lambda x: run_options[x])
                new_run_name = st.text_input("Nhập tên mới cho Run:", 
                                             value=run_options[selected_run_id_for_rename].split(" - ")[0])
                if st.button("Cập nhật tên Run"):
                    if new_run_name.strip():
                        client.set_tag(selected_run_id_for_rename, "mlflow.runName", new_run_name.strip())
                        st.success(f"Đã cập nhật tên Run thành: {new_run_name.strip()}")
                    else:
                        st.warning("Vui lòng nhập tên mới cho Run.")
            else:
                st.info("Chưa có Run nào được log.")
    
            # 2) Xóa Run
            st.subheader("Danh sách Run")
            if runs:
                selected_run_id_to_delete = st.selectbox("", 
                                                         options=list(run_options.keys()), 
                                                         format_func=lambda x: run_options[x])
                if st.button("Xóa Run", key="delete_run"):
                    client.delete_run(selected_run_id_to_delete)
                    st.success(f"Đã xóa Run {run_options[selected_run_id_to_delete]} thành công!")
                    st.experimental_rerun()  # Tự động làm mới giao diện
            else:
                st.info("Chưa có Run nào để xóa.")
    
            # 3) Danh sách các thí nghiệm
            st.subheader("Danh sách các Run đã log")
            if runs:
                selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
                                               options=list(run_options.keys()), 
                                               format_func=lambda x: run_options[x])
    
                # 4) Hiển thị thông tin chi tiết của Run được chọn
                selected_run = client.get_run(selected_run_id)
                st.write(f"**Run ID:** {selected_run_id}")
                st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")
    
                st.markdown("### Tham số đã log")
                st.json(selected_run.data.params)
    
                st.markdown("### Chỉ số đã log")
                metrics = {
                    "Mean CV Score (R²)": selected_run.data.metrics.get("mean_cv_score", "N/A"),
                    "Validation MSE": selected_run.data.metrics.get("validation_mse", "N/A"),
                    "Validation R²": selected_run.data.metrics.get("validation_r2", "N/A"),
                    "Validation Accuracy": selected_run.data.metrics.get("validation_accuracy", "N/A"),
                    "Test MSE": selected_run.data.metrics.get("test_mse", "N/A"),
                    "Test R²": selected_run.data.metrics.get("test_r2", "N/A"),
                    "Test Accuracy": selected_run.data.metrics.get("test_accuracy", "N/A")
                }
                st.json(metrics)
    
                # 5) Nút bấm mở MLflow UI
                st.subheader("Truy cập MLflow UI")
                mlflow_url = "https://dagshub.com/Dung2204/HMVPython.mlflow"
                if st.button("Mở MLflow UI"):
                    st.markdown(f'**[Click để mở MLflow UI]({mlflow_url})**')
            else:
                st.info("Chưa có Run nào được log. Vui lòng huấn luyện mô hình trước.")
    
        except Exception as e:
            st.error(f"Không thể kết nối với MLflow: {e}")    

if __name__ == "__main__":
    run_ClusteringMinst_app()    


# st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
# with st.expander("🖼️ Đánh giá hiệu suất mô hình phân cụm", expanded=True):
#     # st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
#     print("🎯 Kiểm tra trên DagsHub: https://dagshub.com/Dung2204/Minst-mlflow.mlflow")


# # # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh4"
