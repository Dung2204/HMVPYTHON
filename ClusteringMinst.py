import streamlit as st
import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
from scipy import stats
import struct
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import mlflow
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, adjusted_rand_score
from collections import Counter

def run_ClusteringMinst_app():
    @st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
    def get_sampled_pixels(images, sample_size=100_000):
        return np.random.choice(images.flatten(), sample_size, replace=False)

    @st.cache_data  # Cache danh sách ảnh ngẫu nhiên
    def get_random_indices(num_images, total_images):
        return np.random.randint(0, total_images, size=num_images)

    # Định nghĩa hàm để đọc file .idx
    def load_mnist_images(filename):
        try:
            with open(filename, 'rb') as f:
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
            return images
        except FileNotFoundError:
            st.error(f"Không tìm thấy file: {filename}")
            return np.array([])

    def load_mnist_labels(filename):
        try:
            with open(filename, 'rb') as f:
                magic, num = struct.unpack('>II', f.read(8))
                labels = np.fromfile(f, dtype=np.uint8)
            return labels
        except FileNotFoundError:
            st.error(f"Không tìm thấy file: {filename}")
            return np.array([])

    # Định nghĩa đường dẫn đến các file MNIST
    dataset_path = os.path.dirname(os.path.abspath(__file__)) 
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    # Tải dữ liệu vào session_state nếu chưa có
    if "train_images" not in st.session_state:
        st.session_state.train_images = load_mnist_images(train_images_path)
        st.session_state.train_labels = load_mnist_labels(train_labels_path)
        st.session_state.test_images = load_mnist_images(test_images_path)
        st.session_state.test_labels = load_mnist_labels(test_labels_path)

    # Tải dữ liệu
    train_images = st.session_state.train_images
    train_labels = st.session_state.train_labels
    test_images = st.session_state.test_images
    test_labels = st.session_state.test_labels

    # Kiểm tra xem dữ liệu có tải thành công không
    if train_images.size == 0 or test_images.size == 0:
        st.error("Không thể tải dữ liệu MNIST. Vui lòng kiểm tra các file dữ liệu!")
        return

    # Giao diện Streamlit
    st.title("📸 MNIST Clustering")
    tabs = st.tabs(["Thông tin", "Tập dữ liệu", "Phân cụm dữ liệu", "Thông tin & Mlflow"])
    tab_note, tab_info, tab_preprocess, tab_mlflow = tabs

    with tab_note:
        # Giữ nguyên nội dung tab_note (không liên quan đến lỗi)
        with st.expander("**Thông tin mô hình**", expanded=True):
            model_option1 = st.selectbox("Chọn mô hình", ["K-Means", "DBSCAN"])
            # ... (Giữ nguyên phần còn lại của tab_note)

    with tab_info:
        # Giữ nguyên nội dung tab_info (không liên quan đến lỗi)
        with st.expander("**Thông tin dữ liệu**", expanded=True):
            st.markdown(
                '''
                **MNIST** là phiên bản được chỉnh sửa từ bộ dữ liệu **NIST gốc** của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
                Bộ dữ liệu ban đầu gồm các chữ số viết tay từ **nhân viên bưu điện** và **học sinh trung học**.  

                Các nhà nghiên cứu **Yann LeCun, Corinna Cortes, và Christopher Burges** đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành **MNIST**  
                để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
                '''
            )
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
            st.write("🔍 Hình dạng tập huấn luyện:", train_images.shape)
            st.write("🔍 Hình dạng tập kiểm tra:", test_images.shape)
            if (train_images.min() < 0) or (train_images.max() > 255):
                st.error("⚠️ Cảnh báo: Có giá trị pixel ngoài phạm vi 0-255!")
            else:
                st.success("✅ Dữ liệu pixel hợp lệ (0 - 255).")

            train_images = train_images.astype("float32") / 255.0
            test_images = test_images.astype("float32") / 255.0
            st.success("✅ Dữ liệu đã được chuẩn hóa về khoảng [0,1].")

            num_samples = 5
            df_normalized = pd.DataFrame(train_images[:num_samples].reshape(num_samples, -1))  
            st.write("**Bảng dữ liệu sau khi chuẩn hóa**")
            st.dataframe(df_normalized)

            sample_size = 10_000  
            pixel_sample = np.random.choice(train_images.flatten(), sample_size, replace=False)

    with tab_preprocess:
        st.write("***Phân cụm dữ liệu***")
        
        sample_size = st.slider("🔹 Chọn số lượng mẫu để xử lý:", 
                                min_value=100, 
                                max_value=train_images.shape[0], 
                                value=10000, 
                                step=100, 
                                help=f"Tối đa: {train_images.shape[0]} mẫu")
        
        if st.button("🚀 Lấy mẫu dữ liệu"):
            with st.spinner("Đang lấy mẫu dữ liệu..."):
                if train_images.size == 0:
                    st.error("Lỗi: train_images rỗng! Vui lòng kiểm tra file dữ liệu.")
                    return
                random_indices = np.random.choice(train_images.shape[0], sample_size, replace=False)
                X_train = train_images[random_indices].reshape(sample_size, -1)
                if not isinstance(X_train, np.ndarray) or len(X_train.shape) != 2:
                    st.error("Lỗi: X_train không phải mảng 2D sau khi reshape!")
                    return
                st.session_state.X_train = X_train
                st.session_state.X_test = test_images.reshape(test_images.shape[0], -1)
                st.success(f"✅ Đã lấy {sample_size} mẫu từ tập huấn luyện. Shape của X_train: {X_train.shape}")

        if "X_train" in st.session_state and "X_test" in st.session_state:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test

            # Kiểm tra X_train trước khi sử dụng
            if X_train is None or not isinstance(X_train, np.ndarray):
                st.error("Lỗi: X_train chưa được khởi tạo hoặc không phải NumPy array!")
                return
            st.write(f"Shape của X_train: {X_train.shape}")

            if X_train.size == 0:
                st.error("Lỗi: X_train rỗng!")
                return
            if np.any(np.isnan(X_train)):
                X_train = np.nan_to_num(X_train, nan=0.0)
                st.warning("Đã thay thế giá trị NaN trong X_train bằng 0.")

            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Giảm chiều bằng PCA (2D) để phân cụm
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train_scaled)

            # Lưu các đối tượng vào session_state
            st.session_state.scaler = scaler
            st.session_state.pca = pca
            st.session_state.X_train_pca = X_train_pca

            clustering_method = st.selectbox("🔹 Chọn phương pháp phân cụm:", ["K-means", "DBSCAN"])

            if clustering_method == "K-means":
                k = st.slider("🔸 Số cụm (K-means)", min_value=2, max_value=20, value=10)

                if st.button("🚀 Chạy K-means"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                        
                    with mlflow.start_run():
                        start_time = time.time()

                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                            status_text.text(f"Tiến trình huấn luyện: {i + 1}%")
                            
                        labels = kmeans.fit_predict(X_train_pca)
                        progress_bar.progress(100)
                        status_text.text("Hoàn tất huấn luyện!")

                        end_time = time.time()
                        clustering_time = round(end_time - start_time, 2)

                        mlflow.log_param("algorithm", "K-means")
                        mlflow.log_param("k", k)
                        mlflow.log_param("max_iter", 300)

                        inertia = kmeans.inertia_
                        max_possible_inertia = np.sum(np.sum((X_train_pca - np.mean(X_train_pca, axis=0)) ** 2))
                        accuracy_percentage = (1 - (inertia / max_possible_inertia)) * 100 if max_possible_inertia > 0 else 0.0
                        accuracy_percentage = max(0.0, min(100.0, accuracy_percentage))
                        accuracy_percentage = round(accuracy_percentage, 2)
                        num_samples = X_train_pca.shape[0]
                        num_clusters_actual = len(set(labels))
                        mlflow.log_metric("inertia", inertia)

                        with st.container(border=True):
                            st.write("### Kết quả phân cụm và thông tin của K-Means:")
                            st.write(f"**Phương pháp phân cụm đã chọn:** K-means")
                            st.write(f"**Số cụm đã chọn:** {k}")
                            st.write(f"**Số cụm thực tế:** {num_clusters_actual}")
                            st.write(f"**Số mẫu đã xử lý:** {num_samples}")
                            st.write(f"**Thời gian phân cụm:** {clustering_time} giây")
                            st.write(f"**Độ chính xác của phân cụm K-Means:** {accuracy_percentage:.2f}%")

                            st.write("### Biểu đồ minh họa phân cụm K-Means")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=labels, palette="deep", ax=ax)
                            ax.set_title("Phân cụm K-Means (PCA 2D)")
                            ax.set_xlabel("Thành phần chính 1")
                            ax.set_ylabel("Thành phần chính 2")
                            ax.legend(title="Cụm")
                            st.pyplot(fig)

                        mlflow.sklearn.log_model(kmeans, "kmeans_model")
                        st.session_state.clustering_model = kmeans
                        st.session_state.clustering_method = "K-means"
                        st.session_state.labels = labels

                        mlflow.end_run()

            elif clustering_method == "DBSCAN":
                eps = st.slider("🔸 Bán kính vùng lân cận (eps):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider("🔸 Số lượng điểm tối thiểu:", min_value=1, max_value=20, value=10)

                st.markdown("**Tùy chọn giảm nhiễu**")
                preprocess_noise = st.checkbox("Loại bỏ ngoại lệ trước khi phân cụm (dùng Z-score)", value=False)

                if st.button("🚀 Chạy DBSCAN"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with mlflow.start_run():
                        start_time = time.time()

                        X_processed = X_train_pca.copy()

                        if preprocess_noise:
                            z_scores = np.abs(stats.zscore(X_processed))
                            threshold = 3
                            mask = (z_scores < threshold).all(axis=1)
                            X_processed = X_processed[mask]
                            st.write(f"Đã loại bỏ {X_train_pca.shape[0] - X_processed.shape[0]} điểm ngoại lệ bằng Z-score.")

                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                            status_text.text(f"Tiến trình huấn luyện: {i + 1}%")

                        labels = dbscan.fit_predict(X_processed)
                        progress_bar.progress(100)
                        status_text.text("Hoàn tất huấn luyện!")

                        end_time = time.time()
                        clustering_time = round(end_time - start_time, 2)

                        mlflow.log_param("algorithm", "DBSCAN")
                        mlflow.log_param("eps", eps)
                        mlflow.log_param("min_samples", min_samples)
                        mlflow.log_param("preprocess_noise", preprocess_noise)

                        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        noise_points = np.sum(labels == -1)
                        num_samples = X_processed.shape[0]

                        with st.container(border=True):
                            st.write("### Kết quả phân cụm và thông tin của DBSCAN:")
                            st.write(f"**Phương pháp phân cụm đã chọn:** DBSCAN")
                            st.write(f"**Số cụm đã chọn:** Không áp dụng (tự động xác định)")
                            st.write(f"**Số cụm thực tế:** {num_clusters}")
                            st.write(f"**Số mẫu đã xử lý:** {num_samples}")
                            st.write(f"**Thời gian phân cụm:** {clustering_time} giây")
                            st.write(f"**Số lượng điểm nhiễu (Noise Points):** {noise_points} ({round((noise_points / num_samples) * 100, 2)}%)")

                            st.write("### Biểu đồ minh họa phân cụm DBSCAN")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            scatter = sns.scatterplot(
                                x=X_processed[:, 0], 
                                y=X_processed[:, 1], 
                                hue=labels, 
                                palette="deep", 
                                ax=ax, 
                                legend="brief"
                            )

                            ax.set_title("Phân cụm DBSCAN (PCA 2D)")
                            ax.set_xlabel("Thành phần chính 1")
                            ax.set_ylabel("Thành phần chính 2")

                            cluster_counts = Counter(labels)
                            total_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                            max_display_clusters = 5
                            top_clusters = sorted(
                                [(cluster, count) for cluster, count in cluster_counts.items() if cluster != -1],
                                key=lambda x: x[1],
                                reverse=True
                            )[:max_display_clusters]

                            new_labels = labels.copy()
                            other_clusters = set(cluster_counts.keys()) - set([c[0] for c in top_clusters]) - {-1}
                            for cluster in other_clusters:
                                if cluster != -1:
                                    new_labels[new_labels == cluster] = -2

                            ax.clear()
                            scatter = sns.scatterplot(
                                x=X_processed[:, 0], 
                                y=X_processed[:, 1], 
                                hue=new_labels, 
                                palette="deep", 
                                ax=ax
                            )

                            ax.set_title("Phân cụm DBSCAN (PCA 2D)")
                            ax.set_xlabel("Thành phần chính 1")
                            ax.set_ylabel("Thành phần chính 2")

                            legend_labels = {label: f"Cụm {label}" for label in set(new_labels) if label != -1 and label != -2}
                            legend_labels[-1] = "Nhiễu (-1)"
                            legend_labels[-2] = f"Khác ({len(other_clusters)} cụm)"

                            handles, labels = scatter.get_legend_handles_labels()
                            labels_with_handles = [(legend_labels[int(label)] if label in legend_labels else f"Cụm {label}", handle) 
                                                for label, handle in zip(labels, handles)]
                            labels_with_handles.sort(key=lambda x: x[0].startswith("Nhiễu"), reverse=True)

                            new_handles = [item[1] for item in labels_with_handles]
                            new_labels = [item[0] for item in labels_with_handles]
                            scatter.legend(new_handles, new_labels, title=f"Tổng số cụm: {total_clusters}", 
                                        loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

                            st.pyplot(fig)
                            
                        mlflow.log_metric("num_clusters", num_clusters)
                        mlflow.log_metric("noise_points", noise_points)
                        for cluster, count in Counter(labels).items():
                            if cluster != -1:
                                mlflow.log_metric(f"cluster_{cluster}_size", count)

                        mlflow.sklearn.log_model(dbscan, "dbscan_model")
                        st.session_state.clustering_model = dbscan
                        st.session_state.clustering_method = "DBSCAN"
                        st.session_state.labels = labels

                    mlflow.end_run()

                    if noise_points / num_samples > 0.3:
                        st.warning(
                            f"Cảnh báo: Tỷ lệ nhiễu cao ({round((noise_points / num_samples) * 100, 2)}%). "
                            "Hãy thử tăng `eps` hoặc giảm `min_samples` để giảm nhiễu."
                        )
        else:
            st.info("📌 Vui lòng chọn số lượng mẫu và nhấn 'Lấy mẫu dữ liệu' để bắt đầu xử lý.")

    # Giữ nguyên tab_mlflow (nếu cần)
    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        st.info("Chức năng MLflow tạm thời bị tắt. Hãy bật lại khi cần.")

if __name__ == "__main__":
    run_ClusteringMinst_app()
