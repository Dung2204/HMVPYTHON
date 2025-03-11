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
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, adjusted_rand_score
from collections import Counter
from mlflow.tracking import MlflowClient

def run_ClusteringMinst_app():
    @st.cache_data
    def get_sampled_pixels(images, sample_size=100_000):
        return np.random.choice(images.flatten(), sample_size, replace=False)

    @st.cache_data
    def get_random_indices(num_images, total_images):
        return np.random.randint(0, total_images, size=num_images)

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
    
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)

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

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    st.title("📸 MNIST Clustering")
    tabs = st.tabs([
        "Thông tin",
        "Tập dữ liệu",
        "Phân cụm dữ liệu",
        "Thông tin & Mlflow",
    ])
    tab_note, tab_info, tab_preprocess, tab_mlflow = tabs

    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):
            model_option1 = st.selectbox("Chọn mô hình", ["K-Means", "DBSCAN"])
            
            if model_option1 == "K-Means":
                st.markdown("## 🔹 K-Means Clustering")
                st.markdown("---")
                st.markdown("**Khái niệm**")
                st.write("""
                - **K-Means** là một thuật toán phân cụm không giám sát, chia tập dữ liệu thành $K$ cụm (clusters) sao cho các điểm trong cùng một cụm gần nhau nhất, dựa trên khoảng cách (thường là khoảng cách Euclidean) đến tâm cụm (centroid).
                - Mục tiêu của K-Means là **tối thiểu hóa tổng bình phương khoảng cách** (Within-Cluster Sum of Squares - WCSS) giữa các điểm dữ liệu và tâm cụm tương ứng của chúng.
                """)
                # Giữ nguyên phần code giải thích K-Means (không liên quan StandardScaler)
                st.markdown("""
                ### 🔄 **Quy trình hoạt động của K-Means**
                - **Bước 1**: Khởi tạo ngẫu nhiên $K$ tâm cụm $\mu_1, \mu_2, ..., \mu_K$.
                - **Bước 2**: Lặp lại quá trình cập nhật tâm cụm cho tới khi dừng:
                - **a**. Xác định nhãn cho từng điểm dữ liệu $c_i$ dựa vào khoảng cách tới từng tâm cụm:
                    $$
                    c_i = arg\min_j \|x_i - \mu_j\|^2
                    $$
                **Trong đó**:  
                - $$c_i$$ là chỉ số cụm (từ 1 đến $K$).  
                - $$\|x_i - \mu_j\|^2$$ là khoảng cách Euclidean bình phương giữa điểm $x_i$ và tâm cụm $\mu_j$.
                - **b**. Tính lại tâm cụm $\mu_j$ bằng trung bình của tất cả các điểm dữ liệu thuộc cụm $j$:
                """)
                st.latex(r"\mu_j = \frac{\sum_{i=1}^{n} I(c_i = j) x_i}{\sum_{i=1}^{n} I(c_i = j)}")
                # ... (giữ nguyên phần còn lại của K-Means)

            elif model_option1 == "DBSCAN":
                st.markdown("## 🔹 DBSCAN (Density-Based Clustering)")
                st.markdown("---")
                st.markdown("**Khái niệm**")
                st.write("""
                - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là một thuật toán phân cụm dựa trên mật độ, không yêu cầu xác định trước số lượng cụm.
                - Phù hợp với dữ liệu có hình dạng cụm phức tạp và có khả năng phát hiện nhiễu (outlier).
                """)
                # Giữ nguyên phần code giải thích DBSCAN

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
                random_indices = np.random.choice(train_images.shape[0], sample_size, replace=False)
                X_train = train_images[random_indices].reshape(sample_size, -1)
                st.session_state.X_train = X_train
                st.session_state.X_test = test_images.reshape(test_images.shape[0], -1)
                st.success(f"✅ Đã lấy {sample_size} mẫu từ tập huấn luyện.")

        if "X_train" in st.session_state and "X_test" in st.session_state:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test

            # Giảm chiều bằng PCA (2D) để phân cụm
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train)

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

    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "Clustering"

            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
                st.success(f"Experiment mới được tạo với ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                st.info(f"Đang sử dụng experiment ID: {experiment_id}")

            mlflow.set_experiment(experiment_name)

            runs = client.search_runs(experiment_ids=[experiment_id])

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

            st.subheader("Danh sách Run")
            if runs:
                selected_run_id_to_delete = st.selectbox("", 
                                                        options=list(run_options.keys()), 
                                                        format_func=lambda x: run_options[x])
                if st.button("Xóa Run", key="delete_run"):
                    client.delete_run(selected_run_id_to_delete)
                    st.success(f"Đã xóa Run {run_options[selected_run_id_to_delete]} thành công!")
                    st.experimental_rerun()
            else:
                st.info("Chưa có Run nào để xóa.")

            st.subheader("Danh sách các Run đã log")
            if runs:
                selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
                                            options=list(run_options.keys()), 
                                            format_func=lambda x: run_options[x])

                selected_run = client.get_run(selected_run_id)
                st.write(f"**Run ID:** {selected_run_id}")
                st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")

                st.markdown("### Tham số đã log")
                params = {}
                algorithm = selected_run.data.params.get("algorithm", "N/A")
                params["Algorithm"] = algorithm

                if algorithm == "K-means":
                    params["K"] = selected_run.data.params.get("k", "N/A")
                    params["Max Iterations"] = selected_run.data.params.get("max_iter", "N/A")
                elif algorithm == "DBSCAN":
                    params["EPS"] = selected_run.data.params.get("eps", "N/A")
                    params["Min Samples"] = selected_run.data.params.get("min_samples", "N/A")
                    params["Preprocess Noise"] = selected_run.data.params.get("preprocess_noise", "N/A")
                
                st.json(params)

                st.markdown("### Chỉ số đã log")
                metrics = {}
                if algorithm == "K-means":
                    metrics["Inertia"] = selected_run.data.metrics.get("inertia", "N/A")
                elif algorithm == "DBSCAN":
                    metrics["Number of Clusters"] = selected_run.data.metrics.get("num_clusters", "N/A")
                    metrics["Noise Points"] = selected_run.data.metrics.get("noise_points", "N/A")
                    for key, value in selected_run.data.metrics.items():
                        if key.startswith("cluster_") and key.endswith("_size"):
                            cluster_id = key.split("_")[1]
                            metrics[f"Cluster {cluster_id} Size"] = value
                
                st.json(metrics)

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
