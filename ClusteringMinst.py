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
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay,adjusted_rand_score
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
    st.title("📸 MNIST Clustering")
    tabs = st.tabs([
            "Thông tin",
            "Tập dữ liệu",
            "Phân cụm dữ liệu",
            "Thông tin & Mlflow",
    ])
    tab_note,tab_info, tab_preprocess,tab_mlflow= tabs


    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):
            # Chọn mô hình
            model_option1 = st.selectbox("Chọn mô hình", ["K-Means", "DBSCAN"])
            
            if model_option1 == "K-Means":
                st.markdown("## 🔹 K-Means Clustering")
                st.markdown("---")

                st.markdown("**Khái niệm**")
                st.write("""
                - **K-Means** là một thuật toán phân cụm không giám sát, chia tập dữ liệu thành $K$ cụm (clusters) sao cho các điểm trong cùng một cụm gần nhau nhất, dựa trên khoảng cách (thường là khoảng cách Euclidean) đến tâm cụm (centroid).
                - Mục tiêu của K-Means là **tối thiểu hóa tổng bình phương khoảng cách** (Within-Cluster Sum of Squares - WCSS) giữa các điểm dữ liệu và tâm cụm tương ứng của chúng.
                """)

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

                st.markdown("""
                **Trong đó**: 
                - Giá trị của $$(I(c_i = j))$$ phụ thuộc vào điều kiện $$(c_i = j)$$:
                    - Nếu $$(c_i = j)$$ (tức là điểm dữ liệu thứ $$i$$ thuộc cụm $$j$$), thì $$(I(c_i = j))$$ = 1.
                    - Nếu $$(c_i ≠ j)$$ (tức là điểm dữ liệu thứ $$i$$ không thuộc cụm $$j$$), thì $$(I(c_i = j))$$ = 0.
                - $$(n)$$ là số lượng điểm dữ liệu.
                - Thuật toán dừng khi tâm cụm không thay đổi giữa các vòng lặp hoặc đạt số lần lặp tối đa (`max_iter`).
                """)

                st.markdown("### 🔄 **Minh họa quy trình K-Means dưới dạng biểu đồ**")

                # Tạo dữ liệu mẫu phức tạp hơn để thấy rõ sự khác biệt giữa các bước
                np.random.seed(42)
                means = [[2, 2], [8, 3], [4, 7]]  # Các cụm gần nhau hơn để quá trình rõ ràng
                cov = [[1.5, 0.5], [0.5, 1.5]]    # Tăng độ phân tán
                N = 150
                X0 = np.random.multivariate_normal(means[0], cov, N)
                X1 = np.random.multivariate_normal(means[1], cov, N)
                X2 = np.random.multivariate_normal(means[2], cov, N)
                X = np.concatenate((X0, X1, X2), axis=0)

                # Hàm vẽ biểu đồ với thông tin bổ sung
                def plot_kmeans_step(X, labels, centroids, step_title, iteration=None):
                    fig, ax = plt.subplots(figsize=(10, 7))
                    colors = ['blue', 'green', 'red']
                    
                    # Vẽ các điểm dữ liệu và nhãn
                    for k in range(len(centroids)):
                        cluster_points = X[labels == k]
                        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], alpha=0.5, label=f'Cụm {k} ({len(cluster_points)} điểm)')
                    
                    # Vẽ tâm cụm cũ (nếu có iteration > 0)
                    if iteration and iteration > 0:
                        old_centroids = kmeans_history[iteration - 1]['centroids']
                        ax.scatter(old_centroids[:, 0], old_centroids[:, 1], c='gray', marker='o', s=100, alpha=0.5, label='Tâm cũ')
                    
                    # Vẽ tâm cụm hiện tại
                    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, linewidths=2, label='Tâm cụm hiện tại')
                    
                    ax.set_title(step_title)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.legend()
                    ax.axis('equal')
                    return fig

                # Lưu lịch sử các bước để truy cập
                kmeans_history = []

                # Khởi tạo K-Means ban đầu
                kmeans = KMeans(n_clusters=3, init='random', n_init=1, max_iter=1, random_state=42)
                kmeans.fit(X)
                kmeans_history.append({
                    'labels': kmeans.predict(X),
                    'centroids': kmeans.cluster_centers_.copy()
                })

                # Thực hiện thêm vài bước để minh họa
                for _ in range(3):  # 3 lần lặp để thấy rõ sự thay đổi
                    kmeans = KMeans(n_clusters=3, init=kmeans_history[-1]['centroids'], n_init=1, max_iter=1, random_state=42)
                    kmeans.fit(X)
                    kmeans_history.append({
                        'labels': kmeans.labels_.copy(),
                        'centroids': kmeans.cluster_centers_.copy()
                    })

                # Hiển thị dữ liệu gốc
                st.markdown("**Dữ liệu mẫu ban đầu:**")
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.5, label=f'Dữ liệu gốc ({len(X)} điểm)')
                ax.set_title("Dữ liệu gốc trước khi phân cụm")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.axis('equal')
                st.pyplot(fig)

                # Tùy chọn bước để người dùng chọn
                step_options = [
                    "Bước 1: Khởi tạo ngẫu nhiên tâm cụm",
                    "Bước 2: Gán nhãn và cập nhật tâm cụm (Lần 1)",
                    "Bước 3: Gán nhãn và cập nhật tâm cụm (Lần 2)",
                    "Bước 4: Kết quả cuối cùng"
                ]
                selected_step = st.selectbox("Chọn bước để xem:", step_options)

                # Hiển thị bước được chọn
                step_index = step_options.index(selected_step)
                fig = plot_kmeans_step(X, kmeans_history[step_index]['labels'], kmeans_history[step_index]['centroids'], selected_step, step_index)
                st.pyplot(fig)

                # Hiển thị thông tin chi tiết
                st.markdown("**Thông tin chi tiết:**")
                for k in range(3):
                    points_in_cluster = len(X[kmeans_history[step_index]['labels'] == k])
                    centroid = kmeans_history[step_index]['centroids'][k]
                    st.write(f"- Cụm {k}: {points_in_cluster} điểm, Tâm cụm: ({centroid[0]:.2f}, {centroid[1]:.2f})")
                





                st.markdown("---")
                st.markdown("### 📐 **Công thức toán học**")
                st.write("""
                - Mục tiêu tối ưu hóa của K-Means là:  
                $$
                J = \sum_{k=1}^{K}\sum_{i=1}^{n} ||x_i - \mu_k||^2
                $$
                Trong đó:  
                - \(J\): Tổng bình phương khoảng cách (WCSS - Within-Cluster Sum of Squares).  
                - \(n\): Số lượng điểm dữ liệu.  
                - \(K\): Số cụm.  
                - $$(x_i)$$: Điểm dữ liệu thứ \(i\).  
                - $$(\mu_k)$$: Tâm cụm của cụm \(k\).  
                - $$(||x_i - \mu_k||^2)$$: Khoảng cách Euclidean bình phương giữa điểm $$(x_i)$$ và tâm cụm $$(\mu_k)$$.
                """)
                st.markdown("---")
                st.markdown("### 🔧 **Một số cải tiến của K-Means**")
                st.write("""
                - **Mini-Batch K-Means**: Sử dụng các batch nhỏ của dữ liệu để cập nhật tâm cụm, giúp giảm thời gian tính toán trên dữ liệu lớn.
                - **K-Means với chuẩn hóa dữ liệu**: Chuẩn hóa (scaling) dữ liệu trước khi áp dụng K-Means để tránh đặc trưng có thang đo lớn ảnh hưởng đến kết quả phân cụm.
                """)
                

                st.markdown("---")
                st.markdown("### 👍 **Ưu điểm**")
                st.write("""
                - Đơn giản, dễ triển khai và tính toán nhanh với dữ liệu nhỏ hoặc trung bình.
                - Hiệu quả khi các cụm có hình dạng cầu (spherical) và kích thước tương đương.
                """)

                st.markdown("### ⚠️ **Nhược điểm**")
                st.write("""
                - Cần chọn trước số cụm $K$, thường sử dụng phương pháp Elbow hoặc Silhouette để ước lượng.
                - Nhạy cảm với giá trị ban đầu của tâm cụm, có thể dẫn đến kết quả khác nhau.
                - Không hoạt động tốt nếu cụm có hình dạng phức tạp (không phải hình cầu) hoặc có kích thước, mật độ khác nhau.
                - Nhạy cảm với nhiễu (outliers) và dữ liệu có thang đo khác nhau (yêu cầu chuẩn hóa).
                """)

            elif model_option1 == "DBSCAN":
                st.markdown("## 🔹 DBSCAN (Density-Based Clustering)")
                st.markdown("---")

                st.markdown("**Khái niệm**")
                st.write("""
                - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là một thuật toán phân cụm dựa trên mật độ, không yêu cầu xác định trước số lượng cụm.
                - Phù hợp với dữ liệu có hình dạng cụm phức tạp và có khả năng phát hiện nhiễu (outlier).
                """)
                st.markdown("---")
                st.markdown("### 🔄 **Minh họa quy trình hoạt động của DBSCAN**")

                # Quy trình hoạt động
                st.write("""
                - **Bước 1:** Chọn ngẫu nhiên một điểm dữ liệu chưa được thăm (unvisited).
                - **Bước 2:** Kiểm tra số lượng điểm trong vùng lân cận bán kính `eps` của điểm đó:
                    - Nếu số điểm >= `min_samples`, tạo một cụm mới và thêm điểm này vào cụm.
                    - Nếu không, đánh dấu điểm này là nhiễu (noise).
                - **Bước 3:** Với mỗi điểm trong cụm, kiểm tra lân cận của nó:
                    - Nếu tìm thấy các điểm lân cận mới thỏa mãn `min_samples`, thêm chúng vào cụm và tiếp tục mở rộng.
                - **Bước 4:** Lặp lại quá trình cho đến khi tất cả các điểm được thăm hoặc phân loại.
                - **Bước 5:** Kết thúc khi không còn điểm nào chưa được xử lý.
                """)

                np.random.seed(42)
                means = [[2, 2], [8, 3], [4, 7]]
                cov = [[1.5, 0.5], [0.5, 1.5]]
                N = 100
                X0 = np.random.multivariate_normal(means[0], cov, N)
                X1 = np.random.multivariate_normal(means[1], cov, N)
                X2 = np.random.multivariate_normal(means[2], cov, N)
                X = np.concatenate((X0, X1, X2), axis=0)
                noise = np.random.uniform(low=-2, high=10, size=(20, 2))
                X = np.vstack((X, noise))

                # Hàm vẽ biểu đồ
                def plot_dbscan_step(X, labels, visited, core_points, border_points, current_point=None, eps=None, step_title=""):
                    fig, ax = plt.subplots(figsize=(10, 7))
                    
                    # Vẽ điểm chưa thăm
                    unvisited = X[~visited]
                    ax.scatter(unvisited[:, 0], unvisited[:, 1], c='lightgray', alpha=0.3, label=f'Chưa thăm ({len(unvisited)})')
                    
                    # Vẽ điểm nhiễu
                    noise_points = X[labels == -1]
                    ax.scatter(noise_points[:, 0], noise_points[:, 1], c='gray', marker='x', alpha=0.5, label=f'Nhiễu ({len(noise_points)})')
                    
                    # Vẽ các cụm
                    unique_labels = set(labels) - {-1}
                    colors = ['blue', 'green', 'red', 'purple', 'orange']
                    for label in unique_labels:
                        cluster_points = X[labels == label]
                        core = X[(labels == label) & np.isin(np.arange(len(X)), core_points)]
                        border = X[(labels == label) & np.isin(np.arange(len(X)), border_points)]
                        ax.scatter(core[:, 0], core[:, 1], c=colors[label % len(colors)], marker='o', alpha=0.7, label=f'Cụm {label} - Lõi ({len(core)})')
                        ax.scatter(border[:, 0], border[:, 1], c=colors[label % len(colors)], marker='s', alpha=0.5, label=f'Cụm {label} - Biên ({len(border)})')

                    # Đánh dấu điểm đang xét và vùng lân cận
                    if current_point is not None and eps is not None:
                        ax.scatter(current_point[0], current_point[1], c='black', marker='*', s=200, label='Điểm đang xét')
                        circle = plt.Circle(current_point, eps, color='black', fill=False, linestyle='--')
                        ax.add_artist(circle)

                    ax.set_title(step_title)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.axis('equal')
                    plt.tight_layout()
                    return fig

                # Hàm reset trạng thái
                def reset_dbscan_state(X):
                    return {
                        'visited': np.zeros(X.shape[0], dtype=bool),
                        'labels': np.full(X.shape[0], -1, dtype=int),
                        'core_points': [],
                        'border_points': [],
                        'current_idx': None,
                        'cluster_label': 0
                    }

                # Khởi tạo trạng thái trong session_state
                if 'dbscan_state' not in st.session_state:
                    st.session_state.dbscan_state = reset_dbscan_state(X)

                state = st.session_state.dbscan_state

                # Tham số DBSCAN
                eps = st.slider("Bán kính vùng lân cận (eps)", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.number_input("Số điểm tối thiểu (min_samples)", min_value=1, max_value=20, value=5)

                # Hàm mô phỏng từng bước
                def simulate_dbscan_step(X, eps, min_samples, step, state):
                    # Reset state nếu quay lại bước trước sau khi chạy bước 4 hoặc 5
                    if step < state.get('last_step', -1) and state.get('last_step', -1) >= 3:
                        st.session_state.dbscan_state = reset_dbscan_state(X)
                        state = st.session_state.dbscan_state

                    # Đảm bảo các khóa cần thiết luôn tồn tại
                    if 'core_points' not in state:
                        state['core_points'] = []
                    if 'border_points' not in state:
                        state['border_points'] = []

                    if step == 0:  # Bước 1: Chọn điểm ngẫu nhiên
                        if np.any(~state['visited']):
                            current_idx = np.random.choice(np.where(~state['visited'])[0])
                            state['current_idx'] = current_idx
                            state['visited'][current_idx] = True
                            nbrs = NearestNeighbors(radius=eps).fit(X)
                            distances, indices = nbrs.radius_neighbors([X[current_idx]])
                            neighbors = indices[0]
                            state['last_step'] = step
                            return current_idx, len(neighbors) >= min_samples, neighbors
                        return None, False, []

                    elif step == 1:  # Bước 2: Kiểm tra vùng lân cận
                        current_idx = state['current_idx']
                        if current_idx is None:
                            return None, False, []
                        nbrs = NearestNeighbors(radius=eps).fit(X)
                        distances, indices = nbrs.radius_neighbors([X[current_idx]])
                        neighbors = indices[0]
                        is_core = len(neighbors) >= min_samples
                        if is_core:
                            state['labels'][current_idx] = state['cluster_label']
                            state['core_points'].append(current_idx)
                        else:
                            state['labels'][current_idx] = -1  # Nhiễu
                        state['last_step'] = step
                        return current_idx, is_core, neighbors

                    elif step == 2:  # Bước 3: Mở rộng cụm
                        current_idx = state['current_idx']
                        if current_idx is None or state['labels'][current_idx] == -1:
                            return None, False, []
                        nbrs = NearestNeighbors(radius=eps).fit(X)
                        distances, indices = nbrs.radius_neighbors([X[current_idx]])
                        neighbors = indices[0]
                        for idx in neighbors:
                            if not state['visited'][idx]:
                                state['visited'][idx] = True
                                new_nbrs = NearestNeighbors(radius=eps).fit(X)
                                _, new_neighbors = new_nbrs.radius_neighbors([X[idx]])
                                if len(new_neighbors[0]) >= min_samples:
                                    state['labels'][idx] = state['cluster_label']
                                    state['core_points'].append(idx)
                                else:
                                    state['labels'][idx] = state['cluster_label']
                                    state['border_points'].append(idx)
                        state['last_step'] = step
                        return current_idx, True, neighbors

                    elif step == 3:  # Bước 4: Lặp lại toàn bộ
                        while np.any(~state['visited']):
                            current_idx = np.random.choice(np.where(~state['visited'])[0])
                            state['visited'][current_idx] = True
                            nbrs = NearestNeighbors(radius=eps).fit(X)
                            distances, indices = nbrs.radius_neighbors([X[current_idx]])
                            neighbors = indices[0]
                            if len(neighbors) >= min_samples:
                                state['cluster_label'] += 1
                                state['labels'][current_idx] = state['cluster_label']
                                state['core_points'].append(current_idx)
                                for idx in neighbors:
                                    if not state['visited'][idx]:
                                        state['visited'][idx] = True
                                        new_nbrs = NearestNeighbors(radius=eps).fit(X)
                                        _, new_neighbors = new_nbrs.radius_neighbors([X[idx]])
                                        if len(new_neighbors[0]) >= min_samples:
                                            state['labels'][idx] = state['cluster_label']
                                            state['core_points'].append(idx)
                                        else:
                                            state['labels'][idx] = state['cluster_label']
                                            state['border_points'].append(idx)
                        state['last_step'] = step
                        return None, True, []

                    elif step == 4:  # Bước 5: Kết quả cuối cùng
                        dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                        labels = dbscan.fit_predict(X)
                        core_points = list(dbscan.core_sample_indices_)  # Chuyển thành list
                        border_points = [i for i in range(len(X)) if labels[i] != -1 and i not in core_points]
                        state['labels'] = labels
                        state['core_points'] = core_points
                        state['border_points'] = border_points
                        state['last_step'] = step
                        return None, True, []

                # Hiển thị dữ liệu gốc
                st.markdown("**Dữ liệu mẫu ban đầu:**")
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.5, label=f'Dữ liệu gốc ({len(X)} điểm)')
                ax.set_title("Dữ liệu gốc trước khi phân cụm")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.legend()
                ax.axis('equal')
                st.pyplot(fig)

                # Tùy chọn bước
                step_options = [
                    "Bước 1: Chọn điểm ngẫu nhiên",
                    "Bước 2: Kiểm tra vùng lân cận",
                    "Bước 3: Mở rộng cụm",
                    "Bước 4: Lặp lại toàn bộ",
                    "Bước 5: Kết quả cuối cùng"
                ]
                selected_step = st.selectbox("Chọn bước để xem:", step_options)
                step_index = step_options.index(selected_step)

                # Chạy bước được chọn
                current_idx, is_core, neighbors = simulate_dbscan_step(X, eps, min_samples, step_index, state)

                if step_index < 4 and current_idx is not None:
                    fig = plot_dbscan_step(X, state['labels'], state['visited'], state['core_points'], state['border_points'], X[current_idx], eps, step_options[step_index])
                    st.pyplot(fig)
                    if step_index == 1:
                        st.write(f"Điểm này có {len(neighbors)} lân cận, {'là điểm lõi' if is_core else 'là nhiễu'}")
                    elif step_index == 2:
                        st.write(f"Đang mở rộng cụm {state['cluster_label']} với {len(neighbors)} điểm lân cận.")
                elif step_index == 4:
                    fig = plot_dbscan_step(X, state['labels'], state['visited'], state['core_points'], state['border_points'], None, None, step_options[step_index])
                    st.pyplot(fig)
                    n_clusters = len(set(state['labels']) - {-1})
                    n_noise = list(state['labels']).count(-1)
                    st.write(f"**Số cụm:** {n_clusters}")
                    st.write(f"**Số điểm nhiễu:** {n_noise}")
                    st.write(f"**Số điểm lõi:** {len(state['core_points'])}")
                    st.write(f"**Số điểm biên:** {len(state['border_points'])}")
                else:
                    st.write("Đã xử lý toàn bộ dữ liệu. Chọn 'Bước 5' để xem kết quả cuối cùng.")
                                                                



                
                st.markdown("---")
                st.markdown("### 📐 **Công thức toán học**")
                st.write("""
                DBSCAN sử dụng khoảng cách **Euclidean** để xác định điểm lân cận, được tính bằng công thức:
                $$
                d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
                $$
                **trong đó:**
                - $$d( p , q )$$ là hai điểm trong không gian \( n \) chiều.
                - $$(p_i)$$ và $$(q_i)$$ và là tọa độ của $$(p)$$ và $$(q)$$ điểm trong không gian n chiều.
                """)
                st.markdown("---")
                st.markdown("### 🔧 **Một số cải tiến**")
                st.write("""
                - **OPTICS**: Mở rộng DBSCAN để xử lý dữ liệu có mật độ thay đổi, tạo ra thứ tự phân cấp các cụm.
                - **HDBSCAN**: Kết hợp phân cụm phân cấp với DBSCAN, tự động chọn `eps` và cải thiện hiệu quả trên dữ liệu phức tạp.
                - **GDBSCAN**: Tổng quát hóa DBSCAN để áp dụng cho các loại dữ liệu không gian khác nhau.
                """)

                st.markdown("---")
                st.markdown("### 👍 **Ưu điểm**")
                st.write("""
                - Không cần xác định trước số lượng cụm.
                - Phát hiện tự động các điểm nhiễu (outlier).
                - Hiệu quả với các cụm có hình dạng bất kỳ (không cần giả định hình cầu như K-Means).
                """)

                st.markdown("### ⚠️ **Nhược điểm**")
                st.write("""
                - Nhạy cảm với tham số `eps` và `min_samples`: Chọn sai có thể dẫn đến kết quả không tối ưu.
                - Hiệu suất giảm khi mật độ dữ liệu không đồng đều hoặc dữ liệu có chiều cao (curse of dimensionality).
                - Tốn kém về tính toán với tập dữ liệu lớn (độ phức tạp \( O(n^2) \) nếu không dùng chỉ mục không gian).
                """)




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


    with tab_preprocess:
        st.write("***Phân cụm dữ liệu***")
        
        # Thêm phần chọn số lượng mẫu
        sample_size = st.slider("🔹 Chọn số lượng mẫu để xử lý:", 
                                min_value=100, 
                                max_value=train_images.shape[0], 
                                value=10000, 
                                step=100, 
                                help=f"Tối đa: {train_images.shape[0]} mẫu")
        
        # Lấy dữ liệu trực tiếp từ train_images và train_labels
        if st.button("🚀 Lấy mẫu dữ liệu"):
            with st.spinner("Đang lấy mẫu dữ liệu..."):
                random_indices = np.random.choice(train_images.shape[0], sample_size, replace=False)
                X_train = train_images[random_indices].reshape(sample_size, -1)
                st.session_state.X_train = X_train
                st.session_state.X_test = test_images.reshape(test_images.shape[0], -1)
                st.success(f"✅ Đã lấy {sample_size} mẫu từ tập huấn luyện.")

        # Kiểm tra xem dữ liệu đã được lấy chưa
        if "X_train" in st.session_state and "X_test" in st.session_state:
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test

            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Giảm chiều bằng PCA (2D) để phân cụm
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train_scaled)

            # Lưu các đối tượng vào session_state để sử dụng sau
            st.session_state.scaler = scaler
            st.session_state.pca = pca
            st.session_state.X_train_pca = X_train_pca

            # Chọn phương pháp phân cụm
            clustering_method = st.selectbox("🔹 Chọn phương pháp phân cụm:", ["K-means", "DBSCAN"])

            if clustering_method == "K-means":
                k = st.slider("🔸 Số cụm (K-means)", min_value=2, max_value=20, value=10)

                if st.button("🚀 Chạy K-means"):
                        # Khởi tạo thanh tiến trình
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                        
                    with mlflow.start_run():
                        start_time = time.time()

                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            
                            # Giả lập tiến trình (vì KMeans không cung cấp callback chi tiết)
                        for i in range(100):
                            time.sleep(0.01)  # Giả lập thời gian xử lý nhỏ
                            progress_bar.progress(i + 1)
                            status_text.text(f"Tiến trình huấn luyện: {i + 1}%")
                            
                            # Huấn luyện mô hình
                        labels = kmeans.fit_predict(X_train_pca)
                        progress_bar.progress(100)  # Đảm bảo thanh đạt 100%
                        status_text.text("Hoàn tất huấn luyện!")

                        end_time = time.time()
                        clustering_time = round(end_time - start_time, 2)

                        mlflow.log_param("algorithm", "K-means")
                        mlflow.log_param("k", k)
                        mlflow.log_param("max_iter", 300)

                        # Tính Inertia và độ chính xác
                        inertia = kmeans.inertia_
                        max_possible_inertia = np.sum(np.sum((X_train_pca - np.mean(X_train_pca, axis=0)) ** 2))
                        accuracy_percentage = (1 - (inertia / max_possible_inertia)) * 100 if max_possible_inertia > 0 else 0.0
                        accuracy_percentage = max(0.0, min(100.0, accuracy_percentage))
                        accuracy_percentage = round(accuracy_percentage, 2)
                        num_samples = X_train_pca.shape[0]
                        num_clusters_actual = len(set(labels))
                        mlflow.log_metric("inertia", inertia)
                        # Hiển thị kết quả
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
                # Tham số DBSCAN
                eps = st.slider("🔸 Bán kính vùng lân cận (eps):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider("🔸 Số lượng điểm tối thiểu:", min_value=1, max_value=20, value=10)

                # Thêm tùy chọn tiền xử lý để giảm nhiễu
                st.markdown("**Tùy chọn giảm nhiễu**")
                preprocess_noise = st.checkbox("Loại bỏ ngoại lệ trước khi phân cụm (dùng Z-score)", value=False)
                # normalize_data = st.checkbox("Chuẩn hóa dữ liệu (StandardScaler)", value=True)

                if st.button("🚀 Chạy DBSCAN"):
                    # Khởi tạo thanh tiến trình
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    with mlflow.start_run():
                        start_time = time.time()

                        # Sao chép dữ liệu để xử lý
                        X_processed = X_train_pca.copy()

                        # 1. Tiền xử lý: Loại bỏ ngoại lệ bằng Z-score nếu được chọn
                        if preprocess_noise:
                            z_scores = np.abs(stats.zscore(X_processed))
                            threshold = 3  # Ngưỡng Z-score (có thể điều chỉnh)
                            mask = (z_scores < threshold).all(axis=1)
                            X_processed = X_processed[mask]
                            st.write(f"Đã loại bỏ {X_train_pca.shape[0] - X_processed.shape[0]} điểm ngoại lệ bằng Z-score.")

                        # # 2. Chuẩn hóa dữ liệu nếu được chọn
                        # if normalize_data:
                        #     scaler = StandardScaler()
                        #     X_processed = scaler.fit_transform(X_processed)
                        #     st.write("Dữ liệu đã được chuẩn hóa bằng StandardScaler.")

                        # Khởi tạo DBSCAN
                        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

                        # Giả lập tiến trình
                        for i in range(100):
                            time.sleep(0.01)  # Giả lập thời gian xử lý nhỏ
                            progress_bar.progress(i + 1)
                            status_text.text(f"Tiến trình huấn luyện: {i + 1}%")

                        # Huấn luyện mô hình
                        labels = dbscan.fit_predict(X_processed)
                        progress_bar.progress(100)  # Đảm bảo thanh đạt 100%
                        status_text.text("Hoàn tất huấn luyện!")

                        end_time = time.time()
                        clustering_time = round(end_time - start_time, 2)

                        # Ghi log tham số vào MLflow
                        mlflow.log_param("algorithm", "DBSCAN")
                        mlflow.log_param("eps", eps)
                        mlflow.log_param("min_samples", min_samples)
                        mlflow.log_param("preprocess_noise", preprocess_noise)
                        # mlflow.log_param("normalize_data", normalize_data)

                        # Tính toán số cụm và nhiễu
                        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        noise_points = np.sum(labels == -1)
                        num_samples = X_processed.shape[0]

                        # Hiển thị kết quả
                        with st.container(border=True):
                            st.write("### Kết quả phân cụm và thông tin của DBSCAN:")
                            st.write(f"**Phương pháp phân cụm đã chọn:** DBSCAN")
                            st.write(f"**Số cụm đã chọn:** Không áp dụng (tự động xác định)")
                            st.write(f"**Số cụm thực tế:** {num_clusters}")
                            st.write(f"**Số mẫu đã xử lý:** {num_samples}")
                            st.write(f"**Thời gian phân cụm:** {clustering_time} giây")
                            st.write(f"**Số lượng điểm nhiễu (Noise Points):** {noise_points} ({round((noise_points / num_samples) * 100, 2)}%)")

                            # Thêm biểu đồ minh họa phân cụm
                            st.write("### Biểu đồ minh họa phân cụm DBSCAN")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.scatterplot(x=X_processed[:, 0], y=X_processed[:, 1], hue=labels, palette="deep", ax=ax)
                            ax.set_title("Phân cụm DBSCAN (PCA 2D)")
                            ax.set_xlabel("Thành phần chính 1")
                            ax.set_ylabel("Thành phần chính 2")
                            ax.legend(title="Cụm (-1 là nhiễu)")
                            # st.pyplot(fig)
                            scatter = sns.scatterplot(
                                x=X_processed[:, 0], 
                                y=X_processed[:, 1], 
                                hue=labels, 
                                palette="deep", 
                                ax=ax, 
                                legend="brief"  # Chỉ hiển thị một số nhãn trong legend
                            )

                            # Tùy chỉnh tiêu đề và nhãn trục
                            ax.set_title("Phân cụm DBSCAN (PCA 2D)")
                            ax.set_xlabel("Thành phần chính 1")
                            ax.set_ylabel("Thành phần chính 2")

                            # Đếm số lượng điểm trong mỗi cụm
                            cluster_counts = Counter(labels)
                            total_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                            # Tùy chỉnh legend
                            # Hiển thị nhiễu (-1) và tối đa 5 cụm lớn nhất, phần còn lại gộp vào "Others"
                            max_display_clusters = 5
                            top_clusters = sorted(
                                [(cluster, count) for cluster, count in cluster_counts.items() if cluster != -1],
                                key=lambda x: x[1],
                                reverse=True
                            )[:max_display_clusters]

                            # Tạo nhãn mới cho legend
                            new_labels = labels.copy()
                            other_clusters = set(cluster_counts.keys()) - set([c[0] for c in top_clusters]) - {-1}
                            for cluster in other_clusters:
                                if cluster != -1:
                                    new_labels[new_labels == cluster] = -2  # Gộp các cụm nhỏ vào "Others"

                            # Cập nhật scatter plot với nhãn mới
                            # Tạo lại scatter plot để cập nhật legend
                            ax.clear()
                            scatter = sns.scatterplot(
                                x=X_processed[:, 0], 
                                y=X_processed[:, 1], 
                                hue=new_labels, 
                                palette="deep", 
                                ax=ax
                            )

                            # Tùy chỉnh lại tiêu đề và nhãn trục
                            ax.set_title("Phân cụm DBSCAN (PCA 2D)")
                            ax.set_xlabel("Thành phần chính 1")
                            ax.set_ylabel("Thành phần chính 2")

                            # Tùy chỉnh legend
                            legend_labels = {label: f"Cụm {label}" for label in set(new_labels) if label != -1 and label != -2}
                            legend_labels[-1] = "Nhiễu (-1)"
                            legend_labels[-2] = f"Khác ({len(other_clusters)} cụm)"

                            # Sắp xếp lại legend để "Nhiễu" lên đầu
                            handles, labels = scatter.get_legend_handles_labels()
                            labels_with_handles = [(legend_labels[int(label)] if label in legend_labels else f"Cụm {label}", handle) 
                                                for label, handle in zip(labels, handles)]
                            labels_with_handles.sort(key=lambda x: x[0].startswith("Nhiễu"), reverse=True)  # Đưa "Nhiễu" lên đầu

                            # Tạo legend mới
                            new_handles = [item[1] for item in labels_with_handles]
                            new_labels = [item[0] for item in labels_with_handles]
                            scatter.legend(new_handles, new_labels, title=f"Tổng số cụm: {total_clusters}", 
                                        loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

                            # Hiển thị biểu đồ
                            st.pyplot(fig)
                            
                        # Ghi log các chỉ số vào MLflow
                        mlflow.log_metric("num_clusters", num_clusters)
                        mlflow.log_metric("noise_points", noise_points)
                        for cluster, count in Counter(labels).items():
                            if cluster != -1:
                                mlflow.log_metric(f"cluster_{cluster}_size", count)

                        # Lưu mô hình
                        mlflow.sklearn.log_model(dbscan, "dbscan_model")
                        st.session_state.clustering_model = dbscan
                        st.session_state.clustering_method = "DBSCAN"
                        st.session_state.labels = labels

                    mlflow.end_run()

                    # Gợi ý điều chỉnh tham số nếu nhiễu vẫn cao
                    if noise_points / num_samples > 0.3:  # Nếu nhiễu chiếm hơn 30%
                        st.warning(
                            f"Cảnh báo: Tỷ lệ nhiễu cao ({round((noise_points / num_samples) * 100, 2)}%). "
                            "Hãy thử tăng `eps` hoặc giảm `min_samples` để giảm nhiễu."
                        )
        else:
            st.info("📌 Vui lòng chọn số lượng mẫu và nhấn 'Lấy mẫu dữ liệu' để bắt đầu xử lý.")
    


    # with tab_mlflow:
    #     st.header("Thông tin Huấn luyện & MLflow UI")
    #     try:
    #         client = MlflowClient()
    #         experiment_name = "Clustering"

    #         # Kiểm tra nếu experiment đã tồn tại
    #         experiment = client.get_experiment_by_name(experiment_name)
    #         if experiment is None:
    #             experiment_id = client.create_experiment(experiment_name)
    #             st.success(f"Experiment mới được tạo với ID: {experiment_id}")
    #         else:
    #             experiment_id = experiment.experiment_id
    #             st.info(f"Đang sử dụng experiment ID: {experiment_id}")

    #         mlflow.set_experiment(experiment_name)

    #         # Truy vấn các run trong experiment
    #         runs = client.search_runs(experiment_ids=[experiment_id])

    #         # 1) Chọn và đổi tên Run Name
    #         st.subheader("Đổi tên Run")
    #         if runs:
    #             run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
    #                         for run in runs}
    #             selected_run_id_for_rename = st.selectbox("Chọn Run để đổi tên:", 
    #                                                     options=list(run_options.keys()), 
    #                                                     format_func=lambda x: run_options[x])
    #             new_run_name = st.text_input("Nhập tên mới cho Run:", 
    #                                         value=run_options[selected_run_id_for_rename].split(" - ")[0])
    #             if st.button("Cập nhật tên Run"):
    #                 if new_run_name.strip():
    #                     client.set_tag(selected_run_id_for_rename, "mlflow.runName", new_run_name.strip())
    #                     st.success(f"Đã cập nhật tên Run thành: {new_run_name.strip()}")
    #                 else:
    #                     st.warning("Vui lòng nhập tên mới cho Run.")
    #         else:
    #             st.info("Chưa có Run nào được log.")

    #         # 2) Xóa Run
    #         st.subheader("Danh sách Run")
    #         if runs:
    #             selected_run_id_to_delete = st.selectbox("", 
    #                                                     options=list(run_options.keys()), 
    #                                                     format_func=lambda x: run_options[x])
    #             if st.button("Xóa Run", key="delete_run"):
    #                 client.delete_run(selected_run_id_to_delete)
    #                 st.success(f"Đã xóa Run {run_options[selected_run_id_to_delete]} thành công!")
    #                 st.experimental_rerun()  # Tự động làm mới giao diện
    #         else:
    #             st.info("Chưa có Run nào để xóa.")

    #         # 3) Danh sách các thí nghiệm
    #         st.subheader("Danh sách các Run đã log")
    #         if runs:
    #             selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
    #                                         options=list(run_options.keys()), 
    #                                         format_func=lambda x: run_options[x])

    #             # 4) Hiển thị thông tin chi tiết của Run được chọn
    #             selected_run = client.get_run(selected_run_id)
    #             st.write(f"**Run ID:** {selected_run_id}")
    #             st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")

    #             # Hiển thị tham số đã log
    #             st.markdown("### Tham số đã log")
    #             params = {}
    #             algorithm = selected_run.data.params.get("algorithm", "N/A")
    #             params["Algorithm"] = algorithm

    #             if algorithm == "K-means":
    #                 params["K"] = selected_run.data.params.get("k", "N/A")
    #                 params["Max Iterations"] = selected_run.data.params.get("max_iter", "N/A")
    #             elif algorithm == "DBSCAN":
    #                 params["EPS"] = selected_run.data.params.get("eps", "N/A")
    #                 params["Min Samples"] = selected_run.data.params.get("min_samples", "N/A")
    #                 params["Preprocess Noise"] = selected_run.data.params.get("preprocess_noise", "N/A")
    #                 # params["Normalize Data"] = selected_run.data.params.get("normalize_data", "N/A")
                
    #             st.json(params)

    #             # Hiển thị chỉ số đã log
    #             st.markdown("### Chỉ số đã log")
    #             metrics = {}
    #             if algorithm == "K-means":
    #                 metrics["Inertia"] = selected_run.data.metrics.get("inertia", "N/A")
    #             elif algorithm == "DBSCAN":
    #                 metrics["Number of Clusters"] = selected_run.data.metrics.get("num_clusters", "N/A")
    #                 metrics["Noise Points"] = selected_run.data.metrics.get("noise_points", "N/A")
    #                 # Thêm kích thước của từng cụm nếu có
    #                 for key, value in selected_run.data.metrics.items():
    #                     if key.startswith("cluster_") and key.endswith("_size"):
    #                         cluster_id = key.split("_")[1]
    #                         metrics[f"Cluster {cluster_id} Size"] = value
                
    #             st.json(metrics)

    #             # 5) Nút bấm mở MLflow UI
    #             st.subheader("Truy cập MLflow UI")
    #             mlflow_url = "https://dagshub.com/Dung2204/HMVPython.mlflow"
    #             if st.button("Mở MLflow UI"):
    #                 st.markdown(f'**[Click để mở MLflow UI]({mlflow_url})**')
    #         else:
    #             st.info("Chưa có Run nào được log. Vui lòng huấn luyện mô hình trước.")

    #     except Exception as e:
    #         st.error(f"Không thể kết nối với MLflow: {e}")
    # with tab_mlflow:
    #     st.header("Thông tin Huấn luyện & MLflow UI")
    #     try:
    #         client = MlflowClient()
    #         experiment_name = "Clustering"

    #         # Kiểm tra nếu experiment đã tồn tại
    #         experiment = client.get_experiment_by_name(experiment_name)
    #         if experiment is None:
    #             experiment_id = client.create_experiment(experiment_name)
    #             st.success(f"Experiment mới được tạo với ID: {experiment_id}")
    #         else:
    #             experiment_id = experiment.experiment_id
    #             st.info(f"Đang sử dụng experiment ID: {experiment_id}")

    #         mlflow.set_experiment(experiment_name)

    #         # Truy vấn các run trong experiment
    #         runs = client.search_runs(experiment_ids=[experiment_id])

    #         # 1) Chọn và đổi tên Run Name
    #         st.subheader("Đổi tên Run")
    #         if runs:
    #             run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
    #                         for run in runs}
    #             selected_run_id_for_rename = st.selectbox("Chọn Run để đổi tên:", 
    #                                                     options=list(run_options.keys()), 
    #                                                     format_func=lambda x: run_options[x])
    #             new_run_name = st.text_input("Nhập tên mới cho Run:", 
    #                                         value=run_options[selected_run_id_for_rename].split(" - ")[0])
    #             if st.button("Cập nhật tên Run"):
    #                 if new_run_name.strip():
    #                     client.set_tag(selected_run_id_for_rename, "mlflow.runName", new_run_name.strip())
    #                     st.success(f"Đã cập nhật tên Run thành: {new_run_name.strip()}")
    #                 else:
    #                     st.warning("Vui lòng nhập tên mới cho Run.")
    #         else:
    #             st.info("Chưa có Run nào được log.")

    #         # 2) Xóa Run
    #         st.subheader("Danh sách Run")
    #         if runs:
    #             selected_run_id_to_delete = st.selectbox("", 
    #                                                     options=list(run_options.keys()), 
    #                                                     format_func=lambda x: run_options[x])
    #             if st.button("Xóa Run", key="delete_run"):
    #                 client.delete_run(selected_run_id_to_delete)
    #                 st.success(f"Đã xóa Run {run_options[selected_run_id_to_delete]} thành công!")
    #                 st.experimental_rerun()  # Tự động làm mới giao diện
    #         else:
    #             st.info("Chưa có Run nào để xóa.")

    #         # 3) Danh sách các thí nghiệm và thông tin chi tiết
    #         st.subheader("Danh sách các Run đã log")
    #         if runs:
    #             selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
    #                                         options=list(run_options.keys()), 
    #                                         format_func=lambda x: run_options[x])

    #             # 4) Hiển thị thông tin chi tiết của Run được chọn
    #             selected_run = client.get_run(selected_run_id)
    #             st.write(f"**Run ID:** {selected_run_id}")
    #             st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")

    #             # Hiển thị tham số đã log
    #             st.markdown("### Tham số đã log")
    #             params = {}
    #             algorithm = selected_run.data.params.get("algorithm", "N/A")
    #             params["Algorithm"] = algorithm

    #             if algorithm == "K-means":
    #                 params["K"] = selected_run.data.params.get("k", "N/A")
    #                 params["Max Iterations"] = selected_run.data.params.get("max_iter", "N/A")
    #             elif algorithm == "DBSCAN":
    #                 params["EPS"] = selected_run.data.params.get("eps", "N/A")
    #                 params["Min Samples"] = selected_run.data.params.get("min_samples", "N/A")
    #                 params["Preprocess Noise"] = selected_run.data.params.get("preprocess_noise", "N/A")
    #                 # params["Normalize Data"] = selected_run.data.params.get("normalize_data", "N/A")
                
    #             st.json(params)

    #             # Hiển thị chỉ số đã log
    #             st.markdown("### Chỉ số đã log")
    #             metrics = {}
    #             if algorithm == "K-means":
    #                 metrics["Inertia"] = selected_run.data.metrics.get("inertia", "N/A")
    #             elif algorithm == "DBSCAN":
    #                 metrics["Number of Clusters"] = selected_run.data.metrics.get("num_clusters", "N/A")
    #                 metrics["Noise Points"] = selected_run.data.metrics.get("noise_points", "N/A")
    #                 # Thêm kích thước của từng cụm nếu có
    #                 for key, value in selected_run.data.metrics.items():
    #                     if key.startswith("cluster_") and key.endswith("_size"):
    #                         cluster_id = key.split("_")[1]
    #                         metrics[f"Cluster {cluster_id} Size"] = value
                
    #             st.json(metrics)

    #             # 5) Nút bấm mở MLflow UI
    #             st.subheader("Truy cập MLflow UI")
    #             mlflow_url = "https://dagshub.com/Dung2204/HMVPython.mlflow"
    #             if st.button("Mở MLflow UI"):
    #                 st.markdown(f'**[Click để mở MLflow UI]({mlflow_url})**')
    #         else:
    #             st.info("Chưa có Run nào được log. Vui lòng huấn luyện mô hình trước.")

    #     except Exception as e:
    #         st.error(f"Không thể kết nối với MLflow: {e}")

if __name__ == "__main__":
    run_ClusteringMinst_app()    




# # # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh4"




