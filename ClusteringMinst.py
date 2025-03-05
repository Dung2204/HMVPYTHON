import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import struct
import time
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
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay,adjusted_rand_score
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
    st.title("📸 MNIST Clustering")
    tabs = st.tabs([
            "Thông tin",
            "Tập dữ liệu",
            "Xử lí dữ liệu",
            "Phân cụm dữ liệu",
            "Dự đoán",
            "Thông tin & Mlflow",
    ])
    tab_note,tab_info,tab_load, tab_preprocess, tab_demo ,tab_mlflow= tabs


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
                

                means = [[2, 2], [8, 3], [3, 6]]
                cov = [[1, 0], [0, 1]]
                N = 500

                # Tạo dữ liệu phân cụm
                X0 = np.random.multivariate_normal(means[0], cov, N)
                X1 = np.random.multivariate_normal(means[1], cov, N)
                X2 = np.random.multivariate_normal(means[2], cov, N)

                X = np.concatenate((X0, X1, X2), axis=0)
                K = 3

                # Tạo nhãn ban đầu
                original_label = np.asarray([0]*N + [1]*N + [2]*N)  # Loại bỏ .T vì không cần thiết

                # Hàm hiển thị biểu đồ phân cụm, trả về figure để sử dụng bên ngoài
                def kmeans_display(X, labels):
                    K = np.amax(labels) + 1  # Số cụm (K)
                    
                    # Tạo figure và axes cho matplotlib
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Vẽ từng cụm với màu và kiểu khác nhau
                    colors = ['blue', 'green', 'red']  # Màu cho 3 cụm
                    markers = ['^', 'o', 's']  # Kiểu marker cho 3 cụm
                    
                    for k in range(K):
                        X_k = X[labels == k, :]
                        ax.plot(X_k[:, 0], X_k[:, 1], 
                                marker=markers[k], 
                                color=colors[k], 
                                markersize=4, 
                                alpha=0.8, 
                                label=f'Cụm {k}')
                    
                    # Thiết lập giao diện biểu đồ
                    ax.set_xlabel("Tọa độ X")
                    ax.set_ylabel("Tọa độ Y")
                    ax.set_title("Biểu đồ phân cụm K-Means với dữ liệu cố định")
                    ax.legend()
                    ax.axis('equal')  # Đảm bảo tỷ lệ trục x và y bằng nhau
                    
                    return fig  # Trả về figure để sử dụng bên ngoài

                # Giao diện Streamlit
                st.markdown("---")
                st.markdown("### 📊 **Phân cụm K-Means với dữ liệu cố định**")

                # Hiển thị nút để tạo và hiển thị phân cụm
                if st.button("Hiển thị dữ liệu và phân cụm"):
                    # Hiển thị dữ liệu gốc
                    st.write("**Dữ liệu gốc (3 cụm):**")
                    for i in range(len(X)):
                        point = X[i]
                        cluster = original_label[i]
                        # st.write(f"Điểm {point} thuộc cụm {cluster}")
                    
                    # Hiển thị biểu đồ phân cụm ban đầu
                    fig = kmeans_display(X, original_label)  # Lưu figure từ hàm
                    st.pyplot(fig)  # Hiển thị trên Streamlit
                    
                    # Thực hiện K-Means và hiển thị kết quả
                    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
                    kmeans.fit(X)
                    cluster_labels = kmeans.labels_
                    
                    st.write("**Kết quả phân cụm K-Means:**")
                    for i in range(len(X)):
                        point = X[i]
                        cluster = cluster_labels[i]
                        # st.write(f"Điểm {point} thuộc cụm {cluster}")
                    
                    # Hiển thị biểu đồ phân cụm K-Means
                    fig = kmeans_display(X, cluster_labels)  # Lưu figure từ hàm
                    st.pyplot(fig)  # Hiển thị trên Streamlit
                    
                    

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
                st.markdown("### 🔄 **Quy trình hoạt động của DBSCAN**")
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


                means = [[2, 2], [8, 3], [3, 6]]
                cov = [[1, 0], [0, 1]]
                N = 500

                # Tạo dữ liệu phân cụm
                X0 = np.random.multivariate_normal(means[0], cov, N)
                X1 = np.random.multivariate_normal(means[1], cov, N)
                X2 = np.random.multivariate_normal(means[2], cov, N)

                X = np.concatenate((X0, X1, X2), axis=0)

                # Tạo nhãn ban đầu (dữ liệu gốc)
                original_label = np.asarray([0]*N + [1]*N + [2]*N)

                # Hàm hiển thị biểu đồ phân cụm, trả về figure để sử dụng bên ngoài
                def dbscan_display(X, labels):
                    # Xử lý nhãn - DBSCAN có thể gán nhãn -1 cho các điểm noise
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Loại bỏ noise nếu có
                    
                    # Tạo figure và axes cho matplotlib
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Vẽ từng cụm với màu và kiểu khác nhau
                    colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink']  # Màu cho các cụm
                    markers = ['^', 'o', 's', 'D', '*', 'v']  # Kiểu marker
                    
                    for label in unique_labels:
                        if label == -1:  # Điểm noise
                            color = 'black'
                            marker = 'x'
                            label_text = 'Noise'
                        else:
                            color = colors[label % len(colors)]
                            marker = markers[label % len(markers)]
                            label_text = f'Cụm {label}'
                        
                        X_k = X[labels == label, :]
                        ax.plot(X_k[:, 0], X_k[:, 1], 
                                marker=marker, 
                                color=color, 
                                markersize=4, 
                                alpha=0.8, 
                                label=label_text)
                    
                    # Thiết lập giao diện biểu đồ
                    ax.set_xlabel("Tọa độ X")
                    ax.set_ylabel("Tọa độ Y")
                    ax.set_title("Biểu đồ phân cụm DBSCAN với dữ liệu cố định")
                    ax.legend()
                    ax.axis('equal')  # Đảm bảo tỷ lệ trục x và y bằng nhau
                    
                    # Hiển thị số cụm (nếu có)
                    if n_clusters > 0:
                        st.write(f"Số cụm được phát hiện: {n_clusters}")
                    else:
                        st.write("Không phát hiện cụm nào (có thể tất cả là noise)")
                    
                    return fig  # Trả về figure để sử dụng bên ngoài

                # Giao diện Streamlit
                st.markdown("---")
                st.markdown("### 📊 **Phân cụm DBSCAN với dữ liệu cố định**")

                # Thêm các tham số cho DBSCAN với giá trị mặc định phù hợp hơn
                eps = st.slider("Khoảng cách tối đa (eps):", 0.1, 2.0, 0.5, 0.1)  # Giảm giá trị mặc định từ 0.90 xuống 0.5
                min_samples = st.number_input("Số điểm tối thiểu (min_samples):", min_value=1, max_value=50, value=10)  # Tăng giá trị mặc định từ 5 lên 10

                # Hiển thị nút để tạo và hiển thị phân cụm
                if st.button("Hiển thị dữ liệu và phân cụm"):
                    # Hiển thị dữ liệu gốc
                    st.write("### Dữ liệu gốc (3 cụm):")
                    for i in range(len(X)):
                        point = X[i]
                        cluster = original_label[i]
                        # st.write(f"Điểm {point} thuộc cụm {cluster}")
                    
                    # Hiển thị biểu đồ phân cụm ban đầu
                    fig = dbscan_display(X, original_label)  # Hiển thị dữ liệu gốc
                    st.pyplot(fig)
                    
                    # Thực hiện DBSCAN và hiển thị kết quả
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                    dbscan_labels = dbscan.labels_
                    
                    st.write("### Kết quả phân cụm DBSCAN:")
                    for i in range(len(X)):
                        point = X[i]
                        cluster = dbscan_labels[i]
                        if cluster == -1:
                            cluster_text = "Noise"
                        else:
                            cluster_text = f"Cụm {cluster}"
                        # st.write(f"Điểm {point} thuộc {cluster_text}")
                    
                    # Hiển thị biểu đồ phân cụm DBSCAN
                    fig = dbscan_display(X, dbscan_labels)  # Hiển thị kết quả DBSCAN
                    st.pyplot(fig)




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

 

    with tab_load:
        with st.expander("**Xử lý dữ liệu**", expanded=True):
            # Kiểm tra dữ liệu trong session_state
            if "train_images" in st.session_state and "train_labels" in st.session_state:
                # Chuyển đổi dữ liệu thành vector 1 chiều
                try:
                    X = st.session_state.train_images.reshape(st.session_state.train_images.shape[0], -1)
                    y = st.session_state.train_labels  # Nhãn (0-9)

                    # Kiểm tra kích thước của X và y
                    if len(X) != len(y):
                        st.error("🚨 Kích thước của dữ liệu và nhãn không khớp. Vui lòng kiểm tra dữ liệu đầu vào.")
                        st.stop()

                    # Chọn tỷ lệ tập Test (%)
                    test_size = st.slider("🔹 **Chọn tỷ lệ tập Test (%)**", min_value=10, max_value=90, value=20, step=5) / 100

                    # Tính tỷ lệ tập Train tự động
                    train_size = 1.0 - test_size  # Tự động tính tỷ lệ train

                    # Chia tập dữ liệu thành Train và Test
                    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y  # Thêm stratify để giữ tỷ lệ nhãn
                    )

                    # Lưu vào session_state
                    st.session_state.X_train = X_train_final
                    st.session_state.X_test = X_test_final
                    st.session_state.y_train = y_train_final
                    st.session_state.y_test = y_test_final

                    # Tính tổng số mẫu
                    total_samples = len(X)
                    train_samples = len(X_train_final)
                    test_samples = len(X_test_final)

                    # Tính tỷ lệ thực tế (%)
                    train_ratio = (train_samples / total_samples) * 100
                    test_ratio = (test_samples / total_samples) * 100

                    # Hiển thị thông báo và tỷ lệ phân chia
                    st.write(f"📊 **Tỷ lệ phân chia**: Test={test_ratio:.0f}% , Train={train_ratio:.0f}%")
                    st.write("✅ Dữ liệu đã được xử lý và chia tách.")
                    st.write(f"🔹 **Kích thước tập Train**: `{X_train_final.shape}`")
                    st.write(f"🔹 **Kích thước tập Test**: `{X_test_final.shape}`")
                except Exception as e:
                    st.error(f"🚨 Lỗi khi xử lý dữ liệu: {str(e)}")
            else:
                st.error("🚨 Dữ liệu chưa được tải! Vui lòng tải dữ liệu trước khi xử lý.")
            


    with tab_preprocess:
        st.write("***Phân cụm dữ liệu***")
        if "X_train" in st.session_state and "X_test" in st.session_state:
            # Lấy dữ liệu từ session_state (chỉ dùng X_train cho phân cụm)
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test

            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Giảm chiều bằng PCA (2D) để phân cụm
            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train_scaled)

            st.session_state.scaler = scaler
            st.session_state.pca = pca
            st.session_state.X_train_pca = X_train_pca

            # Chọn phương pháp phân cụm
            clustering_method = st.selectbox("🔹 Chọn phương pháp phân cụm:", ["K-means", "DBSCAN"])

            if clustering_method == "K-means":
                k = st.slider("🔸 Số cụm (K-means)", min_value=2, max_value=20, value=10)

                if st.button("🚀 Chạy K-means"):
                    with st.spinner("Đang huấn luyện mô hình..."):
                        with mlflow.start_run():
                            # Đo thời gian bắt đầu
                            start_time = time.time()

                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            labels = kmeans.fit_predict(X_train_pca)

                            # Đo thời gian kết thúc và tính thời gian phân cụm
                            end_time = time.time()
                            clustering_time = round(end_time - start_time, 2)  # Làm tròn 2 chữ số thập phân

                            mlflow.log_param("algorithm", "K-means")
                            mlflow.log_param("k", k)
                            mlflow.log_param("max_iter", 300)

                            # Tính Inertia
                            inertia = kmeans.inertia_
                            max_possible_inertia = np.sum(np.sum((X_train_pca - np.mean(X_train_pca, axis=0)) ** 2))
                            if max_possible_inertia > 0:
                                accuracy_percentage = (1 - (inertia / max_possible_inertia)) * 100
                            else:
                                accuracy_percentage = 0.0
                            accuracy_percentage = max(0.0, min(100.0, accuracy_percentage))
                            accuracy_percentage = round(accuracy_percentage, 2)

                            # Tính các thông tin bổ sung
                            num_samples = X_train_pca.shape[0]  # Số mẫu đã xử lý
                            num_clusters_actual = len(set(labels))  # Số cụm thực tế (trong trường hợp hiếm, có thể ít hơn k)

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

                            # Log mô hình K-means
                            mlflow.sklearn.log_model(kmeans, "kmeans_model")
                            st.session_state.clustering_model = kmeans
                            st.session_state.clustering_method = "K-means"
                            st.session_state.labels = labels

                        mlflow.end_run()

            elif clustering_method == "DBSCAN":
                eps = st.slider("🔸 Bán kính vùng lân cận (eps):", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider("🔸 Số lượng điểm tối thiểu:", min_value=1, max_value=20, value=10)

                if st.button("🚀 Chạy DBSCAN"):
                    with st.spinner("Đang huấn luyện mô hình DBSCAN..."):
                        with mlflow.start_run():
                            # Đo thời gian bắt đầu
                            start_time = time.time()

                            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                            labels = dbscan.fit_predict(X_train_pca)

                            # Đo thời gian kết thúc và tính thời gian phân cụm
                            end_time = time.time()
                            clustering_time = round(end_time - start_time, 2)

                            mlflow.log_param("algorithm", "DBSCAN")
                            mlflow.log_param("eps", eps)
                            mlflow.log_param("min_samples", min_samples)

                            # Tính số lượng cụm (không tính noise, nhãn -1)
                            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                            # Tính số lượng điểm nhiễu
                            noise_points = np.sum(labels == -1)
                            # Tính số mẫu đã xử lý
                            num_samples = X_train_pca.shape[0]

                            # Hiển thị kết quả
                            with st.container(border=True):
                                st.write("### Kết quả phân cụm và thông tin của DBSCAN:")
                                st.write(f"**Phương pháp phân cụm đã chọn:** DBSCAN")
                                st.write(f"**Số cụm đã chọn:** Không áp dụng (tự động xác định)")
                                st.write(f"**Số cụm thực tế:** {num_clusters}")
                                st.write(f"**Số mẫu đã xử lý:** {num_samples}")
                                st.write(f"**Thời gian phân cụm:** {clustering_time} giây")
                                st.write(f"**Số lượng điểm nhiễu (Noise Points):** {noise_points} ({round((noise_points / num_samples) * 100, 2)}%)")

                            # Log các chỉ số vào MLflow
                            mlflow.log_metric("num_clusters", num_clusters)
                            mlflow.log_metric("noise_points", noise_points)
                            for cluster, count in Counter(labels).items():
                                if cluster != -1:  # Bỏ qua nhãn -1 (noise)
                                    mlflow.log_metric(f"cluster_{cluster}_size", count)

                            # Log mô hình DBSCAN
                            mlflow.sklearn.log_model(dbscan, "dbscan_model")
                            st.session_state.clustering_model = dbscan
                            st.session_state.clustering_method = "DBSCAN"
                            st.session_state.labels = labels

                        mlflow.end_run()
        else:
            st.error("🚨 Dữ liệu chưa được xử lý! Hãy đảm bảo bạn đã chạy phần tiền xử lý dữ liệu trước khi thực hiện phân cụm.")

    with tab_demo:
        with st.expander("**Dự đoán cụm cho ảnh**", expanded=False):
            st.write("Tải lên ảnh chữ số viết tay (28x28 pixel, grayscale) để dự đoán cụm:")

            # Kiểm tra và hiển thị thông tin phân cụm đã huấn luyện
            if "clustering_method" in st.session_state and "clustering_model" in st.session_state:
                clustering_method = st.session_state.clustering_method
                model = st.session_state.clustering_model
                num_samples = st.session_state.X_train_pca.shape[0] if "X_train_pca" in st.session_state else 0
                clustering_time = st.session_state.get("clustering_time", "Không có dữ liệu")

                with st.container(border=True):
                    st.write("### Thông tin phân cụm đã huấn luyện:")
                    st.write(f"**Phương pháp phân cụm:** {clustering_method}")
                    if clustering_method == "K-means":
                        k = model.n_clusters  # Lấy số cụm từ mô hình K-means
                        accuracy_percentage = st.session_state.get("accuracy_percentage", "Không có dữ liệu")
                        num_clusters_actual = len(set(st.session_state.labels)) if "labels" in st.session_state else k
                        st.write(f"**Số cụm đã chọn:** {k}")
                        st.write(f"**Số cụm thực tế:** {num_clusters_actual}")
                        st.write(f"**Số mẫu đã xử lý:** {num_samples}")
                        # st.write(f"**Thời gian phân cụm:** {clustering_time} giây")
                        # st.write(f"**Độ chính xác của phân cụm K-Means:** {accuracy_percentage}")
                    elif clustering_method == "DBSCAN":
                        labels = st.session_state.labels if "labels" in st.session_state else []
                        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        noise_points = np.sum(labels == -1) if labels.size > 0 else 0
                        st.write(f"**Số cụm thực tế:** {num_clusters}")
                        st.write(f"**Số mẫu đã xử lý:** {num_samples}")
                        st.write(f"**Số lượng điểm nhiễu (Noise Points):** {noise_points} ({round((noise_points / num_samples) * 100, 2)}%) ")

                # Tải file ảnh
                uploaded_file = st.file_uploader("Chọn ảnh (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"], key="upload_predict")

                # Dự đoán nếu có ảnh được tải lên
                if uploaded_file is not None:
                    # Đọc và xử lý ảnh
                    image = Image.open(uploaded_file).convert('L')  # Chuyển thành grayscale
                    image = image.resize((28, 28))  # Đảm bảo kích thước 28x28
                    image_array = np.array(image) / 255.0  # Chuẩn hóa về [0, 1]
                    image_vector = image_array.reshape(1, -1)  # Chuyển thành vector 1 chiều (784 chiều)

                    # Chuẩn hóa ảnh mới bằng scaler đã fit trên X_train
                    if "scaler" in st.session_state:
                        image_scaled = st.session_state.scaler.transform(image_vector)
                    else:
                        st.error("🚨 Scaler chưa được huấn luyện. Vui lòng chạy phân cụm trong tab 'Phân cụm dữ liệu' trước.")
                        st.stop()

                    # Giảm chiều ảnh mới bằng PCA đã fit trên X_train
                    if "pca" in st.session_state:
                        image_pca = st.session_state.pca.transform(image_scaled)
                    else:
                        st.error("🚨 PCA chưa được huấn luyện. Vui lòng chạy phân cụm trong tab 'Phân cụm dữ liệu' trước.")
                        st.stop()

                    # Kiểm tra xem mô hình phân cụm đã được huấn luyện chưa
                    if "clustering_model" not in st.session_state or "clustering_method" not in st.session_state:
                        st.error("🚨 Mô hình phân cụm chưa được huấn luyện. Vui lòng chạy phân cụm trong tab 'Phân cụm dữ liệu' trước.")
                    else:
                        clustering_method = st.session_state.clustering_method
                        model = st.session_state.clustering_model

                        if clustering_method == "K-means":
                            # Dự đoán cụm cho ảnh mới dựa trên khoảng cách đến tâm cụm
                            try:
                                cluster_label = model.predict(image_pca)[0]
                                with st.container(border=True):
                                    st.write("### Kết quả dự đoán cho ảnh mới:")
                                    st.write(f"Ảnh của bạn thuộc **Cụm {cluster_label}**")
                                    st.image(image, caption="Ảnh được tải lên", width=100)
                            except Exception as e:
                                st.error(f"🚨 Lỗi khi dự đoán với K-means: {str(e)}")

                        elif clustering_method == "DBSCAN":
                            # Với DBSCAN, dự đoán dựa trên khoảng cách đến các điểm đã phân cụm
                            if "X_train_pca" in st.session_state and "labels" in st.session_state:
                                X_train_pca = st.session_state.X_train_pca
                                labels = st.session_state.labels

                                distances = np.linalg.norm(X_train_pca - image_pca, axis=1)
                                nearest_point_idx = np.argmin(distances)
                                nearest_label = labels[nearest_point_idx]

                                with st.container(border=True):
                                    st.write("### Kết quả dự đoán cho ảnh mới:")
                                    if nearest_label == -1:
                                        st.write("Ảnh của bạn được coi là **điểm nhiễu (Noise)**")
                                    else:
                                        st.write(f"Ảnh của bạn thuộc **Cụm {nearest_label}**")
                                    st.image(image, caption="Ảnh được tải lên", width=100)
                            else:
                                st.error("🚨 Dữ liệu huấn luyện PCA hoặc nhãn chưa được lưu. Vui lòng chạy phân cụm trong tab 'Phân cụm dữ liệu' trước.")
                                st.stop()



    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "Clustering"

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
                    "Algorithm": selected_run.data.params.get("algorithm", "N/A"),
                    "K (for K-means)": selected_run.data.params.get("k", "N/A"),
                    "Max Iter (for K-means)": selected_run.data.params.get("max_iter", "N/A"),
                    "EPS (for DBSCAN)": selected_run.data.params.get("eps", "N/A"),
                    "Min Samples (for DBSCAN)": selected_run.data.params.get("min_samples", "N/A"),
                    "Inertia (for K-means)": selected_run.data.metrics.get("inertia", "N/A"),
                    "Num Clusters (for DBSCAN)": selected_run.data.metrics.get("num_clusters", "N/A"),
                    "Noise Points (for DBSCAN)": selected_run.data.metrics.get("noise_points", "N/A")
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




