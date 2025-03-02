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
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from collections import Counter
from mlflow.tracking import MlflowClient

def run_PcaTSNEMinst_app():
    @st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
    def get_sampled_pixels(images, sample_size=100_000):
        return np.random.choice(images.flatten(), sample_size, replace=False)

    @st.cache_data  # Cache danh sách ảnh ngẫu nhiên
    def get_random_indices(num_images, total_images):
        return np.random.randint(0, total_images, size=num_images)

    # Cấu hình Streamlit
    #   st.set_page_config(page_title="Phân loại ảnh", layout="wide")
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
    st.title("📸 MINST PCA_T-SNE")
    tabs = st.tabs([
            "Tập dữ liệu",
            "Xử lí dữ liệu",
            "Thông tin",
            "kỹ thuật thu gọn chiều",
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

            # st.write("**Bảng dữ liệu sau khi chuẩn hóa**")
            # st.dataframe(df_normalized)

            
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
            # Chọn mô hình*
            model_option1 = st.selectbox("Chọn mô hình", ["PCA (Principal Component Analysis)", "T-SNE (t-Distributed Stochastic Neighbor Embedding)"])
            
            if model_option1 == "PCA (Principal Component Analysis)":
                st.markdown("## 🔹 PCA (Principal Component Analysis)")
                st.markdown("---")
                st.markdown("### PCA - Phân tích thành phần chính")
                st.write(
                    "**PCA (Principal Component Analysis)** là một kỹ thuật giảm chiều dữ liệu tuyến tính, giúp chuyển đổi dữ liệu có nhiều chiều "
                    "thành một tập hợp nhỏ hơn các thành phần chính có thể giữ lại nhiều nhất thông tin gốc. "
                    "PCA hoạt động bằng cách tìm các hướng có phương sai lớn nhất của dữ liệu, sau đó chiếu dữ liệu lên các hướng đó."
                )
                st.markdown("### Tham số quan trọng của PCA")
                st.write("**Tham số `n_components`:**")
                st.write("- Xác định số lượng thành phần chính cần giữ lại.")
                st.write("- Nếu `n_components=k`, PCA giữ lại **k thành phần chính**.")
                st.write("- Nếu `n_components=None`, giữ lại toàn bộ dữ liệu.")
                st.write("- Có thể chọn `n_components=0.95` để giữ lại 95% phương sai.")
                st.markdown("**Tham số `svd_solver`:**")
                st.write("- Xác định thuật toán SVD để tính PCA.")
                st.write("- Các giá trị: `'auto'`, `'full'`, `'arpack'`, `'randomized'`.")
                st.write("- Thường dùng `'randomized'` khi dữ liệu lớn để tăng tốc.")
                st.markdown("**Tham số `whiten`:**")
                st.write("- Nếu `whiten=True`, PCA chuẩn hóa dữ liệu để phương sai mỗi thành phần chính = 1.")
                st.write("- Hữu ích khi cần dữ liệu có dạng chuẩn tắc hơn.")

                st.markdown("### Các bước thu gọn chiều với PCA")
                st.write("1. **Chuẩn hóa dữ liệu**: Đưa dữ liệu về cùng một thang đo (`mean` = 0, `variance` = 1).")
                st.write("2. **Tính ma trận hiệp phương sai**: Đánh giá sự tương quan giữa các biến.")
                st.write("3. **Tính giá trị riêng và vector riêng**: Tìm các thành phần chính dựa trên các vector riêng.")
                st.write("4. **Chọn số lượng thành phần chính**: Giữ lại các thành phần chính có giá trị riêng lớn nhất.")
                st.write("5. **Chiếu dữ liệu vào không gian mới**: Biểu diễn dữ liệu trong hệ trục mới có ít chiều hơn.")
                
                st.markdown("### Ưu điểm & Nhược điểm của PCA")
                st.table({
                    "**Ưu điểm**": [
                        "Giảm chiều nhanh, hiệu quả với dữ liệu tuyến tính.",
                        "Dễ triển khai, giữ lại thông tin quan trọng."
                    ],
                    "**Nhược điểm**": [
                        "Không hoạt động tốt với dữ liệu phi tuyến tính.",
                        "Mất một phần thông tin do nén dữ liệu."
                    ]
                })
                
            elif model_option1 == "T-SNE (t-Distributed Stochastic Neighbor Embedding)":
                st.markdown("## 🔹 T-SNE (t-Distributed Stochastic Neighbor Embedding) ")
                st.markdown("---")
                st.markdown("### T-SNE- Phân tích thành phần chính")
                st.write(
                    "**T-SNE (t-Distributed Stochastic Neighbor Embedding)** là một phương pháp giảm chiều dữ liệu phi tuyến tính, chuyên dùng để trực quan hóa dữ liệu có chiều cao. "
                    "Nó hoạt động bằng cách giữ lại mối quan hệ cục bộ giữa các điểm dữ liệu và ánh xạ chúng vào không gian có chiều thấp hơn.")
                
                st.markdown("### Tham số quan trọng của T-SNE")

                st.markdown("**Tham số `n_components`:**")
                st.write("- Xác định số chiều đầu ra (thường là **2 hoặc 3** để trực quan hóa).")

                st.markdown("**Tham số `perplexity`:**")
                st.write("- Điều chỉnh số lượng hàng xóm quan trọng của mỗi điểm.")
                st.write("- Giá trị hợp lý: **5 đến 50** (cần thử nghiệm để tối ưu).")

                st.markdown("**Tham số `learning_rate`:**")
                st.write("- Kiểm soát tốc độ cập nhật vị trí điểm dữ liệu.")
                st.write("- Giá trị thường dùng: **10 - 1000** (mặc định là `200`).")

                st.markdown("**Tham số `n_iter`:**")
                st.write("- Số vòng lặp tối ưu hóa thuật toán.")
                st.write("- Giá trị thường dùng: **1000 - 5000**.")

                st.markdown("**Tham số `metric`:**")
                st.write("- Xác định khoảng cách để tính độ tương đồng giữa điểm dữ liệu.")
                st.write("- Mặc định là `'euclidean'`, có thể dùng `'cosine'`, `'manhattan'`, v.v.")

                st.markdown("###  Các bước thu gọn chiều với T-SNE")
                st.write("1. **Tính toán phân bố khoảng cách**: Sử dụng phân phối Gaussian ở không gian cao chiều.")
                st.write("2. **Tính toán phân bố trong không gian thấp**: Sử dụng phân phối t-Student để duy trì quan hệ tương đồng.")
                st.write("3. **Giảm thiểu hàm mất mát**: Điều chỉnh vị trí điểm dữ liệu để tối ưu sự tương đồng.")
                st.write("4. **Trực quan hóa dữ liệu**: Hiển thị dữ liệu trong không gian 2D hoặc 3D.")
                
                st.markdown("###  Ưu điểm & Nhược điểm của T-SNE")
                st.table({
                    "**Ưu điểm**": [
                        "Giữ lại tốt mối quan hệ cục bộ giữa các điểm.",
                        "Hiệu quả khi trực quan hóa dữ liệu nhiều chiều."
                    ],
                    "**Nhược điểm**": [
                        "Chậm hơn PCA, không phù hợp cho dữ liệu lớn.",
                        "Không thể dùng để biến đổi dữ liệu mới."
                    ]
                })



    with tab_preprocess:
        with st.expander("**kỹ thuật thu gọn chiều**", expanded=True):    

            if "X_train" in st.session_state and "X_val" in st.session_state and "X_test" in st.session_state:
                # Lấy dữ liệu từ session_state
                X_train = st.session_state.X_train
                X_val = st.session_state.X_val
                X_test = st.session_state.X_test
                # Chuẩn hóa dữ liệu
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)

                st.session_state.X_train_scaled = X_train_scaled
                st.session_state.X_val_scaled = X_val_scaled
                st.session_state.X_test_scaled = X_test_scaled

                    
                    # Chọn phương pháp phân cụm
                dim_reduction_method = st.selectbox("**Chọn phương pháp thu gọn chiều:**", ["PCA", "t-SNE"])
                if dim_reduction_method == "PCA":
                    # Tham số của PCA
                    n_components = st.slider("**Số thành phần chính (n_components):**", min_value=2, max_value=min(X_train.shape[1], 20), value=5)
                    svd_solver = st.selectbox("**Thuật toán SVD:**", ["auto", "full", "arpack", "randomized"])
                    whiten = st.checkbox("**Chuẩn hóa dữ liệu (whiten):**", value=False)

                    if st.button("🚀 Chạy PCA"):
                        with mlflow.start_run():
                            # Áp dụng PCA
                            pca = PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten, random_state=42)
                            X_train_pca = pca.fit_transform(X_train_scaled)

                            # Log tham số vào MLflow
                            mlflow.log_param("algorithm", "PCA")
                            mlflow.log_param("n_components", n_components)
                            mlflow.log_param("svd_solver", svd_solver)
                            mlflow.log_param("whiten", whiten)
                            st.session_state.X_train_pca = X_train_pca
                            st.session_state.explained_variance_ratio_ = pca.explained_variance_ratio_
                            mlflow.log_param("X_train_pca",X_train_pca)
                            # Log phương sai giải thích
                            explained_variance = np.sum(pca.explained_variance_ratio_)
                            mlflow.log_metric("explained_variance", explained_variance)

                            # Vẽ biểu đồ
                            fig, ax = plt.subplots(figsize=(6, 4))
                            scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.5, cmap="coolwarm")
                            ax.set_title(f"PCA với {n_components} thành phần chính")
                            st.pyplot(fig)

                            fig.savefig("pca_result.png")
                            mlflow.log_artifact("pca_result.png")

                            st.markdown(
                                f"""
                                ### Kết quả PCA:
                                - Tổng phương sai được giữ lại: {explained_variance:.2f}  
                                - **PCA** giúp giảm chiều dữ liệu trong khi vẫn giữ lại thông tin quan trọng. 
                                """
                            )
                        mlflow.end_run()

                elif dim_reduction_method == "t-SNE":
                    # Tham số của t-SNE
                    n_components = st.selectbox("**Số chiều đầu ra:**", [2, 3])
                    if st.toggle("Hiển thị thông tin số chiều đầu ra"):
                        st.write("**Nếu chọn 2**: 2D → Dễ vẽ biểu đồ trên mặt phẳng. Chuẩn hóa dữ liệu về khoảng [0,1] hoặc [-1,1], giúp duy trì tỷ lệ giữa các giá trị gốc.")
                        st.write("**Nếu chọn 3**: 3D → Hiển thị tốt hơn với dữ liệu phức tạp. Biến đổi dữ liệu về trung bình 0 và độ lệch chuẩn 1, phù hợp với dữ liệu có phân phối chuẩn.")   
                    perplexity = st.slider("**Perplexity:**", min_value=5, max_value=50, value=30)
                    learning_rate = st.slider("**Learning rate:**", min_value=10, max_value=1000, value=200)
                    n_iter = st.slider("**Số vòng lặp tối đa:**", min_value=250, max_value=5000, value=1000, step=250)
                    metric = st.selectbox("**Khoảng cách:**", ["euclidean", "cosine", "manhattan"])

                    if st.button("🚀 Chạy t-SNE"):
                        with mlflow.start_run():
                            # Áp dụng t-SNE
                            tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, 
                                        n_iter=n_iter, metric=metric, random_state=42)
                            X_train_tsne = tsne.fit_transform(X_train_scaled)
                            st.session_state.X_train_tsne = X_train_tsne
                            try:
                                st.session_state.kl_divergence = tsne.kl_divergence_
                            except AttributeError:
                                st.session_state.kl_divergence = "Không có thông tin"
                            mlflow.log_param("algorithm", "t-SNE")
                            mlflow.log_param("n_components", n_components)
                            mlflow.log_param("perplexity", perplexity)
                            mlflow.log_param("learning_rate", learning_rate)
                            mlflow.log_param("n_iter", n_iter)
                            mlflow.log_param("metric", metric)
                            
                            mlflow.log_param("X_train_tsne",X_train_tsne)
                            
                            fig, ax = plt.subplots(figsize=(6, 4))
                            scatter = ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], alpha=0.5, cmap="coolwarm")
                            ax.set_title(f"t-SNE với Perplexity={perplexity}")
                            st.pyplot(fig)

                            fig.savefig("tsne_result.png")
                            mlflow.log_artifact("tsne_result.png")

                            st.markdown(
                                f"""
                                ### Kết quả t-SNE:
                                - Dữ liệu đã được giảm chiều xuống {n_components} chiều để trực quan hóa.  
                                - **t-SNE** giúp giữ lại cấu trúc cục bộ của dữ liệu, thích hợp cho dữ liệu phi tuyến tính.
                                """
                            )
                        mlflow.end_run()


                            

    with tab_split:
        with st.expander(" Đánh giá hiệu suất mô hình phân cụm", expanded=True):
            if "X_train_pca" in st.session_state and "explained_variance_ratio_" in st.session_state:
                X_reduced = st.session_state.X_train_pca
                explained_var = st.session_state.explained_variance_ratio_

                st.markdown("### PCA (Principal Component Analysis)")
                st.markdown("---")
                st.write(f"✅ **Explained Variance Ratio:** {explained_var}")
                total_explained = np.sum(explained_var) * 100

                # Tạo chuỗi giải thích động dựa trên giá trị thực tế
                explanation = "**Giải thích:**\n"
                for i, var in enumerate(explained_var):
                    explanation += f"- Thành phần chính thứ {i+1} giải thích **{var*100:.2f}%** phương sai.\n"
                st.markdown(explanation)
                st.write(f"✅ **Tổng phương sai giải thích:** {np.sum(explained_var):.4f}")

                # Vẽ biểu đồ trực quan hóa dữ liệu PCA
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, marker='o', c='blue')
                ax.set_title("Phân bố dữ liệu sau PCA")
                ax.set_xlabel("Thành phần 1")
                ax.set_ylabel("Thành phần 2")
                st.pyplot(fig)

            # Kiểm tra xem dữ liệu t-SNE có tồn tại không
            elif "X_train_tsne" in st.session_state:
                X_reduced = st.session_state.X_train_tsne
                kl_divergence = st.session_state.get("kl_divergence", "Không có thông tin")

                st.markdown("### t-SNE (t-Distributed Stochastic Neighbor Embedding)")
                st.markdown("---")
                st.write(f"✅ **KL Divergence:** {kl_divergence}")
                st.markdown("""
                - KL Divergence thấp cho thấy t-SNE hội tụ tốt.
                - KL Divergence cao có thể do perplexity không phù hợp hoặc dữ liệu phức tạp.
                """)

                # Vẽ biểu đồ t-SNE
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, marker='o', c='green')
                ax.set_title("Phân bố dữ liệu sau t-SNE")
                ax.set_xlabel("Thành phần 1")
                ax.set_ylabel("Thành phần 2")
                st.pyplot(fig)
            else:
                st.warning("⚠️ Chưa có dữ liệu giảm chiều để đánh giá!")

            # So sánh cấu trúc dữ liệu trước và sau giảm chiều
            if "X_train_scaled" in st.session_state and ("X_train_pca" in st.session_state or "X_train_tsne" in st.session_state):
                X_original = st.session_state.X_train_scaled  # Dữ liệu gốc
                original_distances = pairwise_distances(X_original[:500])
                reduced_distances = pairwise_distances(X_reduced[:500])
                correlation = np.corrcoef(original_distances.flatten(), reduced_distances.flatten())[0, 1]
                st.write(f"✅ **Tương quan khoảng cách trước và sau giảm chiều:** {correlation:.4f}")

            else:
                st.warning("⚠️ Chưa có dữ liệu giảm chiều để đánh giá!")
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
                    "n_components": selected_run.data.metrics.get("n_components", "N/A"),
                    "perplexity": selected_run.data.metrics.get("perplexity", "N/A"),
                    "learning_rate": selected_run.data.metrics.get("learning_rate", "N/A"),
                    "n_iter": selected_run.data.metrics.get("n_iter", "N/A"),
                    "metric": selected_run.data.metrics.get("metric", "N/A"),
                    "svd_solver": selected_run.data.metrics.get("svd_solver", "N/A"),
                    "whiten": selected_run.data.metrics.get("whiten", "N/A")
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
    run_PcaTSNEMinst_app()  


# st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
# with st.expander("🖼️ Đánh giá hiệu suất mô hình phân cụm", expanded=True):
#     # st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
#     print("🎯 Kiểm tra trên DagsHub: https://dagshub.com/Dung2204/Minst-mlflow.mlflow")


# # # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App"
