import streamlit as st
import os
import numpy as np
import pandas as pd
import random
import struct
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import mlflow
import time 
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.datasets import make_classification
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

    # Tải dữ liệu MNIST
    try:
        train_images = load_mnist_images(train_images_path)
        train_labels = load_mnist_labels(train_labels_path)
        test_images = load_mnist_images(test_images_path)
        test_labels = load_mnist_labels(test_labels_path)

        st.session_state.train_images = train_images
        st.session_state.train_labels = train_labels
        st.session_state.test_images = test_images
        st.session_state.test_labels = test_labels
    except FileNotFoundError as e:
        st.error(f"⚠️ Lỗi: Không tìm thấy file dữ liệu MNIST. Kiểm tra đường dẫn: {e}")
        return
    except Exception as e:
        st.error(f"⚠️ Lỗi khi tải dữ liệu MNIST: {e}")
        return

    # Chuẩn bị dữ liệu cho giảm chiều (reshape images thành vector)
    X_train = train_images.reshape(train_images.shape[0], -1)  # Chuyển 28x28 thành vector 784
    X_test = test_images.reshape(test_images.shape[0], -1)     # Tương tự cho test
    y_train = train_labels                                    # Nhãn huấn luyện
    y_test = test_labels                                      # Nhãn kiểm tra

    # Lưu vào session_state để sử dụng sau
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    

    # Giao diện Streamlit
    st.title("📸 MNIST PCA_T-SNE")
    tabs = st.tabs([
            "Thông tin",
            "Tập dữ liệu",
            "Chọn mẫu",
            "kỹ thuật thu gọn chiều",
            "Thông tin & Mlflow",
    ])
    tab_note, tab_info,tab_samples, tab_preprocess ,tab_mlflow= tabs
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



    

    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):
            # Chọn mô hình
            model_option1 = st.selectbox("Chọn mô hình", ["PCA (Principal Component Analysis)", "T-SNE (t-Distributed Stochastic Neighbor Embedding)"])
            
            if model_option1 == "PCA (Principal Component Analysis)":
                st.markdown("## 🔹 PCA (Principal Component Analysis)")
                st.markdown("---")
                st.markdown("### Khái niệm PCA (Principal Component Analysis)")
                st.markdown(
                """
                - **PCA (Principal Component Analysis)** là một kỹ thuật giảm chiều dữ liệu tuyến tính, giúp chuyển đổi dữ liệu nhiều chiều
                    thành một tập hợp nhỏ hơn các thành phần chính, giữ lại phần lớn thông tin (phương sai) của dữ liệu gốc. "
                - **PCA** hoạt động bằng cách tìm các hướng có phương sai lớn nhất và chiếu dữ liệu lên các hướng đó."
                """)
                st.markdown("---")
                st.markdown("### Các bước thu gọn chiều với PCA")
                st.write("1. **Chuẩn hóa dữ liệu**: Đưa dữ liệu về dạng chuẩn (trung bình = 0, phương sai = 1).")
                st.write("2. **Tính ma trận hiệp phương sai**: Đánh giá sự tương quan giữa các biến.")
                st.write("3. **Phân tích giá trị riêng và vector riêng**: Tìm các thành phần chính từ ma trận hiệp phương sai.")
                st.write("4. **Chọn thành phần chính**: Sắp xếp theo giá trị riêng giảm dần và giữ lại số lượng mong muốn.")
                st.write("5. **Chiếu dữ liệu**: Chuyển dữ liệu sang không gian mới ít chiều hơn.")

                st.markdown("---")
                X, y = make_classification(
                    n_features=6,
                    n_classes=4,  # Giữ 4 lớp để phù hợp với 2**n_informative
                    n_samples=1500,
                    n_informative=2,
                    random_state=42,
                    n_clusters_per_class=1
                )

                # Áp dụng PCA để giảm chiều xuống 3D (hỗ trợ cả 2D và 3D)
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X)

                # Tiêu đề và thông tin
                st.markdown("### Biểu đồ PCA với Đường Xu hướng")

                # Cho phép người dùng chọn loại biểu đồ
                chart_type = st.selectbox("Chọn loại biểu đồ:", ["2D", "3D"], key="chart_type_pca")

                # Hiển thị thông tin về dữ liệu
                st.markdown("**Thông tin dữ liệu:**")
                st.write(f"- Số mẫu (n_samples): {X.shape[0]}")
                st.write(f"- Số đặc trưng (n_features): {X.shape[1]}")
                st.write(f"- Số lớp (n_classes): {len(np.unique(y))}")

                # Hiển thị tỷ lệ phương sai
                st.markdown("**Tỷ lệ phương sai được giải thích bởi các thành phần chính:**")
                explained_variance_ratio = pca.explained_variance_ratio_
                st.write(f"- Thành phần 1 (PC1): {explained_variance_ratio[0]:.4f}")
                st.write(f"- Thành phần 2 (PC2): {explained_variance_ratio[1]:.4f}")
                st.write(f"- Thành phần 3 (PC3): {explained_variance_ratio[2]:.4f}")
                st.write(f"- Tổng phương sai được giữ lại: {sum(explained_variance_ratio):.4f}")

                # Định nghĩa màu sắc cho từng lớp
                color_map = {
                    0: 'red',    # Lớp 0: Màu đỏ
                    1: 'blue',   # Lớp 1: Màu xanh dương
                    2: 'green',  # Lớp 2: Màu xanh lá
                    3: 'purple'  # Lớp 3: Màu tím
                }

                # Tạo và hiển thị biểu đồ
                if chart_type == "3D":
                    # Biểu đồ 3D
                    fig = go.Figure()

                    # Thêm các điểm dữ liệu cho từng lớp
                    for class_label in np.unique(y):
                        indices = y == class_label
                        fig.add_trace(
                            go.Scatter3d(
                                x=X_pca[indices, 0],
                                y=X_pca[indices, 1],
                                z=X_pca[indices, 2],
                                mode='markers',
                                marker=dict(size=5, opacity=0.6),
                                name=f'Lớp {class_label}',  # Đặt tên cho từng lớp
                                line=dict(color=color_map[class_label])
                            )
                        )

                    # Sắp xếp dữ liệu theo PC1 để tạo đường xu hướng
                    sort_indices = np.argsort(X_pca[:, 0])
                    x_sorted = X_pca[sort_indices, 0]
                    y_sorted = X_pca[sort_indices, 1]

                    # Tạo đường xu hướng (dựa trên PC1 và PC2, z=0 cho đơn giản)
                    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                    y_smooth = UnivariateSpline(x_sorted, y_sorted, k=3, s=0)(x_smooth)

                    # Thêm đường xu hướng vào biểu đồ 3D
                    fig.add_trace(
                        go.Scatter3d(
                            x=x_smooth,
                            y=y_smooth,
                            z=np.zeros_like(x_smooth),
                            mode='lines',
                            line=dict(color='black', width=2),
                            name='Đường xu hướng tổng quát'
                        )
                    )

                    # Cập nhật layout cho biểu đồ 3D
                    fig.update_layout(
                        title="Biểu đồ PCA 3D với Đường Xu hướng",
                        scene=dict(
                            xaxis_title='PC1',
                            yaxis_title='PC2',
                            zaxis_title='PC3'
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        showlegend=True,
                        legend_title_text='Chú thích'
                    )

                    st.markdown("### Biểu đồ 3D:")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    # Biểu đồ 2D
                    fig = go.Figure()

                    # Thêm các điểm dữ liệu cho từng lớp
                    for class_label in np.unique(y):
                        indices = y == class_label
                        fig.add_trace(
                            go.Scatter(
                                x=X_pca[indices, 0],
                                y=X_pca[indices, 1],
                                mode='markers',
                                marker=dict(size=5, opacity=0.6),
                                name=f'Lớp {class_label}',  # Đặt tên cho từng lớp
                                line=dict(color=color_map[class_label])
                            )
                        )

                    # Sắp xếp dữ liệu theo PC1 để tạo đường xu hướng
                    sort_indices = np.argsort(X_pca[:, 0])
                    x_sorted = X_pca[sort_indices, 0]
                    y_sorted = X_pca[sort_indices, 1]

                    # Tạo đường xu hướng
                    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                    y_smooth = UnivariateSpline(x_sorted, y_sorted, k=3, s=0)(x_smooth)

                    # Thêm đường xu hướng vào biểu đồ 2D
                    fig.add_trace(
                        go.Scatter(
                            x=x_smooth,
                            y=y_smooth,
                            mode='lines',
                            line=dict(color='black', width=2),
                            name='Đường xu hướng tổng quát'
                        )
                    )

                    # Cập nhật layout cho biểu đồ 2D
                    fig.update_layout(
                        title="Biểu đồ PCA 2D với Đường Xu hướng",
                        xaxis_title='PC1',
                        yaxis_title='PC2',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        showlegend=True,
                        legend_title_text='Chú thích'
                    )

                    st.markdown("### Biểu đồ 2D:")
                    st.plotly_chart(fig, use_container_width=True)

                # Hiển thị bảng màu sắc của các lớp
                st.markdown("### Màu sắc của từng lớp:")
                color_table = {f"Lớp {k}": v for k, v in color_map.items()}
                st.table(color_table)

                st.markdown("---")
                
                st.markdown("### Công thức toán học")
                st.markdown("""**Công thức giảm chiều PCA (Principal Component Analysis):**""")
                st.latex(r"X_{PCA} = X_{std} V_k")
                st.markdown("""
                **Trong đó:**
                - $$( X_{std} )$$: Ma trận dữ liệu đã chuẩn hóa.  
                - $$( V_k )$$: Ma trận các vector riêng (eigenvectors) tương ứng với $$( k )$$ giá trị riêng lớn nhất của ma trận hiệp phương sai $$( C )$$. Các vector này đại diện cho các thành phần chính.  
                - Kết quả $$( X_{PCA} )$$ là ma trận dữ liệu đã được chiếu lên không gian của các thành phần chính, giúp giảm chiều dữ liệu mà vẫn giữ được phần lớn thông tin. Công thức này cũng đúng, nhưng cần đảm bảo $$( V_k )$$ được tính chính xác từ ma trận hiệp phương sai.
                """)
                st.markdown("""**Tính ma trận hiệp phương sai (Covariance Matrix Formula):**)""")
                st.latex(r"C = \frac{1}{n-1} X_{std}^T X_{std}")
                st.markdown("""  
                **Trong đó:** 
                - $$( X_{std} )$$: Ma trận dữ liệu đã được chuẩn hóa (mỗi cột là một biến đã chuẩn hóa).  
                - $$( X_{std}^T )$$: Ma trận chuyển vị của $$( X_{std} )$$.  
                - $$( n )$$: Số mẫu (số hàng trong ma trận $$( X_{std} ))$$.  
                """)
                st.markdown("---")



                st.markdown("---")
                st.markdown("### Ưu điểm & Nhược điểm của PCA")
                st.table({
                    "**Ưu điểm**": [
                        "Giảm chiều nhanh, hiệu quả với dữ liệu tuyến tính.",
                        "Dễ triển khai, giữ lại phần lớn thông tin quan trọng.",
                        "Giảm thiểu đa cộng tuyến."
                    ],
                    "**Nhược điểm**": [
                        "Không hiệu quả với dữ liệu phi tuyến tính.",
                        "Mất một phần thông tin do nén dữ liệu.",
                        "Khó diễn giải ý nghĩa các thành phần chính."
                    ]
                })
                
            elif model_option1 == "T-SNE (t-Distributed Stochastic Neighbor Embedding)":
                st.markdown("## 🔹 T-SNE (t-Distributed Stochastic Neighbor Embedding)")
                st.markdown("---")
                st.markdown("### Khái niệm T-SNE (t-Distributed Stochastic Neighbor Embedding)")
                st.markdown("""
                - **T-SNE (t-Distributed Stochastic Neighbor Embedding)** là một kỹ thuật giảm chiều phi tuyến tính.
                - chuyên dùng để trực quan hóa dữ liệu nhiều chiều trong không gian 2D hoặc 3D. 
                - Nó giữ lại cấu trúc cục bộ của dữ liệu bằng cách tối ưu hóa sự tương đồng giữa các điểm trong không gian gốc và không gian thấp chiều.
                """)
            
                st.markdown("---")
                st.markdown("### Các bước thu gọn chiều với T-SNE")
                st.write("1. **Tính độ tương đồng trong không gian gốc**: Dùng phân phối Gaussian để đo khoảng cách.")
                st.write("2. **Tính độ tương đồng trong không gian thấp chiều**: Dùng phân phối t-Student để mô phỏng.")
                st.write("3. **Tối ưu hóa hàm mất mát**: Dùng Gradient Descent để giảm KL Divergence.")
                st.write("4. **Trực quan hóa**: Đưa dữ liệu về không gian 2D/3D.")

                X, y = make_classification(
                    n_features=6,
                    n_classes=3,
                    n_samples=1500,
                    n_informative=2,
                    random_state=5,
                    n_clusters_per_class=1
                )

                # Tiêu đề và đường phân cách
                st.markdown("---")
                st.markdown("### Biểu đồ Dữ liệu Phân loại với t-SNE & Plotly")

                # Cho phép người dùng chọn loại biểu đồ (2D hoặc 3D)
                chart_type = st.selectbox("Chọn loại biểu đồ:", ["2D", "3D"])

                # Áp dụng t-SNE để giảm chiều dữ liệu
                n_components = 3 if chart_type == "3D" else 2
                X_embedded = TSNE(n_components=n_components, random_state=42).fit_transform(X)

                # Hiển thị thông tin về dữ liệu
                st.markdown("### Thông tin dữ liệu:")
                st.write(f"- Số mẫu (n_samples): {X.shape[0]}")
                st.write(f"- Số đặc trưng (n_features): {X.shape[1]}")
                st.write(f"- Số lớp (n_classes): {len(np.unique(y))}")

                # Định nghĩa màu sắc cho từng lớp
                color_map = {
                    0: 'red',    # Lớp 0: Màu đỏ
                    1: 'blue',   # Lớp 1: Màu xanh dương
                    2: 'green'   # Lớp 2: Màu xanh lá
                }

                # Tạo và hiển thị biểu đồ dựa trên lựa chọn của người dùng
                if chart_type == "3D":
                    # Biểu đồ 3D
                    fig = go.Figure()

                    # Thêm các điểm dữ liệu cho từng lớp
                    for class_label in np.unique(y):
                        indices = y == class_label
                        fig.add_trace(
                            go.Scatter3d(
                                x=X_embedded[indices, 0],
                                y=X_embedded[indices, 1],
                                z=X_embedded[indices, 2],
                                mode='markers',
                                marker=dict(size=5, opacity=0.8, color=color_map[class_label]),
                                name=f'Lớp {class_label}'  # Đặt tên cho từng lớp
                            )
                        )

                    # Cập nhật layout cho biểu đồ 3D
                    fig.update_layout(
                        title="Biểu đồ Dữ liệu Phân loại 3D (t-SNE)",
                        scene=dict(
                            xaxis_title='t-SNE Dim 1',
                            yaxis_title='t-SNE Dim 2',
                            zaxis_title='t-SNE Dim 3'
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        showlegend=True,
                        legend_title_text='Chú thích'
                    )

                    st.markdown("### Biểu đồ 3D:")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    # Biểu đồ 2D
                    fig = go.Figure()

                    # Thêm các điểm dữ liệu cho từng lớp
                    for class_label in np.unique(y):
                        indices = y == class_label
                        fig.add_trace(
                            go.Scatter(
                                x=X_embedded[indices, 0],
                                y=X_embedded[indices, 1],
                                mode='markers',
                                marker=dict(size=5, opacity=0.8, color=color_map[class_label]),
                                name=f'Lớp {class_label}'  # Đặt tên cho từng lớp
                            )
                        )

                    # Cập nhật layout cho biểu đồ 2D
                    fig.update_layout(
                        title="Biểu đồ Dữ liệu Phân loại 2D (t-SNE)",
                        xaxis_title='t-SNE Dim 1',
                        yaxis_title='t-SNE Dim 2',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font_color='black',
                        showlegend=True,
                        legend_title_text='Chú thích'
                    )

                    st.markdown("### Biểu đồ 2D:")
                    st.plotly_chart(fig, use_container_width=True)

                # Hiển thị bảng màu sắc của các lớp
                st.markdown("### Màu sắc của từng lớp:")
                color_table = {f"Lớp {k}": v for k, v in color_map.items()}
                st.table(color_table)


                st.markdown("---")
                st.markdown("### Công thức toán học")
                st.markdown("**Công thức xác suất có điều kiện trong t-SNE (Conditional Probability Formula for t-SNE):**")
                
                st.latex(r"p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}")
                st.markdown("""
                - **Ý nghĩa**: Tính xác suất có điều kiện $$( p_{j|i} )$$, thể hiện khả năng điểm dữ liệu $$( x_j )$$ là một "hàng xóm" của điểm $$( x_i )$$ trong không gian đầu vào (dữ liệu gốc, thường là không gian cao chiều).  
                - **Trong đó:**
                    - $$( x_i, x_j )$$: Các điểm dữ liệu trong không gian gốc.  
                    - $$( ||x_i - x_j|| )$$: Khoảng cách Euclidean giữa $$( x_i )$$ và $$( x_j )$$.  
                    - $$( \sigma_i )$$: Độ lệch chuẩn cục bộ, được điều chỉnh dựa trên mật độ dữ liệu xung quanh $$( x_i )$$.  
                    - Công thức này sử dụng hàm Gaussian (bình thường) và thuộc về t-SNE, giúp mô hình hóa sự tương đồng giữa các điểm trong không gian dữ liệu gốc.
                """)
                st.markdown("**Công thức xác suất trong không gian nhúng thấp của t-SNE (Low-Dimensional Similarity Probability Formula for t-SNE):**")
                # Công thức 2: Xác suất q_{ij} trong không gian nhúng thấp
                st.latex(r"q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}")
                st.markdown("""
                - **Ý nghĩa**: Tính xác suất $$( q_{ij} )$$, thể hiện sự tương đồng giữa các điểm $$( y_i )$$ và $$( y_j )$$ trong không gian nhúng thấp chiều (thường là 2D hoặc 3D) sau khi giảm chiều.
                - **Trong đó:**  
                    - $$( y_i, y_j )$$: Các điểm trong không gian nhúng thấp.  
                    - $$( ||y_i - y_j|| )$$: Khoảng cách Euclidean giữa $$( y_i )$$ và $$( y_j )$$.  
                    - Công thức này sử dụng phân phối t-Student (với 1 bậc tự do, tương đương phân phối Cauchy) để mô hình hóa sự tương đồng, giúp giảm thiểu sự chồng lấn của các điểm trong không gian thấp chiều. Đây là một phần cốt lõi của t-SNE.
                """)
                st.markdown("**Hàm chi phí Kullback-Leibler divergence trong t-SNE (KL Divergence Cost Function for t-SNE):**")
                # Công thức 3: Hàm chi phí (Cost Function) của t-SNE
                st.latex(r"C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}")
                st.markdown("""
                - **Ý nghĩa**: Đây là hàm chi phí (cost function) hoặc hàm mất mát của t-SNE, đo lường sự khác biệt giữa phân phối xác suất $$( p_{ij} )$$ (từ không gian dữ liệu gốc) và $$( q_{ij} )$$ (từ không gian nhúng thấp).  
                - **Trong đó:**
                    - $$( p_{ij} )$$: Xác suất tương đồng từ không gian dữ liệu gốc .  
                    - $$( q_{ij} )$$: Xác suất tương đồng trong không gian nhúng thấp.  
                    - Hàm chi phí này sử dụng Kullback-Leibler (KL) divergence để tối ưu hóa, nhằm làm cho phân phối trong không gian nhúng thấp khớp với phân phối $$( p_{ij} )$$ trong không gian gốc càng gần càng tốt. Đây là một phần cốt lõi của t-SNE.
                """)

                

                st.markdown("---")
                st.markdown("### Ưu điểm & Nhược điểm của T-SNE")
                st.table({
                    "**Ưu điểm**": [
                        "Giữ lại tốt cấu trúc cục bộ của dữ liệu.",
                        "Hiệu quả cho trực quan hóa dữ liệu phức tạp."
                    ],
                    "**Nhược điểm**": [
                        "Tốn thời gian tính toán, không phù hợp với dữ liệu lớn.",
                        "Không áp dụng được cho dữ liệu mới.",
                        "Phụ thuộc nhiều vào tham số."
                    ]
                })
            
    with tab_samples:
        with st.expander("**Số lượng mẫu để giảm chiều**", expanded=True):
            # Khởi tạo X_train và X_test với giá trị mặc định (None) nếu không có trong session_state
            X_train = None
            X_test = None

            if "X_train" not in st.session_state or "X_test" not in st.session_state:
                st.error("⚠️ Dữ liệu huấn luyện hoặc kiểm tra chưa được tải. Vui lòng kiểm tra lại.")
            else:
                if "X_train_shape" not in st.session_state:
                    st.session_state.X_train_shape = st.session_state.X_train.shape
                if "X_test_shape" not in st.session_state:
                    st.session_state.X_test_shape = st.session_state.X_test.shape

                # Lấy dữ liệu từ session_state
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test

                # Chuẩn hóa dữ liệu
                try:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    st.session_state.X_train_scaled = X_train_scaled
                    st.session_state.X_test_scaled = X_test_scaled
                except Exception as e:
                    st.error(f"⚠️ Lỗi khi chuẩn hóa dữ liệu: {e}")

                # Chọn số lượng mẫu để giảm chiều
                if "X_train_scaled" in st.session_state:
                    n_samples = st.slider("**Số lượng mẫu để giảm chiều:**", 
                                        min_value=100, 
                                        max_value=st.session_state.X_train_scaled.shape[0], 
                                        value=st.session_state.X_train_scaled.shape[0], 
                                        step=100)
                    X_train_subset = st.session_state.X_train_scaled[:n_samples, :]  # Lấy tập con dữ liệu huấn luyện
                    # Lưu n_samples và X_train_subset vào session_state để tab_methods sử dụng
                    st.session_state.n_samples = n_samples
                    st.session_state.X_train_subset = X_train_subset
                    st.success(f"Đã chọn {n_samples} mẫu để giảm chiều. Chuyển sang tab 'Chọn phương pháp thu gọn chiều' để tiếp tục.")
                else:
                    st.warning("Dữ liệu chưa được chuẩn hóa. Vui lòng kiểm tra lại.")


    with tab_preprocess:
        with st.expander("**Chọn phương pháp thu gọn chiều**", expanded=True):
            # Kiểm tra xem dữ liệu đã được chuẩn bị chưa
            if "X_train_subset" not in st.session_state or "n_samples" not in st.session_state:
                st.error("⚠️ Vui lòng chọn số lượng mẫu ở tab 'Số lượng mẫu để giảm chiều' trước khi tiếp tục.")
            else:
                # Lấy dữ liệu từ session_state
                X_train_subset = st.session_state.X_train_subset
                n_samples = st.session_state.n_samples

                # Chọn phương pháp giảm chiều
                dim_reduction_method = st.selectbox("**Chọn phương pháp thu gọn chiều:**", ["PCA", "t-SNE"])

                if dim_reduction_method == "PCA":
                    n_components = st.slider("**Số chiều:**", 
                                            min_value=2, 
                                            max_value=min(X_train_subset.shape[1], 20), 
                                            value=5)
                    chart_type = st.selectbox("**Chọn loại biểu đồ:**", ["2D", "3D"])

                    if st.button("🚀 Chạy PCA"):
                        with st.spinner("Đang huấn luyện mô hình PCA..."):
                            with mlflow.start_run():
                                try:
                                    # Khởi tạo thanh tiến trình
                                    progress_bar = st.progress(0)
                                    progress_text = st.empty()

                                    # Áp dụng PCA
                                    start_time = time.time()
                                    n_components_to_use = n_components if chart_type == "3D" else 2
                                    pca = PCA(n_components=n_components_to_use, random_state=42)
                                    X_train_pca = pca.fit_transform(X_train_subset)
                                    end_time = time.time()

                                    # Giả lập thanh tiến trình dựa trên thời gian thực thi
                                    duration = end_time - start_time
                                    for i in range(100):
                                        time.sleep(duration / 100)
                                        progress_bar.progress(i + 1)
                                        progress_text.text(f"Đang chạy PCA... {i + 1}%")

                                    # Lưu kết quả vào session_state
                                    st.session_state.X_train_pca = X_train_pca
                                    st.session_state.explained_variance_ratio_ = pca.explained_variance_ratio_

                                    # Ghi lại các tham số vào MLflow
                                    mlflow.log_param("algorithm", "PCA")
                                    mlflow.log_param("n_components", n_components_to_use)
                                    mlflow.log_param("n_samples", n_samples)
                                    mlflow.log_param("chart_type", chart_type)
                                    mlflow.log_param("X_train_pca_shape", X_train_pca.shape)
                                    # Chỉ ghi lại 1 hàng đầu tiên thay vì 5 hàng
                                    mlflow.log_param("X_train_pca_sample", X_train_pca[:1].tolist() if X_train_pca.size > 0 else "Empty")

                                    # Ghi lại các chỉ số
                                    explained_variance = np.sum(pca.explained_variance_ratio_)
                                    mlflow.log_metric("explained_variance", round(explained_variance, 4))

                                    # Tính silhouette score để đánh giá chất lượng phân cụm (nếu có nhãn)
                                    if len(np.unique(st.session_state.y_train[:n_samples])) > 1:
                                        silhouette_avg = silhouette_score(X_train_pca, st.session_state.y_train[:n_samples])
                                        mlflow.log_metric("silhouette_score", round(silhouette_avg, 4))
                                    else:
                                        silhouette_avg = "Không tính được (cần ít nhất 2 lớp)"
                                        mlflow.log_metric("silhouette_score", 0.0)

                                    # Tạo và hiển thị biểu đồ
                                    if chart_type == "3D":
                                        fig = px.scatter_3d(
                                            x=X_train_pca[:, 0], 
                                            y=X_train_pca[:, 1], 
                                            z=X_train_pca[:, 2] if n_components_to_use >= 3 else np.zeros_like(X_train_pca[:, 0]),
                                            color=st.session_state.y_train[:n_samples],
                                            opacity=0.6,
                                            title=f"PCA 3D với {n_components_to_use} thành phần chính"
                                        )
                                    else:
                                        fig = px.scatter(
                                            x=X_train_pca[:, 0], 
                                            y=X_train_pca[:, 1],
                                            color=st.session_state.y_train[:n_samples],
                                            opacity=0.6,
                                            title="PCA 2D với 2 thành phần chính"
                                        )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Hiển thị kết quả
                                    st.markdown(
                                        f"""
                                        ### Kết quả PCA:
                                        - Tổng phương sai được giữ lại: {explained_variance:.2f}  
                                        - Silhouette Score: {silhouette_avg}  
                                        - **PCA** giúp giảm chiều dữ liệu trong khi vẫn giữ lại thông tin quan trọng.
                                        """
                                    )
                                    progress_text.text("Hoàn tất PCA! 100%")

                                except Exception as e:
                                    st.error(f"⚠️ Lỗi khi chạy PCA: {e}")
                                    mlflow.log_param("error", str(e))
                                finally:
                                    mlflow.end_run()

                elif dim_reduction_method == "t-SNE":
                    n_components = st.selectbox("**Số chiều đầu ra:**", [2, 3])
                    n_iter = st.slider("**Số vòng lặp tối đa:**", min_value=250, max_value=5000, value=1000, step=250)

                    if st.button("🚀 Chạy t-SNE"):
                        with st.spinner("Đang huấn luyện mô hình t-SNE..."):
                            with mlflow.start_run():
                                try:
                                    # Khởi tạo thanh tiến trình
                                    progress_bar = st.progress(0)
                                    progress_text = st.empty()

                                    # Áp dụng t-SNE
                                    tsne = TSNE(n_components=n_components, n_iter=n_iter, random_state=42, verbose=1)
                                    X_train_tsne = tsne.fit_transform(X_train_subset)

                                    # Cập nhật thanh tiến trình dựa trên số vòng lặp
                                    for i in range(n_iter):
                                        if i % (n_iter // 100) == 0:
                                            progress = int((i / n_iter) * 100)
                                            progress_bar.progress(progress)
                                            progress_text.text(f"Đang chạy t-SNE... {progress}%")

                                    # Lưu kết quả vào session_state
                                    st.session_state.X_train_tsne = X_train_tsne

                                    # Ghi lại các tham số vào MLflow
                                    mlflow.log_param("algorithm", "t-SNE")
                                    mlflow.log_param("n_components", n_components)
                                    mlflow.log_param("n_iter", n_iter)
                                    mlflow.log_param("n_samples", n_samples)
                                    mlflow.log_param("X_train_tsne_shape", X_train_tsne.shape)
                                    # Chỉ ghi lại 1 hàng đầu tiên thay vì 5 hàng
                                    mlflow.log_param("X_train_tsne_sample", X_train_tsne[:1].tolist() if X_train_tsne.size > 0 else "Empty")

                                    # Ghi lại các chỉ số
                                    try:
                                        kl_divergence = tsne.kl_divergence_
                                        st.session_state.kl_divergence = kl_divergence
                                        mlflow.log_metric("kl_divergence", round(kl_divergence, 4))
                                    except AttributeError:
                                        kl_divergence = "Không có thông tin"
                                        st.session_state.kl_divergence = kl_divergence
                                        mlflow.log_metric("kl_divergence", 0.0)
                                        st.warning("⚠️ KL Divergence không khả dụng trong phiên bản t-SNE này.")

                                    # Tính silhouette score để đánh giá chất lượng phân cụm (nếu có nhãn)
                                    if len(np.unique(st.session_state.y_train[:n_samples])) > 1:
                                        silhouette_avg = silhouette_score(X_train_tsne, st.session_state.y_train[:n_samples])
                                        mlflow.log_metric("silhouette_score", round(silhouette_avg, 4))
                                    else:
                                        silhouette_avg = "Không tính được (cần ít nhất 2 lớp)"
                                        mlflow.log_metric("silhouette_score", 0.0)

                                    # Tạo và hiển thị biểu đồ
                                    if n_components == 3:
                                        fig = px.scatter_3d(
                                            x=X_train_tsne[:, 0], 
                                            y=X_train_tsne[:, 1], 
                                            z=X_train_tsne[:, 2],
                                            color=st.session_state.y_train[:n_samples],
                                            opacity=0.6,
                                            title=f"t-SNE 3D với {n_components} chiều"
                                        )
                                    else:
                                        fig = px.scatter(
                                            x=X_train_tsne[:, 0], 
                                            y=X_train_tsne[:, 1],
                                            color=st.session_state.y_train[:n_samples],
                                            opacity=0.6,
                                            title="t-SNE 2D với 2 chiều"
                                        )

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Hiển thị kết quả
                                    st.markdown(
                                        f"""
                                        ### Kết quả t-SNE:
                                        - KL Divergence: {kl_divergence}  
                                        - Silhouette Score: {silhouette_avg}  
                                        - **t-SNE** giúp giữ lại cấu trúc cục bộ của dữ liệu, thích hợp cho dữ liệu phi tuyến tính.
                                        """
                                    )
                                    progress_text.text("Hoàn tất t-SNE! 100%")

                                except Exception as e:
                                    st.error(f"⚠️ Lỗi khi chạy t-SNE: {e}")
                                    mlflow.log_param("error", str(e))
                                finally:
                                    mlflow.end_run()
    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "PCA_TSNE"

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
                    st.experimental_rerun()
            else:
                st.info("Chưa có Run nào để xóa.")

            # 3) Danh sách các thí nghiệm và chi tiết run
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
                # Lấy thuật toán từ tham số
                algorithm = selected_run.data.params.get("algorithm", "Unknown")

                # Tạo dictionary chỉ số dựa trên thuật toán
                if algorithm == "PCA":
                    metrics = {
                        "explained_variance": selected_run.data.metrics.get("explained_variance", "N/A"),
                        "kl_divergence": "N/A",
                        "n_components": selected_run.data.params.get("n_components", "N/A"),
                        "n_iter": "N/A",
                        "n_samples": selected_run.data.params.get("n_samples", "N/A"),
                        "chart_type": selected_run.data.params.get("chart_type", "N/A"),
                        "silhouette_score": selected_run.data.metrics.get("silhouette_score", "N/A")
                    }
                elif algorithm == "t-SNE":
                    metrics = {
                        "explained_variance": "N/A",
                        "kl_divergence": selected_run.data.metrics.get("kl_divergence", "N/A"),
                        "n_components": selected_run.data.params.get("n_components", "N/A"),
                        "n_iter": selected_run.data.params.get("n_iter", "N/A"),
                        "n_samples": selected_run.data.params.get("n_samples", "N/A"),
                        "silhouette_score": selected_run.data.metrics.get("silhouette_score", "N/A")
                    }
                else:
                    metrics = {
                        "explained_variance": "N/A",
                        "kl_divergence": "N/A",
                        "n_components": "N/A",
                        "n_iter": "N/A",
                        "n_samples": "N/A",
                        "chart_type": "N/A",
                        "silhouette_score": "N/A"
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
