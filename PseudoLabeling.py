import streamlit as st
import os
import numpy as np
import pandas as pd
import random
import struct
import cv2
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import mlflow
import time 
from PIL import Image
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
import networkx as nx
import plotly.express as px
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks  # Thêm import callbacks
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import make_classification
from mlflow.tracking import MlflowClient

def run_PseudoLabeling_app():
    @st.cache_data  # Lưu cache để tránh load lại dữ liệu mỗi lần chạy lại Streamlit
    def get_sampled_pixels(images, sample_size=100_000):
        return np.random.choice(images.flatten(), sample_size, replace=False)

    @st.cache_data  # Cache danh sách ảnh ngẫu nhiên
    def get_random_indices(num_images, total_images):
        return np.random.randint(0, total_images, size=num_images)

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
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    dataset_path = os.path.dirname(os.path.abspath(__file__))
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

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
    st.title("📸 PseudoLabeling Neural NetWork")
    tabs = st.tabs([
            "Thông tin",
            "Tập dữ liệu",
            " Phân chia tỉ lệ",
            "Huấn luyện Neural Network",
            "Huấn luyện Pseudo Labeling",
            "Dự đoán",
            "Thông tin & Mlflow",
    ])
    tab_note, tab_info, tab_samples, tab_preprocess,tab_pseudo_labeling,tab_demo,tab_mlflow = tabs

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

    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):
            model_option = st.selectbox("Chọn mô hình", ["Neural Network (Mạng Nơ-ron Nhân tạo)", "Pseudo Labelling (Gắn nhãn Giả)"])

            if model_option == "Neural Network (Mạng Nơ-ron Nhân tạo)":
                st.markdown("## 🔹 Neural Network (Mạng Nơ-ron Nhân tạo)")
                st.markdown("---")
                st.markdown("### Khái niệm Neural Network")
                st.markdown(
                        """
                        - **Neural Network (Mạng Nơ-ron Nhân tạo)** là một mô hình học máy được lấy cảm hứng từ cấu trúc của mạng nơ-ron sinh học trong não người.  
                        - Nó bao gồm các **nơ-ron** (nodes) được tổ chức thành **lớp** (layers): lớp đầu vào (input layer), các lớp ẩn (hidden layers), và lớp đầu ra (output layer).  
                        - Neural Network đặc biệt mạnh trong việc xử lý các bài toán phi tuyến tính và học các đặc trưng phức tạp từ dữ liệu.
                        """
                )
                st.image("image1.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)
                # st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image1.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)

                st.markdown("---")
                st.markdown("### Cấu trúc Neural Network")

                st.markdown(
                    """
                    Mạng Nơ-ron Nhân tạo (Neural Network) có cấu trúc cơ bản bao gồm các thành phần chính sau:

                    1. **Lớp đầu vào (Input Layer)**:  
                    - Đây là nơi nhận dữ liệu thô từ bài toán (ví dụ: giá trị pixel của ảnh, số liệu thống kê, văn bản, v.v.).  
                    - Số lượng nơ-ron trong lớp này tương ứng với số đặc trưng (features) của dữ liệu đầu vào.

                    2. **Các lớp ẩn (Hidden Layers)**:  
                    - Là các lớp trung gian giữa lớp đầu vào và lớp đầu ra, nơi diễn ra quá trình xử lý và học hỏi.  
                    - Mỗi lớp ẩn bao gồm nhiều nơ-ron, và số lượng lớp ẩn cũng như nơ-ron trong mỗi lớp có thể thay đổi tùy thuộc vào độ phức tạp của bài toán.  
                    - Các nơ-ron trong lớp ẩn áp dụng **hàm kích hoạt (activation function)** như ReLU, Sigmoid hoặc Tanh để xử lý tính phi tuyến tính.
                    - **Lưu ý rằng:** một Neural Network chỉ có 1 lớp đầu vào và 1 lớp đầu ra nhưng có thể có nhiều các lớp ẩn 
                    3. **Lớp đầu ra (Output Layer)**:  
                    - Lớp này tạo ra kết quả cuối cùng của mạng (dự đoán hoặc phân loại).  
                    - Số lượng nơ-ron trong lớp đầu ra phụ thuộc vào loại bài toán:  
                        - **Phân loại nhị phân**: 1 nơ-ron (ví dụ: dùng hàm Sigmoid).  
                        - **Phân loại đa lớp**: Số nơ-ron bằng số lớp (ví dụ: dùng hàm Softmax).  
                        - **Hồi quy**: 1 hoặc nhiều nơ-ron tùy theo số lượng giá trị cần dự đoán.
                    """
                )
                st.image("image2.png", caption="Cấu trúc Neural Network có 2 hoặc nhiều lớp ẩn (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)
                # st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image2.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)

                st.markdown("---")
                st.markdown("### Các bước huấn luyện Neural Network")
                st.write("1. **Khởi tạo mô hình**: Xác định số lớp ẩn, số nơ-ron trong mỗi lớp, và hàm kích hoạt.")
                st.write("2. **Chuẩn hóa dữ liệu**: Đưa dữ liệu về dạng chuẩn để tăng hiệu quả huấn luyện.")
                st.write("3. **Lan truyền xuôi (Forward Propagation)**: Tính toán đầu ra từ đầu vào qua các lớp.")
                st.write("4. **Lan truyền ngược (Backpropagation)**: Cập nhật trọng số dựa trên hàm mất mát.")
                st.write("5. **Dự đoán**: Sử dụng mô hình đã huấn luyện để dự đoán trên dữ liệu mới.")
                st.markdown("---")
                st.markdown("### Các hàm kích hoạt trong Neural Network")
                # Hàm Sigmoid
                st.markdown("**1. Hàm Sigmoid:**")
                st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")
                st.markdown(
                    """
                    - **Trong đó:**
                        - $$e^x$$ và $$e^{-x}$$: là hàm mũ với cơ số $$e$$ và số mũ $$x$$ hoặc $$-x$$
                        - $$( x )$$: Giá trị đầu vào của nơ-ron (tổng có trọng số cộng với bias).  
                        - $$(( f(x) )$$: Đầu ra của hàm Sigmoid, nằm trong khoảng $$(((0, 1))$$.  
                    """
                )
                st.image("image3.png", caption="Biểu đồ hàm Sigmoid (Nguồn:https://byvn.net/qW4e)", use_container_width=True)
                # st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image3.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)

                # Hàm Tanh
                st.markdown("**2. Hàm Hyperbolic Tangent (Tanh):**")
                st.latex(r"f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}")
                st.markdown(
                    """
                    - **Trong đó:**
                        - $$( x )$$: Giá trị đầu vào của nơ-ron.  
                        - $$(( f(x) )$$: Đầu ra của hàm Tanh, nằm trong khoảng $$(((-1 , 1))$$.  

                    """
                )
                st.image("image4.png", caption="Biểu đồ hàm Hyperbolic Tangent (Tanh) (Nguồn:https://byvn.net/qW4e)", use_container_width=True)
                # st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image4.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)

                # Hàm ReLU
                st.markdown("**3. Hàm ReLU (Rectified Linear Unit):**")
                st.latex(r"f(x) = \max(0, x)")
                st.markdown(
                    """
                    - **Trong đó:**
                        - $$( x )$$: Giá trị đầu vào của nơ-ron.  
                        - $$(( f(x) )$$: Đầu ra của hàm ReLU, bằng 0 nếu $$( x < 0 )$$, bằng $$( x )$$ nếu $$( x \geq 0 )$$.  
                    """
                )
                st.image("image5.png", caption="Biểu đồ hàm ReLU (Rectified Linear Unit)(Nguồn:https://byvn.net/qW4e)", use_container_width=True)
                # st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image5.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)

                # Hàm Softmax
                st.markdown("**5. Hàm Softmax:**")
                st.latex(r"f(x_i) = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}")
                st.markdown(
                    """
                    - **Trong đó:**
                        - $$( x_i )$$: Giá trị đầu vào của nơ-ron thứ $$( i )$$.  
                        - $$( N )$$: Số lượng nơ-ron trong lớp đầu ra (tương ứng với số lớp trong bài toán phân loại).  
                        - $$( f(x_i) )$$: Đầu ra của hàm Softmax, nằm trong khoảng $$((0, 1))$$ và tổng các đầu ra bằng 1.  
                
                    """
                )
                st.image("image6.png", caption="Biểu đồ hàm Softmax (Rectified Linear Unit)(Nguồn:https://byvn.net/yvvj)", use_container_width=True)
                # st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image6.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)
                st.markdown("---")
                st.markdown(
                    """
                    ### Giải thích vai trò của hàm kích hoạt (Activation Function)
                    Hàm kích hoạt:
                    - **Tính phi tuyến:** Giúp mạng nơ-ron học các mối quan hệ phức tạp.  
                    - **Điều chỉnh đầu ra:** Giới hạn giá trị (VD: Sigmoid: (0, 1), Tanh: (-1, 1), ReLU: ≥0).  
                    - **Hỗ trợ học:** Cung cấp đạo hàm cho lan truyền ngược, tối ưu hóa mô hình.  
                    - **Kích hoạt nơ-ron:** Quyết định nơ-ron hoạt động (VD: ReLU loại giá trị âm).  
                    - **Cải thiện hiệu suất:** Giảm vấn đề vanishing gradient, tăng tốc huấn luyện.  
                    Là yếu tố cốt lõi giúp mạng nơ-ron linh hoạt và mạnh mẽ.
                    """
                )
                st.markdown(
                    """
                    ### Vai trò của hàm kích hoạt và liên hệ với công thức
                    Hàm kích hoạt biến đổi đầu vào $$x$$ thành đầu ra $$f(x)$$:  
                    - **Sigmoid $$f(x) = \\frac{1}{1 + e^{-x}}$$:** Đưa $$x$$ vào (0, 1), thêm tính phi tuyến cho phân loại nhị phân.  

                    - **Tanh $$f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$$:** Chuẩn hóa $$x$$ vào (-1, 1), cân bằng giá trị âm/dương.  

                    - **ReLU $$f(x) = \\max(0, x)$$:** Loại giá trị âm, tăng thưa thớt, giảm gradient biến mất.  

                    - **Softmax $$f(x_i) = \\frac{e^{z_i}}{\\sum e^{z_j}}$$:** Chuẩn hóa $$x_i$$ thành xác suất (0, 1), tổng bằng 1, cho phân loại đa lớp.  
                    Công thức quyết định cách $$x$$ được biến đổi, hỗ trợ học phi tuyến và tối ưu hóa.
                    """,
                    unsafe_allow_html=True  # Cho phép Streamlit render LaTeX
                )
                st.markdown("---")
                st.markdown("### Công thức toán học")
                st.markdown("**1. Lan truyền xuôi (Forward Propagation):**")
                st.latex(r"h_l = f(W_l h_{l-1} + b_l)")
                st.markdown(
                        """
                        - **Trong đó:**
                        - $$( h_l )$$: Đầu ra của lớp $$( l )$$.  
                        - $$( W_l )$$: Ma trận trọng số của lớp $$( l )$$.  
                        - $$( h_{l-1} )$$: Đầu ra của lớp trước đó (hoặc dữ liệu đầu vào nếu là lớp đầu tiên).  
                        - $$( b_l )$$: Vector bias của lớp $$( l )$$.  
                        - $$( f )$$: Hàm kích hoạt (ví dụ: ReLU, Sigmoid, Tanh).
                        """
                )
                st.markdown("**2. Hàm mất mát (Loss Function) - Cross-Entropy cho phân loại:**")
                st.latex(r"L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]")
                st.markdown(
                        """
                        - **Trong đó:**
                        - $$( y_i )$$: Nhãn thực tế của mẫu $$( i )$$ (0 hoặc 1).  
                        - $$( \hat{y}_i )$$: Dự đoán của mô hình cho mẫu $$( i )$$ (xác suất từ 0 đến 1).  
                        - $$( N )$$: Số mẫu.
                        """
                )
                st.markdown("**3. Lan truyền ngược (Backpropagation) - Cập nhật trọng số:**")
                st.latex(r"W_l = W_l - \eta \frac{\partial L}{\partial W_l}")
                st.markdown(
                    r"""
                    - **Trong đó:**
                    - $$\eta$$: Tốc độ học (learning rate).  
                    - $$\frac{\partial L}{\partial W_l}$$: Đạo hàm của hàm mất mát theo trọng số $$W_l$$.
                    """
                )
                st.markdown("---")
                st.markdown("### Ưu điểm & Nhược điểm của Neural Network")
                st.table({
                        "**Ưu điểm**": [
                            "Học được các đặc trưng phi tuyến tính phức tạp.",
                            "Linh hoạt với nhiều loại dữ liệu và bài toán.",
                            "Hiệu quả cao với dữ liệu lớn khi được tối ưu tốt."
                        ],
                        "**Nhược điểm**": [
                            "Yêu cầu lượng dữ liệu lớn để huấn luyện.",
                            "Tốn tài nguyên tính toán.",
                            "Khó diễn giải kết quả."
                        ]
                })
            elif model_option == "Pseudo Labelling (Gắn nhãn Giả)":
                st.markdown("## 🔹 Pseudo Labelling (Gắn nhãn Giả)")
                st.markdown("---")
                st.markdown("### Khái niệm Pseudo Labelling")
                st.markdown(
                        """
                        - **Pseudo Labelling (Gắn nhãn Giả)** là một kỹ thuật học bán giám sát **(semi-supervised learning)** nhằm tận dụng dữ liệu chưa được gắn nhãn **(unlabeled data)** để cải thiện hiệu suất mô hình học máy.  
                        - Ý tưởng chính là sử dụng mô hình đã được huấn luyện trên dữ liệu có nhãn **(labeled data)** để dự đoán nhãn cho dữ liệu chưa có nhãn, sau đó sử dụng các nhãn giả **(pseudo-labels)** này để huấn luyện lại mô hình.  
                        - Kỹ thuật này đặc biệt hữu ích khi lượng dữ liệu có nhãn ít, nhưng dữ liệu chưa có nhãn lại dồi dào.
                        """
                )
                # st.image("pseudo_label_diagram.png", caption="Quy trình Pseudo Labelling (Nguồn: Tự tạo hoặc tham khảo từ tài liệu học thuật)", use_container_width=True)

                st.markdown("---")
                st.markdown("### Quy trình Pseudo Labelling")
                st.markdown(
                    """
                    1. **Huấn luyện mô hình với dữ liệu có nhãn (labeled data):**  
                        - Ở phần trên cùng, bạn thấy một tập dữ liệu có nhãn (**labeled data**) được biểu thị bằng các chấm tròn màu xanh và đỏ (mỗi màu đại diện cho một lớp khác nhau).  
                        - Dữ liệu này được sử dụng để huấn luyện một mô hình ban đầu (**Model**).  
                        - Mũi tên màu đỏ từ "labeled data" đến "Model" thể hiện quá trình huấn luyện mô hình bằng dữ liệu có nhãn.  

                    2. **Sử dụng mô hình đã huấn luyện để dự đoán nhãn cho dữ liệu chưa có nhãn (unlabeled data):**  
                        - Bên dưới tập dữ liệu có nhãn là tập dữ liệu chưa có nhãn (**unlabeled data**), được biểu thị bằng các chấm tròn màu xám (chưa được phân loại).  
                        - Mô hình đã huấn luyện ở bước 1 được sử dụng để dự đoán nhãn cho tập dữ liệu chưa có nhãn này.  
                        - Kết quả là tập dữ liệu **pseudo-labeled data** (dữ liệu được gắn nhãn giả), được biểu thị bằng các chấm tròn màu xanh và đỏ tương tự dữ liệu có nhãn ban đầu.  
                        - Mũi tên từ "unlabeled data" đến "Model" và sau đó đến "pseudo-labeled data" thể hiện quá trình này.  

                    3. **Huấn luyện lại mô hình với cả dữ liệu có nhãn và dữ liệu được gắn nhãn giả:**  
                        - Sau khi có tập dữ liệu **pseudo-labeled**, nó được kết hợp với tập dữ liệu có nhãn ban đầu (**labeled data**).  
                        - Cả hai tập dữ liệu này (**labeled data** và **pseudo-labeled data**) được sử dụng để huấn luyện lại mô hình (**Model**).  
                        - Mũi tên từ cả hai tập dữ liệu (**labeled data** và **pseudo-labeled data**) đến "Model" ở dưới cùng thể hiện bước huấn luyện lại này.
                    """
                )
                st.image("image7.png", caption="Quy trình cơ bản của Pseudo Labelling (Nguồn: https://byvn.net/dJoE)", use_container_width=True)

                st.markdown("---")
                st.markdown("### Công thức toán học")
                st.markdown("**1. Dự đoán nhãn giả:**")
                st.latex(r"\hat{y}_u = \arg\max_{y} P(y|x_u; \theta)")
                st.markdown(
                        """
                        - **Trong đó:**
                        - $$x_u$$: Dữ liệu chưa có nhãn (unlabeled data).  
                        - $$\hat{y}_u$$: Nhãn giả được dự đoán cho $$x_u$$.  
                        - $$P(y|x_u;\theta)$$: Xác suất dự đoán của mô hình với tham số $$\theta$$.  
                        - $$ arg\max$$: Lấy nhãn có xác suất cao nhất.
                        """
                )
                st.markdown("**2. Hàm mất mát kết hợp:**")
                st.latex(r"L = L_l + \alpha L_u")
                st.markdown(
                        """
                        - **Trong đó:**
                        - $$L_l$$: Hàm mất mát trên dữ liệu có nhãn (labeled data), ví dụ: Cross-Entropy.  
                        - $$L_u$$: Hàm mất mát trên dữ liệu có nhãn giả (pseudo-labeled data).  
                        - $$alpha$$: Hệ số cân bằng giữa hai thành phần mất mát (thường được điều chỉnh trong khoảng [0, 1]).
                        """
                )

                st.markdown("---")
                st.markdown("### Ưu điểm & Nhược điểm của Pseudo Labelling")
                st.table({
                        "**Ưu điểm**": [
                            "Tận dụng được dữ liệu chưa có nhãn, giảm chi phí gắn nhãn thủ công.",
                            "Cải thiện hiệu suất mô hình khi dữ liệu có nhãn bị hạn chế.",
                            "Dễ triển khai và kết hợp với các mô hình học máy khác."
                        ],
                        "**Nhược điểm**": [
                            "Nhãn giả có thể chứa sai sót, ảnh hưởng đến chất lượng mô hình.",
                            "Hiệu quả phụ thuộc vào độ chính xác của mô hình ban đầu.",
                            "Cần điều chỉnh cẩn thận tham số (ví dụ: $$alpha$$) để tránh overfitting."
                        ]
                })

    with tab_samples:
        with st.expander("**Phân chia dữ liệu**", expanded=True):
            if "train_images" in st.session_state:
                train_images = st.session_state.train_images
                train_labels = st.session_state.train_labels
                test_images = st.session_state.test_images
                test_labels = st.session_state.test_labels

                # Kết hợp dữ liệu thành X và y
                X = np.concatenate((train_images, test_images), axis=0)
                y = np.concatenate((train_labels, test_labels), axis=0)
                X = X.reshape(X.shape[0], -1)

                # Thêm phần người dùng chọn số mẫu dữ liệu
                total_available_samples = X.shape[0]
                st.write(f"📊 **Tổng số mẫu dữ liệu hiện có**: {total_available_samples}")
                num_samples = st.number_input(
                    "🔹 Chọn số mẫu dữ liệu muốn sử dụng", 
                    min_value=100, 
                    max_value=total_available_samples, 
                    value=min(10000, total_available_samples),  # Mặc định là 10,000 hoặc nhỏ hơn nếu tổng mẫu ít hơn
                    step=100, 
                    key="num_samples"
                )

                # Lấy mẫu ngẫu nhiên từ dữ liệu dựa trên số lượng người dùng chọn
                if num_samples < total_available_samples:
                    sampled_indices = np.random.choice(total_available_samples, num_samples, replace=False)
                    X = X[sampled_indices]
                    y = y[sampled_indices]
                st.write(f"🔹 Số mẫu dữ liệu được chọn: `{X.shape[0]}`")

                # (0) Chia tập train/test với tỷ lệ do người dùng chọn
                test_size_percent = st.slider("🔹 Chọn % tỷ lệ tập test", min_value=10, max_value=50, value=20, step=5, key="test_size") / 100
                X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size_percent, random_state=42)

                # (1) Lấy 1% số lượng ảnh cho mỗi class từ tập train làm tập train ban đầu
                def sample_per_class(X, y, percentage=0.01):
                    unique_classes = np.unique(y)
                    X_sampled = []
                    y_sampled = []
                    sampled_indices = []
                    for cls in unique_classes:
                        cls_indices = np.where(y == cls)[0]
                        num_samples = max(1, int(len(cls_indices) * percentage))  # Đảm bảo ít nhất 1 mẫu
                        cls_sampled_indices = np.random.choice(cls_indices, num_samples, replace=False)
                        sampled_indices.extend(cls_sampled_indices)
                        X_sampled.append(X[cls_sampled_indices])
                        y_sampled.append(y[cls_sampled_indices])
                    return np.concatenate(X_sampled), np.concatenate(y_sampled), sampled_indices

                X_train_initial, y_train_initial, sampled_indices = sample_per_class(X_train_full, y_train_full, percentage=0.01)

                # Tạo mặt nạ để loại bỏ các chỉ số đã chọn
                mask = np.ones(len(X_train_full), dtype=bool)
                mask[sampled_indices] = False
                X_unlabeled = X_train_full[mask]
                y_unlabeled = y_train_full[mask]

                # Lưu vào session_state
                st.session_state.X_train_full = X_train_full  # Tập train đầy đủ
                st.session_state.y_train_full = y_train_full
                st.session_state.X_train_initial = X_train_initial  # Tập 1% ban đầu
                st.session_state.y_train_initial = y_train_initial
                st.session_state.X_unlabeled = X_unlabeled  # Tập chưa gán nhãn
                st.session_state.y_unlabeled = y_unlabeled  # Nhãn thật của tập chưa gán (dùng để kiểm tra sau)
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                # Hiển thị thông tin phân chia hoàn thiện
                total_samples = X.shape[0]
                train_percent = (X_train_full.shape[0] / total_samples) * 100
                test_percent = (X_test.shape[0] / total_samples) * 100
                train_initial_percent = (X_train_initial.shape[0] / total_samples) * 100
                unlabeled_percent = (X_unlabeled.shape[0] / total_samples) * 100

                st.write(f"📊 **Tỷ lệ phân chia**: Train = {train_percent:.1f}% , Test = {test_percent:.1f}%")
                st.write(f"🔹 Train ban đầu (1%) = {train_initial_percent:.1f}% , Unlabeled = {unlabeled_percent:.1f}%")
                st.write("✅ Dữ liệu đã được xử lý và chia tách.")
                st.write(f"🔹 Kích thước tập train ban đầu (1%): `{X_train_initial.shape}`")
                st.write(f"🔹 Kích thước tập unlabeled: `{X_unlabeled.shape}`")
                st.write(f"🔹 Kích thước tập kiểm tra: `{X_test.shape}`")
            else:
                st.error("🚨 Dữ liệu chưa được nạp. Hãy đảm bảo `train_images`, `train_labels` và `test_images` đã được tải trước khi chạy.")


    with tab_preprocess:
        with st.expander("**Huấn luyện Neural Network**", expanded=True):
            if "X_train_initial" not in st.session_state:
                st.error("🚨 Vui lòng phân chia dữ liệu ở tab 'Phân chia dữ liệu' trước khi tiếp tục.")
            else:
                X_train_initial = st.session_state.X_train_initial
                y_train_initial = st.session_state.y_train_initial
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                # Chuẩn hóa dữ liệu
                X_train_initial = X_train_initial / 255.0
                X_test = X_test / 255.0

                # In lại thông tin của tập train ban đầu (1%)
                total_samples = X_train_initial.shape[0] + st.session_state.X_unlabeled.shape[0] + X_test.shape[0]
                train_initial_percent = (X_train_initial.shape[0] / total_samples) * 100
                unlabeled_percent = (st.session_state.X_unlabeled.shape[0] / total_samples) * 100
                test_percent = (X_test.shape[0] / total_samples) * 100

                st.write(f"📊 **Thông tin tập dữ liệu ban đầu**:")
                st.write(f"🔹 Train ban đầu (1%)={train_initial_percent:.1f}%, Kích thước: `{X_train_initial.shape}`")
                st.write(f"🔹 Unlabeled={unlabeled_percent:.1f}%, Kích thước: `{st.session_state.X_unlabeled.shape}`")
                st.write(f"🔹 Test={test_percent:.1f}%, Kích thước: `{X_test.shape}`")
                st.markdown("---")

                # Cấu hình huấn luyện ban đầu
                num_hidden_layers_initial = st.slider("🔹 Số lớp ẩn (huấn luyện ban đầu)", min_value=1, max_value=5, value=3, key="num_hidden_layers_initial")
                hidden_layer_neurons_initial = [st.number_input(f"🔹 Số nơ-ron lớp ẩn {i+1} (ban đầu)", min_value=32, max_value=1024, value=512 if i == 0 else 256 if i == 1 else 128, key=f"neurons_initial_{i}") for i in range(num_hidden_layers_initial)]
                activation_initial = st.selectbox("🔹 Hàm kích hoạt (ban đầu)", options=['relu', 'sigmoid', 'tanh', 'softmax'], index=0, key="activation_initial")
                epochs_initial = st.slider("🔹 Số epoch (ban đầu)", min_value=5, max_value=50, value=10, key="epochs_initial")
                batch_size_initial = st.selectbox("🔹 Batch size (ban đầu)", options=[32, 64, 128, 256], index=0, key="batch_size_initial")
                optimizer_initial = st.selectbox("🔹 Bộ tối ưu (ban đầu)", options=['adam', 'sgd', 'rmsprop'], index=0, key="optimizer_initial")
                learning_rate_initial = st.number_input("🔹 Learning Rate (ban đầu)", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f", key="learning_rate_initial")

                # Thêm phần đặt tên mô hình
                model_name = st.text_input("🔹 Đặt tên cho mô hình (dùng để demo sau này)", value="My_NN_Model", key="model_name")

                if st.button("🚀 Bắt đầu huấn luyện Neural Network trên 1% dữ liệu", key="train_initial_button"):
                    with st.spinner(f"Đang huấn luyện mô hình '{model_name}'..."):
                        # Tạo thanh trạng thái
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        with mlflow.start_run(run_name=model_name):  # Sử dụng tên người dùng đặt
                            # Ghi lại tham số
                            mlflow.log_param("num_hidden_layers", num_hidden_layers_initial)
                            for i, neurons in enumerate(hidden_layer_neurons_initial):
                                mlflow.log_param(f"hidden_layer_{i+1}_neurons", neurons)
                            mlflow.log_param("activation", activation_initial)
                            mlflow.log_param("epochs", epochs_initial)
                            mlflow.log_param("batch_size", batch_size_initial)
                            mlflow.log_param("optimizer", optimizer_initial)
                            mlflow.log_param("learning_rate", learning_rate_initial)

                            # Xây dựng mô hình NN
                            input_shape = X_train_initial.shape[1]
                            num_classes = len(np.unique(y_train_initial))
                            model = models.Sequential([
                                layers.Input(shape=(input_shape,)),
                                *[layer for i in range(num_hidden_layers_initial) for layer in [layers.Dense(hidden_layer_neurons_initial[i], activation=activation_initial), layers.Dropout(0.2)]],
                                layers.Dense(num_classes, activation='softmax')
                            ])

                            # Cấu hình optimizer với learning rate được chọn
                            if optimizer_initial == 'adam':
                                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_initial)
                            elif optimizer_initial == 'sgd':
                                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_initial, momentum=0.9)
                            else:  # rmsprop
                                optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_initial)

                            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                            # Callback để cập nhật thanh trạng thái
                            class ProgressCallback(tf.keras.callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress = (epoch + 1) / epochs_initial
                                    progress_bar.progress(min(int(progress * 100), 100))
                                    status_text.text(f"Tiến trình huấn luyện: {int(progress * 100)}%")

                            # Huấn luyện mô hình trên tập 1% ban đầu
                            history = model.fit(
                                X_train_initial, 
                                y_train_initial, 
                                epochs=epochs_initial, 
                                batch_size=batch_size_initial, 
                                verbose=0,
                                callbacks=[ProgressCallback()]
                            )

                            for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
                                mlflow.log_metric("train_loss", loss, step=epoch)
                                mlflow.log_metric("train_accuracy", acc, step=epoch)

                            # Đánh giá nhanh trên tập test
                            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                            st.session_state["initial_test_accuracy"] = test_accuracy
                            mlflow.log_metric("test_loss", test_loss)
                            mlflow.log_metric("test_accuracy", test_accuracy)

                            # Cập nhật thanh trạng thái hoàn tất
                            progress_bar.progress(100)
                            status_text.text("Huấn luyện hoàn tất: 100%")
                            st.success(f"✅ Huấn luyện mô hình '{model_name}' hoàn tất! Test Accuracy: {test_accuracy:.4f}")

                            # Lưu mô hình vào session_state
                            st.session_state['initial_model'] = model
                            st.session_state['initial_history'] = history
                            # Không cần gán st.session_state['model_name'] vì giá trị đã được lưu tự động trong st.session_state['model_name'] qua key

    with tab_pseudo_labeling:
        with st.expander("**Huấn luyện với Pseudo-Labeling**", expanded=True):
            if "X_train_initial" not in st.session_state or "initial_model" not in st.session_state:
                st.error("🚨 Vui lòng phân chia dữ liệu ở tab 'Phân chia dữ liệu' và huấn luyện Neural Network trước khi tiếp tục.")
            else:
                X_train_initial = st.session_state.X_train_initial
                y_train_initial = st.session_state.y_train_initial
                X_unlabeled = st.session_state.X_unlabeled
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                model = st.session_state['initial_model']

                # Chuẩn hóa dữ liệu
                X_train_initial = X_train_initial / 255.0
                X_unlabeled = X_unlabeled / 255.0
                X_test = X_test / 255.0

                # In thông tin ban đầu
                total_samples = X_train_initial.shape[0] + X_unlabeled.shape[0] + X_test.shape[0]
                st.write(f"📊 **Tổng số mẫu:** {total_samples}")
                st.write(f"🔹 Train ban đầu (1%): `{X_train_initial.shape}`")
                st.write(f"🔹 Unlabeled: `{X_unlabeled.shape}`")
                st.write(f"🔹 Test: `{X_test.shape}`")
                st.markdown("---")

                # Cấu hình huấn luyện với pseudo-labeling
                epochs_pl = st.slider("🔹 Số epoch mỗi vòng (PL)", min_value=5, max_value=50, value=20, key="epochs_pl")
                batch_size_pl = st.selectbox("🔹 Batch size (PL)", options=[32, 64, 128, 256], index=0, key="batch_size_pl")
                threshold = st.number_input("🔹 Ngưỡng Pseudo-Label", min_value=0.5, max_value=0.99, value=0.90, step=0.01, format="%.2f", key="threshold")
                max_iterations = st.slider("🔹 Số vòng lặp tối đa", min_value=1, max_value=20, value=10, key="max_iterations")
                learning_rate_pl = st.number_input("🔹 Learning Rate (PL)", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f", key="learning_rate_pl")
                
                # Thêm phần đặt tên mô hình
                pseudo_model_name = st.text_input("🔹 Đặt tên cho mô hình Pseudo-Labeling", value="My_Pseudo_Model", key="pseudo_model_name")

                if st.button("🚀 Bắt đầu huấn luyện với Pseudo-Labeling", key="pseudo_train_button"):
                    with st.spinner(f"Đang huấn luyện mô hình '{pseudo_model_name}'..."):
                        # Tạo thanh trạng thái
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        with mlflow.start_run(run_name=pseudo_model_name):  # Sử dụng tên người dùng đặt
                            # Ghi lại tham số
                            mlflow.log_param("epochs_per_iteration", epochs_pl)
                            mlflow.log_param("batch_size", batch_size_pl)
                            mlflow.log_param("threshold", threshold)
                            mlflow.log_param("max_iterations", max_iterations)
                            mlflow.log_param("learning_rate", learning_rate_pl)

                            # Cập nhật optimizer
                            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_pl)
                            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                            # Khởi tạo tập labeled và unlabeled
                            X_current_labeled = X_train_initial.copy()
                            y_current_labeled = y_train_initial.copy()
                            X_current_unlabeled = X_unlabeled.copy()

                            # Khởi tạo DataFrame để lưu kết quả
                            results_df = pd.DataFrame(columns=[
                                "Vòng", "Test Accuracy", "Min Prob", "Max Prob", "Mean Prob",
                                "Số mẫu gán nhãn", "Tập labeled", "Số mẫu unlabeled"
                            ])

                            # Tạo placeholder để hiển thị bảng và hình ảnh
                            table_placeholder = st.empty()
                            image_placeholder = st.empty()

                            # Quá trình Pseudo-Labeling
                            for iteration in range(max_iterations):
                                with st.spinner(f"Đang huấn luyện vòng {iteration + 1}/{max_iterations} cho mô hình '{pseudo_model_name}'..."):
                                    # Callback để cập nhật thanh trạng thái trong mỗi vòng
                                    class ProgressCallback(tf.keras.callbacks.Callback):
                                        def on_epoch_end(self, epoch, logs=None):
                                            progress = ((iteration * epochs_pl) + (epoch + 1)) / (max_iterations * epochs_pl)
                                            progress_bar.progress(min(int(progress * 100), 100))
                                            status_text.text(f"Tiến trình huấn luyện: {int(progress * 100)}%")

                                    # (2) Huấn luyện mô hình trên tập labeled hiện tại
                                    history = model.fit(
                                        X_current_labeled,
                                        y_current_labeled,
                                        epochs=epochs_pl,
                                        batch_size=batch_size_pl,
                                        verbose=0,
                                        callbacks=[ProgressCallback()]
                                    )

                                    # Ghi lại chỉ số từ history cho mỗi vòng
                                    for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
                                        mlflow.log_metric(f"train_loss_iter_{iteration+1}", loss, step=epoch)
                                        mlflow.log_metric(f"train_accuracy_iter_{iteration+1}", acc, step=epoch)

                                    # Đánh giá trên tập test
                                    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                                    st.session_state["pseudo_test_accuracy"] = test_accuracy
                                    mlflow.log_metric(f"test_loss_iter_{iteration+1}", test_loss)
                                    mlflow.log_metric(f"test_accuracy_iter_{iteration+1}", test_accuracy)

                                    # (3) Dự đoán nhãn cho tập unlabeled
                                    probs = model.predict(X_current_unlabeled, verbose=0)
                                    max_probs = np.max(probs, axis=1)
                                    pseudo_labels = np.argmax(probs, axis=1)

                                    # (4) Lọc các mẫu vượt ngưỡng
                                    confident_mask = max_probs >= threshold
                                    X_confident = X_current_unlabeled[confident_mask]
                                    y_confident = pseudo_labels[confident_mask]

                                    # Ghi lại các chỉ số thống kê
                                    mlflow.log_metric(f"min_prob_iter_{iteration+1}", np.min(max_probs))
                                    mlflow.log_metric(f"max_prob_iter_{iteration+1}", np.max(max_probs))
                                    mlflow.log_metric(f"mean_prob_iter_{iteration+1}", np.mean(max_probs))
                                    mlflow.log_metric(f"num_labeled_samples_iter_{iteration+1}", len(X_confident))

                                    # (5) Minh họa các mẫu được gán nhãn Pseudo
                                    if len(X_confident) > 0:
                                        st.write(f"### Minh họa các mẫu được gán nhãn Pseudo ở vòng {iteration + 1}")
                                        num_samples_to_show = min(5, len(X_confident))  # Hiển thị tối đa 5 mẫu
                                        cols = st.columns(num_samples_to_show)
                                        for i in range(num_samples_to_show):
                                            # Giả sử dữ liệu là ảnh (ví dụ: 28x28 như MNIST)
                                            sample_image = X_confident[i].reshape(28, 28)  # Điều chỉnh kích thước tùy theo dữ liệu
                                            with cols[i]:
                                                st.image(sample_image, caption=f"Nhãn: {y_confident[i]}, Xác suất: {max_probs[confident_mask][i]:.4f}", width=100)

                                    # (6) Cập nhật tập labeled và unlabeled
                                    if len(X_confident) > 0:
                                        X_current_labeled = np.concatenate([X_current_labeled, X_confident])
                                        y_current_labeled = np.concatenate([y_current_labeled, y_confident])
                                        X_current_unlabeled = X_current_unlabeled[~confident_mask]

                                    # Thêm kết quả vào DataFrame
                                    new_row = pd.DataFrame({
                                        "Vòng": [iteration + 1],
                                        "Test Accuracy": [test_accuracy],
                                        "Min Prob": [np.min(max_probs)],
                                        "Max Prob": [np.max(max_probs)],
                                        "Mean Prob": [np.mean(max_probs)],
                                        "Số mẫu gán nhãn": [len(X_confident)],
                                        "Tập labeled": [str(X_current_labeled.shape)],
                                        "Số mẫu unlabeled": [X_current_unlabeled.shape[0]]
                                    })
                                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                                    # Cập nhật bảng trong placeholder
                                    with table_placeholder.container():
                                        st.markdown(f"### 🔄 **Kết quả huấn luyện qua các vòng**")
                                        st.dataframe(results_df.style.format({
                                            "Test Accuracy": "{:.4f}",
                                            "Min Prob": "{:.4f}",
                                            "Max Prob": "{:.4f}",
                                            "Mean Prob": "{:.4f}",
                                            "Số mẫu gán nhãn": "{:d}",
                                            "Số mẫu unlabeled": "{:d}"
                                        }))

                                    # Điều kiện dừng
                                    if len(X_confident) == 0:
                                        st.write(f"🔹 Vòng {iteration + 1}: Không có mẫu nào vượt ngưỡng {threshold}. Dừng lại.")
                                        break
                                    if len(X_confident) < 10:
                                        st.write(f"🔹 Vòng {iteration + 1}: Số mẫu gán nhãn quá ít ({len(X_confident)}). Dừng lại.")
                                        break
                                    if len(X_current_unlabeled) == 0:
                                        st.write("✅ Đã gán nhãn hết tập unlabeled!")
                                        break

                            # Đánh giá mô hình cuối cùng
                            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                            mlflow.log_metric("final_test_loss", test_loss)
                            mlflow.log_metric("final_test_accuracy", test_accuracy)

                            # Lưu mô hình cuối cùng vào MLflow
                            mlflow.keras.log_model(model, "pseudo_labeled_model")

                            # Cập nhật thanh trạng thái hoàn tất
                            progress_bar.progress(100)
                            status_text.text("Huấn luyện hoàn tất: 100%")
                            st.success(f"✅ Quá trình huấn luyện mô hình '{pseudo_model_name}' hoàn tất! Test Accuracy cuối cùng: {test_accuracy:.4f}")
                            st.session_state['pseudo_model'] = model


    with tab_demo:
        with st.expander("**Demo Dự Đoán Với Mô Hình**", expanded=True):
            if "initial_model" not in st.session_state and "pseudo_model" not in st.session_state:
                st.error("🚨 Vui lòng huấn luyện ít nhất một mô hình trước khi sử dụng Demo!")
            else:
                # Lấy danh sách các mô hình đã huấn luyện
                available_models = {}
                if "initial_model" in st.session_state:
                    initial_model_name = st.session_state.get("model_name", "My_NN_Model")
                    available_models[initial_model_name] = st.session_state["initial_model"]
                if "pseudo_model" in st.session_state:
                    pseudo_model_name = st.session_state.get("pseudo_model_name", "My_Pseudo_Model")
                    available_models[pseudo_model_name] = st.session_state["pseudo_model"]

                # Tùy chọn chọn mô hình
                selected_model_name = st.selectbox(
                    "🔹 Chọn mô hình để dự đoán",
                    options=list(available_models.keys()),
                    key="selected_model"
                )
                selected_model = available_models[selected_model_name]

                # Tùy chọn đầu vào: Tải ảnh hoặc vẽ ảnh
                input_method = st.radio(
                    "🔹 Chọn phương thức nhập ảnh",
                    options=["Tải ảnh lên", "Vẽ ảnh"],
                    key="input_method"
                )

                st.markdown("---")
                if input_method == "Tải ảnh lên":
                    uploaded_file = st.file_uploader("Tải lên một ảnh (PNG/JPG, 28x28)", type=["png", "jpg", "jpeg"])
                    if uploaded_file is not None:
                        image = Image.open(uploaded_file).convert("L")
                        image = image.resize((28, 28))
                        image_array = np.array(image) / 255.0
                        st.image(image, caption="Ảnh đã tải lên", width=150)

                        if st.button("🚀 Dự đoán", key="predict_uploaded"):
                            with st.spinner("Đang dự đoán..."):
                                input_data = image_array.reshape(1, 784)
                                prediction = selected_model.predict(input_data, verbose=0)
                                predicted_label = np.argmax(prediction)
                                confidence = np.max(prediction)
                                st.success(f"✅ Kết quả dự đoán: **Nhãn {predicted_label}** (Xác suất: {confidence:.4f})")
                                fig, ax = plt.subplots()
                                ax.bar(range(10), prediction[0])
                                ax.set_xlabel("Nhãn")
                                ax.set_ylabel("Xác suất")
                                ax.set_title(f"Dự đoán với mô hình '{selected_model_name}'")
                                st.pyplot(fig)

                elif input_method == "Vẽ ảnh":
                    st.write("Vẽ một chữ số (0-9) trên canvas bên dưới:")
                    canvas_result = st_canvas(
                        fill_color="black",        # Nền đen
                        stroke_width=20,           # Độ dày nét vẽ
                        stroke_color="white",      # Nét trắng
                        background_color="black",  # Nền canvas đen
                        height=150,
                        width=150,
                        drawing_mode="freedraw",
                        key="canvas"
                    )

                    if canvas_result.image_data is not None:
                        # Xử lý ảnh từ canvas
                        image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                        image = image.resize((28, 28), Image.Resampling.LANCZOS)
                        image_array = np.array(image)
                        image_array = np.where(image_array < 128, 0, 255)  # Đảm bảo nền đen, chữ trắng
                        image_array = image_array.astype("float32") / 255.0

                        st.image(image_array, caption="Ảnh đã xử lý từ canvas", width=150)

                        if st.button("🚀 Dự đoán", key="predict_drawn"):
                            with st.spinner("Đang dự đoán..."):
                                input_data = image_array.reshape(1, 784)
                                prediction = selected_model.predict(input_data, verbose=0)
                                predicted_label = np.argmax(prediction)
                                confidence = np.max(prediction)
                                st.success(f"✅ Kết quả dự đoán: **Nhãn {predicted_label}** (Xác suất: {confidence:.4f})")
                                fig, ax = plt.subplots()
                                ax.bar(range(10), prediction[0])
                                ax.set_xlabel("Nhãn")
                                ax.set_ylabel("Xác suất")
                                ax.set_title(f"Dự đoán với mô hình '{selected_model_name}'")
                                st.pyplot(fig)



    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "PseudoLabelingExperiment"

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

            # 3) Danh sách các Run đã log và hiển thị chi tiết
            st.subheader("Danh sách các Run đã log")
            if runs:
                selected_run_id = st.selectbox("Chọn Run để xem chi tiết:",
                                            options=list(run_options.keys()),
                                            format_func=lambda x: run_options[x])

                # 4) Hiển thị thông tin chi tiết của Run được chọn
                selected_run = client.get_run(selected_run_id)
                run_name = selected_run.data.tags.get('mlflow.runName', 'Unnamed')
                st.write(f"**Run ID:** {selected_run_id}")
                st.write(f"**Run Name:** {run_name}")

                # Hiển thị các tham số đã log
                st.markdown("### Tham số đã log")
                params = {}

                # Kiểm tra loại mô hình dựa trên tham số đặc trưng
                if "num_hidden_layers" in selected_run.data.params:  # Neural Network
                    params = {
                        "Num Hidden Layers": selected_run.data.params.get("num_hidden_layers", "N/A"),
                        "Hidden Layer 1 Neurons": selected_run.data.params.get("hidden_layer_1_neurons", "N/A"),
                        "Hidden Layer 2 Neurons": selected_run.data.params.get("hidden_layer_2_neurons", "N/A"),
                        "Hidden Layer 3 Neurons": selected_run.data.params.get("hidden_layer_3_neurons", "N/A"),
                        "Hidden Layer 4 Neurons": selected_run.data.params.get("hidden_layer_4_neurons", "N/A"),
                        "Hidden Layer 5 Neurons": selected_run.data.params.get("hidden_layer_5_neurons", "N/A"),
                        "Activation": selected_run.data.params.get("activation", "N/A"),
                        "Epochs": selected_run.data.params.get("epochs", "N/A"),
                        "Batch Size": selected_run.data.params.get("batch_size", "N/A"),
                        "Optimizer": selected_run.data.params.get("optimizer", "N/A"),
                        "Learning Rate": selected_run.data.params.get("learning_rate", "N/A"),
                    }
                elif "threshold" in selected_run.data.params:  # Pseudo-Labeling
                    params = {
                        "Epochs per Iteration": selected_run.data.params.get("epochs_per_iteration", "N/A"),
                        "Batch Size": selected_run.data.params.get("batch_size", "N/A"),
                        "Threshold": selected_run.data.params.get("threshold", "N/A"),
                        "Max Iterations": selected_run.data.params.get("max_iterations", "N/A"),
                        "Learning Rate": selected_run.data.params.get("learning_rate", "N/A"),
                    }
                st.json(params)

                # Hiển thị các chỉ số đã log
                st.markdown("### Chỉ số đã log")
                metrics = {}

                if "num_hidden_layers" in selected_run.data.params:  # Neural Network
                    metrics = {
                        "Train Loss (Last Epoch)": selected_run.data.metrics.get("train_loss", "N/A"),
                        "Train Accuracy (Last Epoch)": selected_run.data.metrics.get("train_accuracy", "N/A"),
                        "Test Loss": selected_run.data.metrics.get("test_loss", "N/A"),
                        "Test Accuracy": selected_run.data.metrics.get("test_accuracy", "N/A"),
                    }
                elif "threshold" in selected_run.data.params:  # Pseudo-Labeling
                    metrics = {
                        "Final Test Loss": selected_run.data.metrics.get("final_test_loss", "N/A"),
                        "Final Test Accuracy": selected_run.data.metrics.get("final_test_accuracy", "N/A"),
                    }
                    # Hiển thị chỉ số theo từng vòng lặp
                    st.markdown("#### Chỉ số theo từng vòng lặp")
                    max_iterations = int(selected_run.data.params.get("max_iterations", 1))
                    for i in range(1, max_iterations + 1):
                        iter_metrics = {
                            f"Iteration {i} - Train Loss (Last Epoch)": selected_run.data.metrics.get(f"train_loss_iter_{i}", "N/A"),
                            f"Iteration {i} - Train Accuracy (Last Epoch)": selected_run.data.metrics.get(f"train_accuracy_iter_{i}", "N/A"),
                            f"Iteration {i} - Test Loss": selected_run.data.metrics.get(f"test_loss_iter_{i}", "N/A"),
                            f"Iteration {i} - Test Accuracy": selected_run.data.metrics.get(f"test_accuracy_iter_{i}", "N/A"),
                            f"Iteration {i} - Min Probability": selected_run.data.metrics.get(f"min_prob_iter_{i}", "N/A"),
                            f"Iteration {i} - Max Probability": selected_run.data.metrics.get(f"max_prob_iter_{i}", "N/A"),
                            f"Iteration {i} - Mean Probability": selected_run.data.metrics.get(f"mean_prob_iter_{i}", "N/A"),
                            f"Iteration {i} - Num Labeled Samples": selected_run.data.metrics.get(f"num_labeled_samples_iter_{i}", "N/A"),
                        }
                        st.json(iter_metrics)
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
    run_PseudoLabeling_app()   
