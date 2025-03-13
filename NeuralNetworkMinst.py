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

def run_NeuralNetwork_app():
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
    # Định nghĩa đường dẫn đến các file MNIST
    
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
    st.title("📸 MNIST Neural NetWork")
    tabs = st.tabs([
            "Thông tin",
            "Tập dữ liệu",
            " Phân chia tỉ lệ",
            "Huấn luyện mô hình",
            "Dự đoán",
            "Thông tin & Mlflow",
    ])
    tab_note, tab_info, tab_samples, tab_preprocess,tab_demo, tab_mlflow = tabs

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
            st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image1.png", caption="Cấu trúc Neural Network (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)

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
            st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image2.png", caption="Cấu trúc Neural Network có 2 hoặc nhiều lớp ẩn (Nguồn:https://byvn.net/m3Sf)", use_container_width=True)

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
            st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image3.png", caption="Biểu đồ hàm Sigmoid (Nguồn:https://byvn.net/qW4e)", use_container_width=True)

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
            st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image4.png", caption="Biểu đồ hàm Hyperbolic Tangent (Tanh) (Nguồn:https://byvn.net/qW4e)", use_container_width=True)

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
            st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image5.png", caption="Biểu đồ hàm ReLU (Rectified Linear Unit)(Nguồn:https://byvn.net/qW4e)", use_container_width=True)

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
            st.image(r"C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image6.png", caption="Biểu đồ hàm Softmax (Rectified Linear Unit)(Nguồn:https://byvn.net/yvvj)", use_container_width=True)

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

    with tab_samples:
        with st.expander("**Phân chia dữ liệu**", expanded=True):    
            if "train_images" in st.session_state:
                train_images = st.session_state.train_images
                train_labels = st.session_state.train_labels
                test_images = st.session_state.test_images
                test_labels = st.session_state.test_labels

                X = np.concatenate((train_images, test_images), axis=0)
                y = np.concatenate((train_labels, test_labels), axis=0)
                X = X.reshape(X.shape[0], -1)

                with mlflow.start_run():
                    test_size = st.slider("🔹 Chọn % tỷ lệ tập test", min_value=10, max_value=50, value=20, step=5, key="test_size") / 100
                    val_size = st.slider("🔹 Chọn % tỷ lệ tập validation (trong phần train)", min_value=10, max_value=50, value=20, step=5, key="val_size") / 100

                    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    val_size_adjusted = val_size / (1 - test_size)
                    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

                    st.session_state.X_train = X_train
                    st.session_state.X_val = X_val
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_val = y_val
                    st.session_state.y_test = y_test

                    total_samples = X.shape[0]
                    test_percent = (X_test.shape[0] / total_samples) * 100
                    val_percent = (X_val.shape[0] / total_samples) * 100
                    train_percent = (X_train.shape[0] / total_samples) * 100

                    st.write(f"📊 **Tỷ lệ phân chia**: Test={test_percent:.0f}%, Validation={val_percent:.0f}%, Train={train_percent:.0f}%")
                    st.write("✅ Dữ liệu đã được xử lý và chia tách.")
                    st.write(f"🔹 Kích thước tập huấn luyện: `{X_train.shape}`")
                    st.write(f"🔹 Kích thước tập validation: `{X_val.shape}`")
                    st.write(f"🔹 Kích thước tập kiểm tra: `{X_test.shape}`")
            else:
                st.error("🚨 Dữ liệu chưa được nạp. Hãy đảm bảo `train_images`, `train_labels` và `test_images` đã được tải trước khi chạy.")


    with tab_preprocess:
        with st.expander("**Huấn luyện mô hình Neural Network**", expanded=True):
            if "X_train" in st.session_state:
                # Lấy dữ liệu từ session_state
                X_train = st.session_state.X_train
                X_val = st.session_state.X_val
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_val = st.session_state.y_val
                y_test = st.session_state.y_test

                # Chuẩn hóa dữ liệu
                X_train = X_train / 255.0
                X_val = X_val / 255.0
                X_test = X_test / 255.0

                # Xác định số lớp và input shape
                num_classes = len(np.unique(y_train))
                input_shape = X_train.shape[1]

                # Cấu hình huấn luyện
                st.markdown("### Cấu hình huấn luyện")

                # Lựa chọn số lượng lớp ẩn
                num_hidden_layers = st.slider(
                    "🔹 Số lượng lớp ẩn",
                    min_value=1,
                    max_value=5,
                    value=3,
                    step=1,
                    key="num_hidden_layers"
                )
                st.write(f"**Số lớp ẩn được chọn:** {num_hidden_layers}")

                # Lựa chọn số nơ-ron cho mỗi lớp ẩn
                hidden_layer_neurons = []
                for i in range(num_hidden_layers):
                    neurons = st.number_input(
                        f"🔹 Số nơ-ron cho lớp ẩn {i+1}",
                        min_value=32,
                        max_value=1024,
                        value=512 if i == 0 else 256 if i == 1 else 128,
                        step=32,
                        key=f"neurons_layer_{i}"
                    )
                    hidden_layer_neurons.append(neurons)
                st.write(f"**Số nơ-ron cho các lớp ẩn:** {hidden_layer_neurons}")

                # Lựa chọn hàm kích hoạt
                activation_function = st.selectbox(
                    "🔹 Hàm kích hoạt cho các lớp ẩn",
                    options=['relu', 'sigmoid', 'tanh'],
                    index=0,
                    key="activation_function"
                )

                # Xây dựng mô hình động dựa trên số lớp và nơ-ron
                model = models.Sequential()
                model.add(layers.Input(shape=(input_shape,)))

                # Thêm các lớp ẩn động
                for neurons in hidden_layer_neurons:
                    model.add(layers.Dense(neurons, activation=activation_function))
                    model.add(layers.Dropout(0.2))

                # Thêm lớp đầu ra
                model.add(layers.Dense(num_classes, activation='softmax'))

                # Số epoch
                epochs = st.slider("🔹 Số epoch", min_value=5, max_value=50, value=10, step=5, key="epochs")

                # Batch size
                batch_size = st.selectbox("🔹 Batch size", options=[32, 64, 128, 256], index=0, key="batch_size")

                # Bộ tối ưu
                optimizer_choice = st.selectbox(
                    "🔹 Bộ tối ưu",
                    options=['adam', 'sgd', 'rmsprop', 'adagrad'],
                    index=0,
                    key="optimizer"
                )

                # Learning rate
                learning_rate = st.slider(
                    "🔹 Learning Rate (Tốc độ học)",
                    min_value=0.0001,
                    max_value=0.1,
                    value=0.001,
                    step=0.0001,
                    format="%.4f",
                    key="learning_rate"
                )
                st.write(f"**Learning Rate được chọn:** {learning_rate}")

                # Cấu hình optimizer với learning rate
                if optimizer_choice == "adam":
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                elif optimizer_choice == "sgd":
                    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
                elif optimizer_choice == "rmsprop":
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                elif optimizer_choice == "adagrad":
                    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

                # Biên dịch mô hình
                model.compile(optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

                # Nút để bắt đầu huấn luyện
                if st.button("🚀 Bắt đầu huấn luyện", key="train_button"):
                    with st.spinner("Đang huấn luyện mô hình..."):
                        with mlflow.start_run():
                            # Khởi tạo progress bar và trạng thái văn bản
                            progress_bar = st.progress(0)
                            status_text = st.empty()  # Tạo một placeholder để cập nhật trạng thái %

                            start_time = time.time()  # Bắt đầu đo thời gian tổng

                            # Callback để đo thời gian mỗi epoch và cập nhật trạng thái
                            class TimeHistory(tf.keras.callbacks.Callback):
                                def on_train_begin(self, logs={}):
                                    self.times = []
                                    # status_text.text("Huấn luyện: 0%")
                                    status_text.markdown(" **Huấn luyện**: 0%")
                                def on_epoch_begin(self, epoch, logs={}):
                                    self.epoch_start = time.time()

                                def on_epoch_end(self, epoch, logs={}):
                                    self.times.append(time.time() - self.epoch_start)
                                    progress = (epoch + 1) / epochs * 100
                                    progress_bar.progress(int(progress))
                                    # status_text.text(f" **Đang huấn luyện**: {int(progress)}%")  # Cập nhật trạng thái %
                                    status_text.markdown(f" **Đang huấn luyện**: {int(progress)}%")
                                def on_train_end(self, logs={}):
                                    status_text.markdown(" **Huấn luyện**: 100% (Hoàn thành)")
                            time_callback = TimeHistory()

                            # Huấn luyện mô hình
                            history = model.fit(X_train, y_train,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(X_val, y_val),
                                            verbose=1,
                                            callbacks=[time_callback])

                            # Thời gian tổng
                            total_time = time.time() - start_time
                            progress_bar.progress(100)

                            # Đánh giá trên tập test
                            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                            train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
                            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

                            # Tổng số tham số
                            total_params = model.count_params()

                            # Hiển thị kết quả
                            st.success("✅ Huấn luyện hoàn tất!")
                            st.write("#### ✅ **Thông tin mô hình và kết quả huấn luyện**")

                            # Kiến trúc mô hình
                            st.write("**1. Kiến trúc mô hình:**")
                            st.write(f" - Số lớp ẩn: {num_hidden_layers}")
                            st.write(f" - Số nơ-ron: {hidden_layer_neurons}")
                            st.write(f" - Hàm kích hoạt: {activation_function}")

                            # Số lượng tham số
                            st.write(f"**2. Số lượng tham số:** {total_params:,}")

                            # Thông tin huấn luyện
                            st.write("**3. Thông tin huấn luyện:**")
                            st.write(f"- Số epoch: {epochs}")
                            st.write(f"- Batch size: {batch_size}")
                            st.write(f"- Learning rate: {learning_rate}")
                            st.write(f"- Bộ tối ưu: {optimizer_choice}")

                            # Loss và Accuracy
                            st.write("**4. Kết quả Loss & Accuracy:**")
                            st.write(f"- **Validation Accuracy**: {val_accuracy:.4f}")
                            st.write(f"- **Test Accuracy**: {test_accuracy:.4f}")

                            # Thời gian huấn luyện
                            st.write("**5. Thời gian huấn luyện:**")
                            st.write(f"- Tổng thời gian: {total_time:.2f} giây")
                            st.write(f"- Thời gian trung bình mỗi epoch: {np.mean(time_callback.times):.2f} giây")
                            
                            
                            st.write("**6. Biểu đồ Kết quả Huấn luyện:**")
                            # Tạo biểu đồ Loss
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(history.history['loss'], label='Training Loss', marker='o', linestyle='-')
                            ax.plot(history.history['val_loss'], label='Validation Loss', marker='s', linestyle='--')
                            ax.set_xlabel("Epochs")
                            ax.set_ylabel("Loss")
                            ax.set_title("Training & Validation Loss")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                            st.markdown("""
                            **Giải thích biểu đồ Loss:**
                            - **Train Loss (Mất mát huấn luyện):** Đại diện cho sai số giữa dự đoán và nhãn thực tế trên tập huấn luyện. Giá trị giảm dần qua các epoch cho thấy mô hình đang học tốt hơn.
                            - **Val Loss (Mất mát validation):** Đo lường sai số trên tập validation, giúp đánh giá khả năng tổng quát hóa. Nếu Val Loss ổn định hoặc giảm chậm, mô hình không bị overfitting.
                            - Hai đường này nên có xu hướng tương tự; nếu Val Loss tăng trong khi Train Loss giảm, đó là dấu hiệu của overfitting.
                            """)
                            st.markdown("---")
                            # Tạo biểu đồ Accuracy
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(history.history['accuracy'], label='Training Accuracy', marker='o', linestyle='-')
                            ax.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linestyle='--')
                            ax.set_xlabel("Epochs")
                            ax.set_ylabel("Accuracy")
                            ax.set_title("Training & Validation Accuracy")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                            st.markdown("""
                            **Giải thích biểu đồ Accuracy:**
                            - **Train Accuracy (Độ chính xác huấn luyện):** Tỷ lệ dự đoán đúng trên tập huấn luyện, thường tăng qua các epoch khi mô hình học.
                            - **Val Accuracy (Độ chính xác validation):** Tỷ lệ dự đoán đúng trên tập validation, phản ánh khả năng tổng quát hóa. Giá trị cao và ổn định cho thấy mô hình hoạt động tốt trên dữ liệu mới.
                            - Sự khác biệt giữa Train Accuracy và Val Accuracy không quá lớn là dấu hiệu của một mô hình cân bằng.
                            """)

                            # Ghi log với MLflow
                            mlflow.log_param("epochs", epochs)
                            mlflow.log_param("batch_size", batch_size)
                            mlflow.log_param("optimizer", optimizer_choice)
                            mlflow.log_param("learning_rate", learning_rate)
                            mlflow.log_param("activation_function", activation_function)
                            mlflow.log_param("num_hidden_layers", num_hidden_layers)
                            mlflow.log_param("hidden_layer_neurons", hidden_layer_neurons)
                            mlflow.log_param("num_classes", num_classes)
                            mlflow.log_param("input_shape", input_shape)
                            mlflow.log_param("total_params", total_params)
                            mlflow.log_metric("train_accuracy", train_accuracy)
                            mlflow.log_metric("val_accuracy", val_accuracy)
                            mlflow.log_metric("test_accuracy", test_accuracy)
                            mlflow.log_metric("train_loss", train_loss)
                            mlflow.log_metric("val_loss", val_loss)
                            mlflow.log_metric("test_loss", test_loss)
                            mlflow.log_metric("total_training_time", total_time)

                            st.session_state['trained_model'] = model
                            st.session_state['history'] = history

            else:
                st.error("🚨 Vui lòng phân chia dữ liệu ở tab 'Phân chia dữ liệu' trước khi huấn luyện mô hình.")


                
    with tab_demo:
        st.header("Dự đoán số viết tay")
        st.write("Chọn cách nhập liệu: tải lên hình ảnh hoặc vẽ trực tiếp.")

        if 'trained_model' not in st.session_state:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước trong tab 'Huấn luyện'!")
        else:
            model = st.session_state['trained_model']

            input_method = st.selectbox("Chọn phương thức nhập liệu", ["Vẽ trực tiếp", "Tải ảnh lên"])

            if input_method == "Vẽ trực tiếp":
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas"
                )

                # Chỉ dự đoán khi người dùng nhấn nút
                if canvas_result.image_data is not None:
                    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                    image = image.convert('L')
                    image = image.resize((28, 28))
                    st.image(image, caption="Hình ảnh bạn vẽ (resize 28x28)", width=100)

                    if st.button("Dự đoán", key="predict_button"):
                        image_array = np.array(image, dtype=np.float32) / 255.0
                        image_array = image_array.reshape(1, 784)

                        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
                        prediction = model.predict(image_tensor, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]

                        st.write(f"**Dự đoán:** {predicted_class}")
                        st.write(f"**Xác suất:** {confidence:.4f}")

                    if st.button("Xóa và vẽ lại", key="clear_button"):
                        st.session_state.pop("canvas")
                        st.rerun()

            elif input_method == "Tải ảnh lên":
                uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('L')
                    image = image.resize((28, 28))
                    st.image(image, caption="Hình ảnh đầu vào", width=100)

                    if st.button("Dự đoán", key="predict_upload_button"):
                        image_array = np.array(image, dtype=np.float32) / 255.0
                        image_array = image_array.reshape(1, 784)

                        image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
                        prediction = model.predict(image_tensor, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]

                        st.write(f"**Dự đoán:** {predicted_class} (Xác suất: {confidence:.4f})")


    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "NeuralNetworkExperiment"  # Đổi tên experiment cho phù hợp với Neural Network

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

                # Hiển thị các tham số đã log
                st.markdown("### Tham số đã log")
                st.json({
                    "epochs": selected_run.data.params.get("epochs", "N/A"),
                    "batch_size": selected_run.data.params.get("batch_size", "N/A"),
                    "optimizer": selected_run.data.params.get("optimizer", "N/A"),
                    "learning_rate": selected_run.data.params.get("learning_rate", "N/A"),
                    "activation_function": selected_run.data.params.get("activation_function", "N/A"),
                    "num_hidden_layers": selected_run.data.params.get("num_hidden_layers", "N/A"),
                    "hidden_layer_neurons": selected_run.data.params.get("hidden_layer_neurons", "N/A"),
                    "num_classes": selected_run.data.params.get("num_classes", "N/A"),
                    "input_shape": selected_run.data.params.get("input_shape", "N/A"),
                    "total_params": selected_run.data.params.get("total_params", "N/A")
                })

                # Hiển thị các chỉ số đã log
                st.markdown("### Chỉ số đã log")
                metrics = {
                    "Train Accuracy": selected_run.data.metrics.get("train_accuracy", "N/A"),
                    "Validation Accuracy": selected_run.data.metrics.get("val_accuracy", "N/A"),
                    "Test Accuracy": selected_run.data.metrics.get("test_accuracy", "N/A"),
                    "Train Loss": selected_run.data.metrics.get("train_loss", "N/A"),
                    "Validation Loss": selected_run.data.metrics.get("val_loss", "N/A"),
                    "Test Loss": selected_run.data.metrics.get("test_loss", "N/A"),
                    "Total Training Time (s)": selected_run.data.metrics.get("total_training_time", "N/A")
                }
                st.json(metrics)

                # 5) Nút bấm mở MLflow UI
                st.subheader("Truy cập MLflow UI")
                mlflow_url = "https://dagshub.com/Dung2204/HMVPython.mlflow"  # Thay bằng URL MLflow của bạn nếu khác
                if st.button("Mở MLflow UI"):
                    st.markdown(f'**[Click để mở MLflow UI]({mlflow_url})**')
            else:
                st.info("Chưa có Run nào được log. Vui lòng huấn luyện mô hình trước.")

        except Exception as e:
            st.error(f"Không thể kết nối với MLflow: {e}")

if __name__ == "__main__":
    run_NeuralNetwork_app()
