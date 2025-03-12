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
    st.title("📸 MNIST PCA_T-SNE")
    tabs = st.tabs([
            "Thông tin",
            "Tập dữ liệu",
            "Chọn mẫu",
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
                    - Các nơ-ron được kết nối bằng **trọng số** (weights), và mô hình học bằng cách điều chỉnh các trọng số này để giảm thiểu sai số dự đoán thông qua quá trình huấn luyện.
                    - Neural Network đặc biệt mạnh trong việc xử lý các bài toán phi tuyến tính và học các đặc trưng phức tạp từ dữ liệu.
                    """
            )
            st.markdown("---")
            st.markdown("### Các bước huấn luyện Neural Network")
            st.write("1. **Khởi tạo mô hình**: Xác định số lớp ẩn, số nơ-ron trong mỗi lớp, và hàm kích hoạt.")
            st.write("2. **Chuẩn hóa dữ liệu**: Đưa dữ liệu về dạng chuẩn để tăng hiệu quả huấn luyện.")
            st.write("3. **Lan truyền xuôi (Forward Propagation)**: Tính toán đầu ra từ đầu vào qua các lớp.")
            st.write("4. **Lan truyền ngược (Backpropagation)**: Cập nhật trọng số dựa trên hàm mất mát.")
            st.write("5. **Dự đoán**: Sử dụng mô hình đã huấn luyện để dự đoán trên dữ liệu mới.")
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

                # Xây dựng mô hình
                model = models.Sequential([
                    layers.Input(shape=(input_shape,)),
                    layers.Dense(512, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Dense(256, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(num_classes, activation='softmax')
                ])

                # Cấu hình huấn luyện
                epochs = st.slider("🔹 Số epoch", min_value=5, max_value=50, value=10, step=5, key="epochs")
                batch_size = st.selectbox("🔹 Batch size", options=[32, 64, 128, 256], index=0, key="batch_size")
                optimizer = st.selectbox("🔹 Bộ tối ưu", options=['adam'], index=0, key="optimizer")
                if optimizer == "adam":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    if st.toggle("Hiển thị thông tin bộ tối ưu: **adam**"):
                        st.write("**Adam (Adaptive Moment Estimation)**: Thuật toán tối ưu hóa tự động, điều chỉnh tốc độ học động, giúp hội tụ nhanh và ổn định trên nhiều bài toán, đặc biệt hiệu quả với mạng sâu và dữ liệu có nhiễu (ví dụ: MNIST).")


                # Biên dịch mô hình
                model.compile(optimizer=optimizer,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])


                # Nút để bắt đầu huấn luyện
                if st.button("🚀 Bắt đầu huấn luyện", key="train_button"):
                    with st.spinner("Đang huấn luyện mô hình..."):
                        with mlflow.start_run():
                            # Khởi tạo progress bar
                            progress_bar = st.progress(0)
                            history = None

                            # Huấn luyện mô hình với callback để cập nhật progress bar
                            class ProgressCallback(tf.keras.callbacks.Callback):
                                def on_epoch_end(self, epoch, logs=None):
                                    progress = (epoch + 1) / epochs * 100
                                    progress_bar.progress(int(progress))

                            # Huấn luyện mô hình
                            history = model.fit(X_train, y_train,
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(X_val, y_val),
                                            verbose=1,
                                            callbacks=[ProgressCallback()])

                            # Hoàn thành progress bar
                            progress_bar.progress(100)

                            # Ghi log với MLflow
                            mlflow.log_param("epochs", epochs)
                            mlflow.log_param("batch_size", batch_size)
                            mlflow.log_param("optimizer", optimizer)
                            mlflow.log_param("num_classes", num_classes)
                            mlflow.log_param("input_shape", input_shape)
                            mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
                            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
                            mlflow.log_metric("test_accuracy", test_accuracy)
                            mlflow.log_metric("test_loss", test_loss)
                            mlflow.log_metric("final_train_loss", history.history['loss'][-1])
                            mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

                            # Đánh giá trên tập test
                            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
                            mlflow.log_metric("test_accuracy", test_accuracy)
                            st.session_state['trained_model'] = model
                            st.session_state['history'] = history
                            # Hiển thị kết quả
                            st.success("✅ Huấn luyện hoàn tất!")
                            st.write(f"#### ✅ **Kết quả huấn luyện**")
                            st.write(f"📈 **Độ chính xác trên tập test**: {test_accuracy:.4f}")
                            st.write("📊 **Trọng số và Bias đã tối ưu hóa:**")
                            for layer in model.layers:
                                if isinstance(layer, layers.Dense):
                                    weights = layer.get_weights()[0]
                                    biases = layer.get_weights()[1]
                                    st.write(f"Layer {layer.name}:")
                                    st.write(f"  - Trọng số shape: {weights.shape}")
                                    st.write(f"  - Bias shape: {biases.shape}")
                            # Giải thích về trọng số và bias đã tối ưu hóa
                            st.markdown("""
                            **Giải thích:**
                            - **Trọng số (Weights):** Là các giá trị được điều chỉnh trong quá trình huấn luyện để mô hình học cách ánh xạ từ đầu vào (ảnh 28x28 pixel) đến đầu ra (chữ số 0-9). Shape của trọng số thể hiện kích thước của ma trận kết nối giữa các lớp (ví dụ: (784, 512) là kết nối từ 784 đầu vào đến 512 nơ-ron).
                            - **Bias:** Là các giá trị bù đắp cho mỗi nơ-ron, giúp mô hình linh hoạt hơn trong việc điều chỉnh đầu ra. Shape của bias tương ứng với số nơ-ron trong mỗi lớp (ví dụ: (512,) cho 512 nơ-ron).
                            - Các giá trị này được tối ưu hóa thông qua thuật toán lan truyền ngược (backpropagation) để giảm thiểu hàm mất mát.
                            """)
                            st.markdown("---")
                            st.markdown("#### ✅**Biểu đồ Accuracy và Loss**")
                            # Vẽ biểu đồ (xóa các giá trị số)
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Biểu đồ Loss
                            ax1.plot(history.history['loss'], label='Train Loss', color='blue')
                            ax1.plot(history.history['val_loss'], label='Val Loss', color='orange')
                            ax1.set_title('Loss')
                            ax1.set_xlabel('Epoch')
                            ax1.set_ylabel('Loss')
                            ax1.legend()
                            # Xóa các giá trị số trên biểu đồ Loss
                            # for i, (train_loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss'])):
                            #     ax1.text(i, train_loss, f'{train_loss:.3f}', ha='center', va='bottom')
                            #     ax1.text(i, val_loss, f'{val_loss:.3f}', ha='center', va='top')

                            
                            # Biểu đồ Accuracy
                            ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
                            ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
                            ax2.set_title('Accuracy')
                            ax2.set_xlabel('Epoch')
                            ax2.set_ylabel('Accuracy')
                            ax2.legend()
                            # Xóa các giá trị số trên biểu đồ Accuracy
                            # for i, (train_acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
                            #     ax2.text(i, train_acc, f'{train_acc:.3f}', ha='center', va='bottom')
                            #     ax2.text(i, val_acc, f'{val_acc:.3f}', ha='center', va='top')

                            # Giải thích biểu đồ Accuracy
                            st.pyplot(fig)
                            st.markdown("""
                            **Giải thích biểu đồ Loss:**
                            - **Train Loss (Mất mát huấn luyện):** Đại diện cho sai số giữa dự đoán và nhãn thực tế trên tập huấn luyện. Giá trị giảm dần qua các epoch cho thấy mô hình đang học tốt hơn.
                            - **Val Loss (Mất mát validation):** Đo lường sai số trên tập validation, giúp đánh giá khả năng tổng quát hóa. Nếu Val Loss ổn định hoặc giảm chậm, mô hình không bị overfitting.
                            - Hai đường này nên có xu hướng tương tự; nếu Val Loss tăng trong khi Train Loss giảm, đó là dấu hiệu của overfitting.
                            """)
                            st.markdown("""
                            **Giải thích biểu đồ Accuracy:**
                            - **Train Accuracy (Độ chính xác huấn luyện):** Tỷ lệ dự đoán đúng trên tập huấn luyện, thường tăng qua các epoch khi mô hình học.
                            - **Val Accuracy (Độ chính xác validation):** Tỷ lệ dự đoán đúng trên tập validation, phản ánh khả năng tổng quát hóa. Giá trị cao và ổn định cho thấy mô hình hoạt động tốt trên dữ liệu mới.
                            - Sự khác biệt giữa Train Accuracy và Val Accuracy không quá lớn là dấu hiệu của một mô hình cân bằng.
                            """)

                            # st.pyplot(fig)
            else:
                st.error("🚨 Vui lòng phân chia dữ liệu ở tab 'Phân chia dữ liệu' trước khi huấn luyện mô hình.")
    # Tab dự đoán
    with tab_demo:
        st.header("Dự đoán số viết tay")
        st.write("Tải lên một hình ảnh số viết tay (28x28 pixel, grayscale) hoặc vẽ trực tiếp để dự đoán.")

        # Kiểm tra xem mô hình đã được huấn luyện chưa
        if 'trained_model' not in st.session_state:
            st.warning("⚠️ Vui lòng huấn luyện mô hình trước trong tab 'Huấn luyện'!")
        else:
            model = st.session_state['trained_model']

            # Tùy chọn tải lên hình ảnh
            uploaded_file = st.file_uploader("Tải lên hình ảnh", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                # Đọc và xử lý hình ảnh
                image = Image.open(uploaded_file).convert('L')  # Chuyển thành grayscale
                image = image.resize((28, 28))  # Đảm bảo kích thước 28x28
                image_array = np.array(image) / 255.0  # Chuẩn hóa giống dữ liệu huấn luyện
                image_array = image_array.reshape(1, 784)  # Reshape thành (1, 784)

                # Dự đoán
                prediction = model.predict(image_array)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]

                # Hiển thị hình ảnh và kết quả
                st.image(image, caption="Hình ảnh đầu vào", width=100)
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

                st.markdown("### Tham số đã log")
                st.json({
                    "epochs": selected_run.data.params.get("epochs", "N/A"),
                    "batch_size": selected_run.data.params.get("batch_size", "N/A"),
                    "optimizer": selected_run.data.params.get("optimizer", "N/A"),
                    "num_classes": selected_run.data.params.get("num_classes", "N/A"),
                    "input_shape": selected_run.data.params.get("input_shape", "N/A")
                })

                st.markdown("### Chỉ số đã log")
                metrics = {
                    "Train Accuracy": selected_run.data.metrics.get("train_accuracy", "N/A"),
                    "Validation Accuracy": selected_run.data.metrics.get("val_accuracy", "N/A"),
                    "Test Accuracy": selected_run.data.metrics.get("test_accuracy", "N/A"),
                    "Test Loss": selected_run.data.metrics.get("test_loss", "N/A"),
                    "Final Train Loss": selected_run.data.metrics.get("final_train_loss", "N/A"),
                    "Final Validation Loss": selected_run.data.metrics.get("final_val_loss", "N/A")
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
