import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import random
import struct
from sklearn.datasets import load_iris
import mlflow
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from collections import Counter
from mlflow.tracking import MlflowClient

def run_ClassificationMinst_app():
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
    dataset_path = os.path.dirname(os.path.abspath(__file__)) 
    train_images_path = os.path.join(dataset_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(dataset_path, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(dataset_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(dataset_path, "t10k-labels.idx1-ubyte")

    # Tải dữ liệu
    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    # Giao diện Streamlit
    st.title("📸 Phân loại ảnh MNIST với Streamlit")
    tabs = st.tabs([
        "Thông tin dữ liệuliệu",
        "Xử lí dữ liệu",
        "Thông tin",
        "Huấn luyện mô hình",
        "Đánh giá mô hình",
        "Demo dự đoán",
        "Thông tin & Mlflow",
    ])
    # tab_info, tab_load, tab_preprocess, tab_split,  tab_demo, tab_log_info = tabs
    tab_info, tab_load,tab_note, tab_preprocess, tab_split,  tab_demo ,tab_mlflow= tabs

    # with st.expander("🖼️ Dữ liệu ban đầu", expanded=True):
    with tab_info:
        with st.expander("**Thông tin dữ liệu**", expanded=True):
            st.markdown(
                '''
                **MNIST** là phiên bản được chỉnh sửa từ bộ dữ liệu NIST gốc của Viện Tiêu chuẩn và Công nghệ Quốc gia Hoa Kỳ.  
                Bộ dữ liệu ban đầu gồm các chữ số viết tay từ nhân viên bưu điện và học sinh trung học.  

                Các nhà nghiên cứu **Yann LeCun, Corinna Cortes, và Christopher Burges** đã xử lý, chuẩn hóa và chuyển đổi bộ dữ liệu này thành **MNIST** để dễ dàng sử dụng hơn cho các bài toán nhận dạng chữ số viết tay.
                '''
            )
            image = Image.open(r'C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image.png')

            # Gắn ảnh vào Streamlit và chỉnh kích thước
            st.image(image, caption='Mô tả ảnh', width=600) 
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

            # # Hiển thị biểu đồ cột
            # st.subheader("📊 Biểu đồ số lượng mẫu của từng chữ số")
            # st.bar_chart(label_counts)

            # Hiển thị bảng dữ liệu dưới biểu đồ
            st.subheader("📋 Số lượng mẫu cho từng chữ số")
            df_counts = pd.DataFrame({"Chữ số": label_counts.index, "Số lượng mẫu": label_counts.values})
            st.dataframe(df_counts)


            st.subheader("Chọn ngẫu nhiên 10 ảnh từ tập huấn luyện để hiển thị")
            num_images = 10
            random_indices = random.sample(range(len(train_images)), num_images)
            fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

            for ax, idx in zip(axes, random_indices):
                ax.imshow(train_images[idx], cmap='gray')
                ax.axis("off")
                ax.set_title(f"Label: {train_labels[idx]}")

            st.pyplot(fig)
        with st.expander("**Kiểm tra hình dạng của tập dữ liệu**", expanded=True):    
            # Kiểm tra hình dạng của tập dữ liệu
            st.write("🔍 Hình dạng tập huấn luyện:", train_images.shape)
            st.write("🔍 Hình dạng tập kiểm tra:", test_images.shape)
            st.write("**Chuẩn hóa dữ liệu (đưa giá trị pixel về khoảng 0-1)**")
            # Chuẩn hóa dữ liệu
            train_images = train_images.astype("float32") / 255.0
            test_images = test_images.astype("float32") / 255.0

            # Hiển thị thông báo sau khi chuẩn hóa
            st.success("✅ Dữ liệu đã được chuẩn hóa về khoảng [0,1].") 

            # Hiển thị bảng dữ liệu đã chuẩn hóa (dạng số)
            num_samples = 5  # Số lượng mẫu hiển thị
            df_normalized = pd.DataFrame(train_images[:num_samples].reshape(num_samples, -1))  

            
            sample_size = 10_000  
            pixel_sample = np.random.choice(train_images.flatten(), sample_size, replace=False)
            if "train_images" not in st.session_state:
                st.session_state.train_images = train_images
                st.session_state.train_labels = train_labels
                st.session_state.test_images = test_images
                st.session_state.test_labels = test_labels


    # with st.expander("🖼️ XỬ LÝ DỮ LIỆU", expanded=True):
    with tab_load:
        with st.expander("**Phân chia dữ liệu**", expanded=True):    

            # Kiểm tra nếu dữ liệu đã được load
            if "train_images" in st.session_state:
                # Lấy dữ liệu từ session_state
                train_images = st.session_state.train_images
                train_labels = st.session_state.train_labels
                test_images = st.session_state.test_images
                test_labels = st.session_state.test_labels

                # Chuyển đổi dữ liệu thành vector 1 chiều
                X_train = train_images.reshape(train_images.shape[0], -1)
                X_test = test_images.reshape(test_images.shape[0], -1)
                y_test = test_labels
                with mlflow.start_run():

                    # Cho phép người dùng chọn tỷ lệ validation và test
                    val_size = st.slider("🔹 Chọn tỷ lệ tập validation (%)", min_value=10, max_value=50, value=20, step=5) / 100
                    test_size = st.slider("🔹 Chọn tỷ lệ tập test (%)", min_value=10, max_value=50, value=20, step=5) / 100

                    # Chia tập train thành train/validation theo tỷ lệ đã chọn
                    X_train, X_val, y_train, y_val = train_test_split(X_train, train_labels, test_size=val_size, random_state=42)
                    
                    # Chia tập test thành test/validation theo tỷ lệ đã chọn
                    # Chúng ta có thể chia tập test thành test và validation nếu tỷ lệ `test_size` đã chọn
                    X_test, X_val, y_test, y_val = train_test_split(X_test, test_labels, test_size=test_size, random_state=42)

                st.write("✅ Dữ liệu đã được xử lý và chia tách.")
                st.write(f"🔹 Kích thước tập huấn luyện: `{X_train.shape}`")
                st.write(f"🔹 Kích thước tập validation: `{X_val.shape}`")
                st.write(f"🔹 Kích thước tập kiểm tra: `{X_test.shape}`")

                # Biểu đồ phân phối nhãn dữ liệu
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=list(Counter(y_train).keys()), y=list(Counter(y_train).values()), palette="Blues", ax=ax)
                ax.set_title("Phân phối nhãn trong tập huấn luyện")
                ax.set_xlabel("Nhãn")
                ax.set_ylabel("Số lượng")
                st.pyplot(fig)

                st.markdown(
                """
                ### 📊 Mô tả biểu đồ  
                Biểu đồ cột hiển thị **phân phối nhãn** trong tập huấn luyện.  
                - **Trục hoành (x-axis):** Biểu diễn các nhãn (labels) từ `0` đến `9`.  
                - **Trục tung (y-axis):** Thể hiện **số lượng mẫu dữ liệu** tương ứng với mỗi nhãn.  
                """
                )
            else:
                st.error("🚨 Dữ liệu chưa được nạp. Hãy đảm bảo `train_images`, `train_labels` và `test_images` đã được tải trước khi chạy.")


    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):    
            # Assume model_option1 is selected from somewhere in the app
            model_option1 = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])
            if model_option1 == "Decision Tree":
                st.markdown("""
                ### Decision Tree (Cây quyết định)

                **Khái niệm**:  
                Cây quyết định là một thuật toán học máy dạng cây dùng để phân loại hoặc dự đoán giá trị liên tục. Mỗi nút trong cây đại diện cho một điều kiện kiểm tra (feature), mỗi nhánh là kết quả của kiểm tra đó, và mỗi lá cây chứa nhãn của lớp hoặc giá trị dự đoán.
                
                            
                **Các tham số của Decision Tree:**
                - **Gini Index**:  
                Gini Index là một chỉ số đo độ không thuần nhất của một tập hợp. Nó được sử dụng trong thuật toán cây quyết định để chọn đặc trưng phân chia dữ liệu sao cho thu được sự phân chia tốt nhất. Gini Index có giá trị từ 0 đến 1, trong đó 0 có nghĩa là các đối tượng trong tập hợp đều thuộc cùng một lớp.
                
                - **Entropy**:  
                Entropy là một chỉ số đo độ không chắc chắn (mức độ hỗn loạn) của một tập hợp dữ liệu. Được dùng trong các thuật toán như ID3 hoặc C4.5 để tính toán mức độ thuần nhất của các phân nhóm dữ liệu.
                
    
                **Biểu đồ cây quyết định**:  
                Một cây quyết định có thể được biểu diễn dưới dạng cấu trúc cây, với các nhánh nối các nút kiểm tra đến các nhánh con. Mỗi nút kiểm tra sẽ phân chia dữ liệu thành các nhánh con dựa vào điều kiện (ví dụ: giá trị của một đặc trưng).
                """)
                # Tải bộ dữ liệu Iris từ sklearn
                iris = load_iris()
                X, y = iris.data, iris.target

                # Huấn luyện mô hình cây quyết định
                clf = DecisionTreeClassifier(max_depth=3, random_state=42)
                clf.fit(X, y)

                # Vẽ biểu đồ cây quyết định
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, ax=ax)

                # Hiển thị biểu đồ trên Streamlit
                st.pyplot(fig)
                st.markdown("""
                 📝 Giải thích về cây quyết định ví dụ trên:
                - **Các nút (Nodes)**: Mỗi hình chữ nhật là một nút quyết định dựa trên một đặc trưng của dữ liệu.
                - **Gini/Entropy**: Độ thuần khiết của dữ liệu tại mỗi nút.
                - **Samples**: Số lượng mẫu tại mỗi nút.
                - **Class**: Nhãn được dự đoán tại nút đó.

                Biểu đồ trên thể hiện cách mô hình phân loại dữ liệu dựa vào đặc trưng của hoa Iris.
                """)


            elif model_option1 == "SVM":
                st.markdown("""
                ### Support Vector Machine (SVM)

                **Khái niệm**:  
                Support Vector Machine (SVM) là một thuật toán học máy dùng để phân loại hoặc hồi quy, đặc biệt nổi bật trong phân loại nhị phân. Mục tiêu của SVM là tìm một siêu phẳng phân chia các lớp sao cho khoảng cách giữa siêu phẳng và các điểm gần nhất của các lớp (gọi là support vectors) là lớn nhất.

                **Các tham số của SVM:**
                - **Linear**:  
                Trong SVM với kernel "linear", siêu phẳng phân chia các điểm dữ liệu bằng một đường thẳng trong không gian hai chiều (hoặc siêu phẳng trong không gian đa chiều). Phù hợp với dữ liệu có thể phân chia trực tiếp bằng một đường thẳng.
                - **Poly**:  
                Kernel polynomial (poly) sử dụng đa thức để tạo ra các ranh giới phân chia phi tuyến tính. Đây là một phương pháp có thể phân chia các lớp không thể phân chia được bằng đường thẳng.

                - **RBF (Radial Basis Function)**:  
                Kernel RBF là một kernel phổ biến giúp tạo ra một không gian cao hơn (không gian đặc trưng) trong đó dữ liệu có thể phân chia tốt hơn. Kernel này giúp SVM phân loại các dữ liệu không tuyến tính.

                - **Sigmoid**:  
                Kernel Sigmoid mô phỏng một hàm kích hoạt trong mạng nơ-ron, sử dụng hàm hyperbolic tangent (tanh). Kernel này có thể phân chia dữ liệu một cách phi tuyến.


                **Biểu đồ của SVM**:  
                Trong SVM, biểu đồ có thể bao gồm các điểm dữ liệu phân tán trong không gian đa chiều, với siêu phẳng phân chia giữa hai lớp. Các support vector nằm gần siêu phẳng và có ảnh hưởng trực tiếp đến việc xác định siêu phẳng.
                """)
                X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])  # 6 điểm (x, y)
                y = np.array([0, 0, 0, 1, 1, 1])  # Nhãn (0 hoặc 1)

                # Huấn luyện mô hình SVM
                model = SVC(kernel="linear")
                model.fit(X, y)

                # Tạo biểu đồ
                fig, ax = plt.subplots()
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

                # Vẽ đường phân chia
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # Tạo lưới điểm để vẽ ranh giới
                xx = np.linspace(xlim[0], xlim[1], 30)
                yy = np.linspace(ylim[0], ylim[1], 30)
                YY, XX = np.meshgrid(yy, xx)
                xy = np.vstack([XX.ravel(), YY.ravel()]).T
                Z = model.decision_function(xy).reshape(XX.shape)

                # Vẽ ranh giới quyết định của SVM
                ax.contour(XX, YY, Z, colors='k', levels=[0], linestyles=['--'])

                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                # Hiển thị trên Streamlit
                st.pyplot(fig)
                st.markdown("""
                📝 Giải thích về biểu đồ SVM ví dụ trên:
                - Các điểm tròn đại diện cho dữ liệu, với **màu sắc khác nhau** cho hai lớp.
                - Đường **đứt nét** là **ranh giới quyết định** của SVM.
                - **Điểm bên trái** thuộc lớp `0`, **điểm bên phải** thuộc lớp `1`.
                """)

    # 3️⃣ HUẤN LUYỆN MÔ HÌNH
    with tab_preprocess:
        with st.expander("**Huấn luyện mô hình**", expanded=True):
            # Lựa chọn mô hình
            model_option = st.radio("🔹 Chọn mô hình huấn luyện:", ("Decision Tree", "SVM"))
            if model_option == "Decision Tree":
                st.subheader("🌳 Decision Tree Classifier")
                        
                        # Lựa chọn tham số cho Decision Tree
                criterion = st.selectbox("Chọn tiêu chí phân nhánh:", ["gini", "entropy"])
                max_depth = st.slider("Chọn độ sâu tối đa của cây:", min_value=1, max_value=20, value=5)

                if st.button("🚀 Huấn luyện mô hình"):
                    with mlflow.start_run():
                        dt_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
                        dt_model.fit(X_train, y_train)
                        y_val_pred_dt = dt_model.predict(X_val)
                        accuracy_dt = accuracy_score(y_val, y_val_pred_dt)

                        mlflow.log_param("model_type", "Decision Tree")
                        mlflow.log_param("criterion", criterion)
                        mlflow.log_param("max_depth", max_depth)
                        mlflow.log_metric("accuracy", accuracy_dt)
                        mlflow.sklearn.log_model(dt_model, "decision_tree_model")

                        st.session_state["selected_model_type"] = "Decision Tree"
                        st.session_state["trained_model"] = dt_model 
                        st.session_state["X_train"] = X_train   

                        st.write(f"✅ **Độ chính xác trên tập validation:** `{accuracy_dt:.4f}`")

                                # Hiển thị kết quả bằng biểu đồ
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=["Decision Tree"], y=[accuracy_dt], palette="Blues", ax=ax)
                        ax.set_ylim(0, 1)
                        ax.set_title("Độ chính xác của Decision Tree")
                        ax.set_ylabel("Accuracy")
                        st.pyplot(fig)
                    mlflow.end_run()

            elif model_option == "SVM":
                st.subheader("🌀 Support Vector Machine (SVM)")
                            
                            # Lựa chọn tham số cho SVM
                kernel = st.selectbox("Chọn kernel:", ["linear", "poly", "rbf", "sigmoid"])
                C = st.slider("Chọn giá trị C (điều chỉnh mức độ regularization):", min_value=0.1, max_value=10.0, value=1.0)

                if st.button("🚀 Huấn luyện mô hình"):
                    with mlflow.start_run(): 
                        svm_model = SVC(kernel=kernel, C=C, random_state=42)
                        svm_model.fit(X_train, y_train)
                        y_val_pred_svm = svm_model.predict(X_val)
                        accuracy_svm = accuracy_score(y_val, y_val_pred_svm)

                        mlflow.log_param("model_type", "SVM")
                        mlflow.log_param("kernel", kernel)
                        mlflow.log_param("C_value", C)
                        mlflow.log_metric("accuracy", accuracy_svm)
                        mlflow.sklearn.log_model(svm_model, "svm_model")

                        st.session_state["selected_model_type"] = "SVM"
                        st.session_state["trained_model"] = svm_model  
                        st.session_state["X_train"] = X_train

                        st.write(f"✅ **Độ chính xác trên tập validation:** `{accuracy_svm:.4f}`")

                                    # Hiển thị kết quả bằng biểu đồ
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=["SVM"], y=[accuracy_svm], palette="Reds", ax=ax)
                        ax.set_ylim(0, 1)
                        ax.set_title("Độ chính xác của SVM")
                        ax.set_ylabel("Accuracy")
                        st.pyplot(fig)
                    mlflow.end_run()

    # 3️⃣ ĐÁNH GIÁ MÔ HÌNH
    with tab_split:
        with st.expander("**Đánh giá mô hình**", expanded=True):
            st.write("**Đánh giá mô hình bằng Confusion Matrix**")
            # Kiểm tra xem mô hình nào đã được huấn luyện
            if "selected_model_type" not in st.session_state or "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện ít nhất một mô hình trước khi đánh giá.")
            else:
                # Lấy mô hình đã được huấn luyện
                best_model_name = st.session_state.selected_model_type  
                best_model = st.session_state.trained_model  

                st.write(f"🔹Mô hình được chọn để đánh giá: `{best_model_name}`")
                    # Hiển thị các tham số đã sử dụng trong quá trình huấn luyện
                with mlflow.start_run():    
                    if best_model_name == "Decision Tree":
                        criterion = st.session_state.get("dt_criterion", "gini")
                        max_depth = st.session_state.get("dt_max_depth", None)
                        st.write("🔹 Tham số mô hình:")
                        st.write(f"- **Tiêu chí phân nhánh**: `{criterion}`")
                        st.write(f"- **Độ sâu tối đa**: `{max_depth}`")

                    elif best_model_name == "SVM":
                        kernel = st.session_state.get("svm_kernel", "linear")
                        C = st.session_state.get("svm_C", 1.0)
                        st.write("🔹 **Tham số mô hình:**")
                        st.write(f"- Kernel: `{kernel}`")
                        st.write(f"- C (Regularization): `{C}`")

                        # Dự đoán trên tập kiểm tra
                    y_test_pred = best_model.predict(X_test)
                    st.session_state["y_test_pred"] = y_test_pred

                        # Confusion Matrix
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, cmap="Blues", ax=ax)
                    ax.set_title(f"Confusion Matrix của {best_model_name} trên tập kiểm tra")

                    st.pyplot(fig)

                        # Hiển thị độ chính xác
                    test_accuracy = accuracy_score(y_test, y_test_pred)
                    st.session_state["test_accuracy"] = test_accuracy
                    st.write(f"✅ **Độ chính xác trên tập kiểm tra:** `{test_accuracy:.4f}`")
                    mlflow.log_param("selected_model", best_model_name)
                    mlflow.log_metric("test_accuracy", test_accuracy)  # Log accuracy trên test set

                        # Lưu Confusion Matrix vào file ảnh
                    confusion_matrix_path = "confusion_matrix.png"
                    fig.savefig(confusion_matrix_path)
                    # mlflow.log_artifact(confusion_matrix_path)  # Log ảnh vào MLflow
                    
                
                st.markdown(
                """
                ### 📈 Tổng kết:
                - 🚀 Mô hình có thể hoạt động tốt hoặc cần cải thiện dựa vào độ chính xác trên tập kiểm tra.
                - 📊 Quan sát ma trận nhầm lẫn** để xem nhãn nào hay bị nhầm lẫn nhất.
                - 🔍 Có thể cần điều chỉnh tham số hoặc dùng mô hình khác nếu độ chính xác chưa đủ cao.
                """
                )

    with tab_demo:   
        with st.expander("**Dự đoán kết quả**", expanded=True):
            st.write("**Dự đoán trên ảnh do người dùng tải lên**")

            # Kiểm tra xem mô hình đã được huấn luyện và lưu kết quả chưa
            if "selected_model_type" not in st.session_state or "trained_model" not in st.session_state:
                st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng huấn luyện mô hình trước khi dự đoán.")
            else:
                best_model_name = st.session_state.selected_model_type
                best_model = st.session_state.trained_model

                st.write(f"🎯 Mô hình đang sử dụng: `{best_model_name}`")
                st.write(f"✅ Độ chính xác trên tập kiểm tra: `{st.session_state.get('test_accuracy', 'N/A'):.4f}`")

                # Cho phép người dùng tải lên ảnh
                uploaded_file = st.file_uploader("📂 Chọn một ảnh để dự đoán", type=["png", "jpg", "jpeg"])

                if uploaded_file is not None:
                    # Đọc ảnh từ tệp tải lên
                    image = Image.open(uploaded_file).convert("L")  # Chuyển sang ảnh xám
                    image = np.array(image)

                    # Kiểm tra xem dữ liệu huấn luyện đã lưu trong session_state hay chưa
                    if "X_train" in st.session_state:
                        X_train_shape = st.session_state["X_train"].shape[1]  # Lấy số đặc trưng từ tập huấn luyện

                        # Resize ảnh về kích thước phù hợp với mô hình đã huấn luyện
                        image = cv2.resize(image, (28, 28))  # Cập nhật kích thước theo dữ liệu ban đầu
                        image = image.reshape(1, -1)  # Chuyển về vector 1 chiều

                        # Đảm bảo số chiều đúng với dữ liệu huấn luyện
                        if image.shape[1] == X_train_shape:
                            prediction = best_model.predict(image)[0]

                            # Hiển thị ảnh và kết quả dự đoán
                            st.image(uploaded_file, caption="📷 Ảnh bạn đã tải lên", use_container_width=True)
                            st.success(f"✅ **Dự đoán:** {prediction}")
                        else:
                            st.error(f"🚨 Ảnh không có số đặc trưng đúng ({image.shape[1]} thay vì {X_train_shape}). Hãy kiểm tra lại dữ liệu đầu vào!")
                    else:
                        st.error("🚨 Dữ liệu huấn luyện không tìm thấy. Hãy huấn luyện mô hình trước khi dự đoán.")

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
    run_ClassificationMinst_app()
    # st.write(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    # print("🎯 Kiểm tra trên DagsHub: https://dagshub.com/Dung2204/MINST.mlflow/")
    # # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App"
    # ClassificationMinst.py
