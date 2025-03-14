import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import random
import struct
from sklearn.datasets import make_circles
from sklearn.datasets import load_iris
import mlflow
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from PIL import Image
from sklearn.model_selection import KFold
import matplotlib.patches as mpatches
import time
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
        "Thông tin dữ liệu",
        "Thông tin",
        "Xử lí dữ liệu",
        "Huấn luyện mô hình",
        "Demo dự đoán",
        "Thông tin & Mlflow",
    ])
    # tab_info, tab_load, tab_preprocess, tab_split,  tab_demo, tab_log_info = tabs
    tab_info,tab_note,tab_load, tab_preprocess,  tab_demo ,tab_mlflow= tabs

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
            # image = Image.open(r'C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\App\image.png')

            # Gắn ảnh vào Streamlit và chỉnh kích thước
            # st.image(image, caption='Mô tả ảnh', width=600) 
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


    with tab_note:
        with st.expander("**Thông tin mô hình**", expanded=True):    
            # Assume model_option1 is selected from somewhere in the app
            model_option1 = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])
            if model_option1 == "Decision Tree":
                
                st.markdown("""
                ### Decision Tree (Cây quyết định)
                """)
                st.markdown("---")
                st.markdown("""
                ### Khái niệm:  
                **Decision Tree (Cây quyết định)**:
                - **Decision Tree (Cây quyết định)** là một thuật toán học máy sử dụng cấu trúc dạng cây để đưa ra quyết định phân loại hoặc dự đoán giá trị liên tục. 
                - Nó hoạt động bằng cách chia dữ liệu thành các tập con nhỏ hơn dựa trên giá trị của các đặc trưng, với mỗi nút trong cây đại diện cho một điều kiện kiểm tra, mỗi nhánh là kết quả của điều kiện đó, và mỗi lá cây là kết quả cuối cùng (nhãn lớp hoặc giá trị dự đoán).

                **Cách hoạt động**:  
                - **Cây quyết định** bắt đầu từ nút gốc (root), kiểm tra một đặc trưng của dữ liệu, và phân chia dữ liệu thành các nhánh con dựa trên kết quả kiểm tra. 
                - Quá trình này lặp lại cho đến khi dữ liệu được phân chia hoàn toàn hoặc đạt đến điều kiện dừng (ví dụ: độ sâu tối đa). 
                - Thuật toán thường sử dụng các tiêu chí như độ thuần nhất để chọn đặc trưng tốt nhất cho mỗi lần phân chia.
                """)

                st.markdown("---")
                st.subheader("**Decision Tree minh họa mô hình phân loại dữ liệu hoa Iris**")
                iris = load_iris()
                X, y = iris.data, iris.target
                feature_names = iris.feature_names
                target_names = iris.target_names

                # Chia dữ liệu thành tập huấn luyện và kiểm tra
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Thanh trượt để chọn độ sâu tối đa
                st.markdown("**Tùy chỉnh độ sâu tối đa của cây**")
                max_depth = st.slider("Chọn độ sâu tối đa (max_depth):", 1, 5, 3)

                # Huấn luyện Cây Quyết định
                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, criterion='gini')
                clf.fit(X_train, y_train)

                # Độ chính xác
                y_pred = clf.predict(X_test)
                accuracy = clf.score(X_test, y_test)
                st.write(f"Độ chính xác trên tập kiểm tra: {accuracy:.2f}")

                # Biểu đồ 1: Cây quyết định
                st.markdown("#### **Biểu đồ Cây Quyết định minh họa mô hình phân loại dữ liệu hoa Iris**")
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=True, rounded=True, ax=ax1)
                plt.title("Cây Quyết định với độ sâu tối đa = {}".format(max_depth))
                st.pyplot(fig1)

                # Biểu đồ 2: Phân chia dữ liệu tại mỗi bước
                st.subheader("**Minh họa quá trình phân chia dữ liệu**")
                # Chọn hai đặc trưng để trực quan hóa (Petal Length và Petal Width)
                X_subset = X_train[:, [2, 3]]  # Petal Length và Petal Width
                feature_subset_names = feature_names[2:4]

                # Huấn luyện lại cây quyết định trên 2 đặc trưng để trực quan hóa
                clf_subset = DecisionTreeClassifier(max_depth=max_depth, random_state=42, criterion='gini')
                clf_subset.fit(X_subset, y_train)

                # Tạo lưới điểm để vẽ ranh giới quyết định
                x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
                y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                Z = clf_subset.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                # Vẽ ranh giới quyết định và dữ liệu
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

                # Vẽ các điểm dữ liệu với màu sắc tương ứng
                scatter = ax2.scatter(X_subset[:, 0], X_subset[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolor='k')

                # Sử dụng mpatches để tạo chú thích mà không cần scatter rỗng
                colors = plt.cm.RdYlBu([0, 0.5, 1])  # Lấy màu từ cmap
                legend_elements = [
                    mpatches.Patch(color=colors[0], label='0: Setosa'),
                    mpatches.Patch(color=colors[1], label='1: Versicolor'),
                    mpatches.Patch(color=colors[2], label='2: Virginica')
                ]
                ax2.legend(handles=legend_elements, title="Nhãn", loc="best")

                ax2.set_xlabel(feature_subset_names[0])
                ax2.set_ylabel(feature_subset_names[1])
                ax2.set_title("Phân chia dữ liệu theo Cây Quyết định")
                st.pyplot(fig2)

                # Giải thích chi tiết
                st.subheader("Giải thích chi tiết")
                st.write("""
                - **Nút gốc (Root)**: Cây bắt đầu bằng cách chọn một đặc trưng (ví dụ: Petal Length) để phân chia dữ liệu. Điều kiện phân chia được hiển thị trong biểu đồ cây (ví dụ: Petal Length <= 2.45).
                - **Nhánh con**: Sau mỗi lần phân chia, dữ liệu được chia thành các tập con (True/False). Quá trình lặp lại cho đến khi đạt độ thuần nhất (Gini = 0) hoặc độ sâu tối đa.
                - **Tiêu chí Gini**: Độ thuần nhất (Gini impurity) được sử dụng để chọn đặc trưng tốt nhất tại mỗi bước. Giá trị Gini càng thấp, tập dữ liệu càng thuần nhất.
                - **Biểu đồ phân chia**: Hiển thị ranh giới quyết định trên không gian 2D (Petal Length và Petal Width), cho thấy cách cây phân chia dữ liệu thành các vùng. Nhãn được hiển thị rõ ràng với giá trị (0: Setosa, 1: Versicolor, 2: Virginica).
                """)



                st.markdown("---")
                st.markdown("""
                ### Công thức toán học:  
                **Entropy**: 
                -Đo lường độ không chắc chắn của tập dữ liệu:  
                $$
                H(S) = - \\sum_{i=1}^{c} p_i \\log_2(p_i)
                $$
                Trong đó:  
                - $$(S)$$: Tập dữ liệu.  
                - $$(c)$$: Số lớp.  
                - $$(p_i)$$: Tỷ lệ mẫu thuộc lớp \(i\).  

                **Information Gain**: Đo lường mức độ giảm **entropy** sau khi phân chia:  
                $$
                IG(S, A) = H(S) - \\sum_{j=1}^{k} \\frac{|S_v|}{|S|} H(S_v)
                $$
                Trong đó:  
                - $$(A)$$: Đặc trưng được chọn để phân chia.  
                - $$(S_v)$$: Tập con của \(S\) với giá trị \(v\) của đặc trưng \(A\).  
                """)
            
                st.markdown("---")
                st.markdown("""
                ### Hoạt động trên MNIST:  
                Với bộ dữ liệu **MNIST** (ảnh chữ số viết tay 28x28, 10 lớp từ 0-9), Decision Tree sẽ:  
                - Mỗi ảnh trong **MNIST** có kích thước 28×28 pixels, mỗi pixel có thể xem là một đặc trưng (feature).
                - Mô hình sẽ quyết định phân tách dữ liệu bằng cách chọn những pixels quan trọng nhất để tạo nhánh.
                - Ví dụ, để phân biệt chữ số 0 và 1, **Decision Tree** có thể kiểm tra:
                    - Pixel ở giữa có sáng không?
                    - Pixel dọc hai bên có sáng không?
                - Dựa trên câu trả lời, mô hình sẽ tiếp tục chia nhỏ tập dữ liệu.
                """)
                

            elif model_option1 == "SVM":
                st.markdown("""
                ### Support Vector Machine (SVM)
                """)    
                st.markdown("---")        
                st.markdown("""            
                ### Khái niệm:  
                **Support Vector Machine (SVM)**:
                - Là một thuật toán học máy mạnh mẽ, thường được sử dụng cho bài toán phân loại (đặc biệt là phân loại nhị phân) hoặc hồi quy. 
                - Ý tưởng chính của **SVM** là tìm một siêu phẳng (hyperplane) trong không gian đa chiều để phân chia các lớp dữ liệu sao cho khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất (**support vectors**) là lớn nhất có thể.

                **Cách hoạt động**:  
                - **SVM** cố gắng tối ưu hóa ranh giới phân chia giữa các lớp bằng cách tối đa hóa "khoảng cách lề" (margin) giữa siêu phẳng và các điểm dữ liệu gần nhất. 
                - Trong trường hợp dữ liệu không thể phân chia tuyến tính, SVM sử dụng các kỹ thuật như biến đổi không gian (thông qua kernel) để đưa dữ liệu vào không gian cao hơn, nơi có thể phân chia được.
                """)

                st.markdown("---")  
                st.markdown("### **Minh họa Support Vector Machine (SVM)**")  
                # Tạo dữ liệu mẫu
                # Dữ liệu tuyến tính với nhiễu
                np.random.seed(42)
                X_linear = np.concatenate([np.random.normal(0, 1, (50, 2)), np.random.normal(3, 1, (50, 2))])
                y_linear = np.array([0] * 50 + [1] * 50)
                X_linear = np.vstack([X_linear, np.random.normal(1.5, 0.5, (10, 2))])
                y_linear = np.hstack([y_linear, [0] * 5 + [1] * 5])

                # Dữ liệu không tuyến tính (hình tròn)
                X_nonlinear, y_nonlinear = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)

                # Thanh trượt để chọn giá trị C
                st.markdown("**Tùy chỉnh tham số C**")  
                C = st.slider("Chọn giá trị C (Regularization):", 0.1, 1000.0, 1.0, step=0.1)

                # Biểu đồ minh họa SVM với kernel tuyến tính
                st.markdown("#### **Biểu đồ minh họa SVM (Kernel tuyến tính)**")  
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                clf_linear = SVC(kernel='linear', C=C)
                clf_linear.fit(X_linear, y_linear)

                # Vẽ dữ liệu
                ax1.scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, cmap='bwr', edgecolor='k', label='Data Points')
                ax1.scatter(clf_linear.support_vectors_[:, 0], clf_linear.support_vectors_[:, 1], s=200, facecolors='none', edgecolor='k', label='Support Vectors')

                # Vẽ siêu phẳng và margin
                w = clf_linear.coef_[0]
                b = clf_linear.intercept_[0]
                x_plot = np.linspace(min(X_linear[:, 0]), max(X_linear[:, 0]), 100)
                y_plot = -(w[0] * x_plot + b) / w[1]  # Siêu phẳng
                margin = 1 / np.sqrt(np.sum(clf_linear.coef_ ** 2))
                y_margin_up = y_plot + margin  # Margin trên
                y_margin_down = y_plot - margin  # Margin dưới

                ax1.plot(x_plot, y_plot, 'k-', label='Siêu phẳng (Decision Boundary)')
                ax1.plot(x_plot, y_margin_up, 'k--', label='Tối ưu hóa lề (Margin)')
                ax1.plot(x_plot, y_margin_down, 'k--')

                ax1.set_title(f'SVM với C = {C} (Kernel tuyến tính)')
                ax1.set_xlabel('Feature 1')
                ax1.set_ylabel('Feature 2')
                ax1.legend()

                # Hiển thị biểu đồ 1
                st.pyplot(fig1)

                # Biểu đồ minh họa SVM với kernel RBF
                st.markdown("#### **Biểu đồ minh họa SVM (Kernel RBF - Kernel Trick)**")  
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                clf_rbf = SVC(kernel='rbf', C=C)
                clf_rbf.fit(X_nonlinear, y_nonlinear)

                # Vẽ dữ liệu
                ax2.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear, cmap='bwr', edgecolor='k', label='Data Points')
                ax2.scatter(clf_rbf.support_vectors_[:, 0], clf_rbf.support_vectors_[:, 1], s=200, facecolors='none', edgecolor='k', label='Support Vectors')

                # Vẽ ranh giới quyết định (không tuyến tính)
                x_min, x_max = X_nonlinear[:, 0].min() - 1, X_nonlinear[:, 0].max() + 1
                y_min, y_max = X_nonlinear[:, 1].min() - 1, X_nonlinear[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                Z = clf_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                ax2.contour(xx, yy, Z, levels=[0], colors='k', linestyles=['-'], label='Siêu phẳng (Decision Boundary)')

                ax2.set_title(f'SVM với C = {C} (Kernel RBF)')
                ax2.set_xlabel('Feature 1')
                ax2.set_ylabel('Feature 2')
                ax2.legend()

                # Hiển thị biểu đồ 2
                st.pyplot(fig2)

                # Giải thích các thành phần
                st.markdown("#### **Giải thích các thành phần trong biểu đồ:**")  
                st.write("""
                - **Siêu phẳng (Decision Boundary)**: 
                    - **Trong biểu đồ kernel tuyến tính**: Đường thẳng đen phân chia hai lớp dữ liệu.
                    - **Trong biểu đồ kernel RBF**: Đường cong đen (ranh giới quyết định) thể hiện siêu phẳng trong không gian cao hơn sau khi áp dụng kernel trick.
                - **Tối ưu hóa lề (Margin)**: 
                    - **Trong biểu đồ kernel tuyến tính**: Hai đường nét đứt đen song song với siêu phẳng, thể hiện khoảng cách tối đa đến support vectors.
                    - Support vectors (điểm khoanh tròn) xác định margin.
                - **Tham số C (Regularization)**: 
                    - Điều chỉnh bằng thanh trượt. Khi \( C \) lớn, margin hẹp hơn, mô hình cố gắng phân loại đúng tất cả các điểm (nhưng dễ overfitting). Khi \( C \) nhỏ, margin rộng hơn, chấp nhận một số lỗi để tổng quát hóa tốt hơn.
                    - Các điểm nhiễu (nếu có) sẽ nằm trong margin hoặc sai phía siêu phẳng khi \( C \) nhỏ.
                - **Kernel Trick**: 
                    - Minh họa trong **biểu đồ kernel RBF**, nơi dữ liệu không tuyến tính (hình tròn) được phân tách bằng ranh giới cong nhờ ánh xạ vào không gian cao hơn.
                """)

                # Giải thích biểu đồ
                st.markdown("#### **Giải thích biểu đồ:**")  
                st.write("""
                - **Điểm dữ liệu**: Các điểm màu đỏ (Class 0) và xanh (Class 1) đại diện cho dữ liệu huấn luyện.
                - **Support Vectors**: Các điểm được khoanh tròn là support vectors, nằm gần nhất với siêu phẳng và xác định margin.
                - **Siêu phẳng**: Đường thẳng đen $$( w^T x + b = 0 )$$ là ranh giới quyết định.
                - **Margin**: Hai đường nét đứt là biên của margin, được tối đa hóa bởi SVM.
                - **Tham số C**: Khi $$( C )$$ lớn, margin hẹp hơn và mô hình cố gắng phân loại đúng tất cả các điểm. Khi $$( C )$$ nhỏ, margin rộng hơn, chấp nhận một số lỗi để tổng quát hóa tốt hơn.
                """)

 
                st.markdown("---")          
                st.markdown("""
                ### Công thức toán học:  
                **Siêu phẳng**:
                - **Siêu phẳng** đóng vai trò làm ranh giới quyết định, phân chia các lớp dữ liệu (ví dụ: lớp 0 và lớp 1) trong không gian đặc trưng, đảm bảo khoảng cách lớn nhất đến các điểm gần nhất.
                - Được định nghĩa bởi phương trình:  
                $$
                w^T x + b = 0
                $$
                Trong đó:  
                - \(w\): Vector trọng số (vuông góc với siêu phẳng).  
                - \(x\): Vector đặc trưng.  
                - \(b\): Độ lệch (bias).  

                **Tối ưu hóa lề**: 
                - **Tối ưu hóa lề** là bài toán tối ưu hóa nhằm tìm **siêu phẳng** tốt nhất, tối đa hóa margin (khoảng cách giữa siêu phẳng và các support vectors) bằng cách giảm thiểu độ dài vector \(w\), đồng thời đảm bảo tất cả các điểm dữ liệu được phân loại đúng.
                - Được định nghĩa bởi phương trình:  
                $$
                \\min_{w, b} \\frac{1}{2} ||w||^2 \\quad \\text{với điều kiện} \\quad y_i (w^T x_i + b) \\geq 1, \\forall i
                $$
                Trong đó:  
                - $$(||w||)$$: Độ dài vector \(w\).
                - $$(y_i)$$: Nhãn của mẫu \(i\) (\(+1\) hoặc \(-1\)).  
                - $$(x_i)$$: Vector đặc trưng của mẫu \(i\).  

                **Kernel Trick**: Khi dữ liệu không tuyến tính, sử dụng hàm kernel $$(K(x_i, x_j))$$ để ánh xạ dữ liệu vào không gian cao hơn.
                            
                **Tham số C (Regularization)**:  
                - Trong trường hợp dữ liệu không thể phân tách tuyến tính hoàn hảo , tham số $$(C)$$ được sử dụng để cân bằng giữa việc tối đa hóa lề và giảm thiểu lỗi phân loại.  
                -$$(C)$$ kiểm soát mức độ phạt cho các điểm dữ liệu bị phân loại sai (vi phạm lề). Giá trị $$(C)$$ càng lớn, mô hình càng cố gắng phân loại đúng tất cả các điểm (giống hard margin), nhưng có thể dẫn đến overfitting. Giá trị $$(C)$$ nhỏ hơn sẽ cho phép một số điểm bị phân loại sai để tăng lề, giúp mô hình tổng quát hóa tốt hơn.  
                - Phương trình tối ưu hóa với soft margin bao gồm \(C\):  
                $$
                \\min_{w, b, \\xi} \\frac{1}{2} ||w||^2 + C \\sum_{i=1}^n \\xi_i \\quad \\text{với điều kiện} \\quad y_i (w^T x_i + b) \\geq 1 - \\xi_i, \\quad \\xi_i \\geq 0, \\forall i
                $$
                Trong đó:  
                - $$(C)$$: Tham số điều chỉnh mức độ phạt (regularization parameter).  
                - $$(\\xi_i)$$: Biến slack (đo lường mức độ vi phạm lề của mẫu $$(i)$$).
                """) 
               
                

                

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
                X = np.concatenate((train_images, test_images), axis=0)  # Gộp toàn bộ dữ liệu
                y = np.concatenate((train_labels, test_labels), axis=0)
                X = X.reshape(X.shape[0], -1)  # Chuyển thành vector 1 chiều
                    # Cho phép người dùng chọn tỷ lệ validation và test
                test_size = st.slider("🔹 Chọn % tỷ lệ tập test", min_value=10, max_value=50, value=20, step=5) / 100
                val_size = st.slider("🔹 Chọn % tỷ lệ tập validation (trong phần train)", min_value=10, max_value=50, value=20, step=5) / 100

                X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                val_size_adjusted = val_size / (1 - test_size)  # Điều chỉnh tỷ lệ val cho phần còn lại
                X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

                    # Tính tỷ lệ thực tế của từng tập
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
        with st.expander("**Huấn luyện mô hình**", expanded=True):
            # Lựa chọn mô hình
            model_option = st.radio("🔹 Chọn mô hình huấn luyện:", ("Decision Tree", "SVM"))
            
            if model_option == "Decision Tree":
                st.subheader("🌳 Decision Tree Classifier")
                        
                # Lựa chọn tham số cho Decision Tree
                max_depth = st.slider("Chọn độ sâu tối đa của cây:", min_value=1, max_value=20, value=5)
                st.session_state["dt_max_depth"] = max_depth
                n_folds = st.slider("Chọn số folds cho K-Fold Cross-Validation:", min_value=2, max_value=10, value=5)

                if st.button("🚀 Huấn luyện mô hình"):
                    with mlflow.start_run():
                        # Khởi tạo mô hình Decision Tree
                        dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

                        # Thực hiện K-Fold Cross-Validation
                        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                        cv_scores = []
                        progress_bar = st.progress(0)  # Thanh trạng thái bắt đầu từ 0%
                        progress_text = st.empty()    # Vùng hiển thị % tiến trình
                        total_steps = n_folds * 100   # Tổng số bước (mỗi fold có 100 bước nhỏ)

                        current_step = 0  # Biến đếm bước hiện tại
                        for i, (train_index, val_index) in enumerate(kf.split(X_train)):
                            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                            # Huấn luyện trên fold hiện tại
                            dt_model.fit(X_train_fold, y_train_fold)
                            y_val_pred_fold = dt_model.predict(X_val_fold)
                            fold_accuracy = accuracy_score(y_val_fold, y_val_pred_fold)
                            cv_scores.append(fold_accuracy)

                            # Giả lập tiến trình mượt mà trong mỗi fold
                            for sub_step in range(100):  # 100 bước nhỏ trong mỗi fold
                                current_step += 1
                                progress_percent = int((current_step / total_steps) * 100)  # Tính % hoàn thành
                                progress_bar.progress(progress_percent / 100)              # Cập nhật thanh trạng thái
                                progress_text.text(f"Tiến trình huấn luyện: {progress_percent}%")
                                time.sleep(0.01)  # Delay nhỏ để tạo hiệu ứng mượt mà

                        # Đảm bảo tiến trình đạt 100%
                        progress_bar.progress(1.0)
                        progress_text.text("Tiến trình huấn luyện: 100% - Đã hoàn thành!")

                        # Tính độ chính xác trung bình từ cross-validation
                        mean_cv_accuracy = np.mean(cv_scores)
                        std_cv_accuracy = np.std(cv_scores)

                        # Huấn luyện mô hình trên toàn bộ X_train
                        dt_model.fit(X_train, y_train)
                        y_val_pred_dt = dt_model.predict(X_val)
                        accuracy_dt = accuracy_score(y_val, y_val_pred_dt)

                        # Ghi log vào MLflow
                        mlflow.log_param("model_type", "Decision Tree")
                        mlflow.log_param("max_depth", max_depth)
                        mlflow.log_param("n_folds", n_folds)
                        mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)
                        mlflow.log_metric("std_cv_accuracy", std_cv_accuracy)
                        mlflow.log_metric("accuracy", accuracy_dt)
                        mlflow.sklearn.log_model(dt_model, "decision_tree_model")

                        # Lưu vào session_state
                        st.session_state["selected_model_type"] = "Decision Tree"
                        st.session_state["trained_model"] = dt_model 
                        st.session_state["X_train"] = X_train 
                        st.session_state["dt_max_depth"] = max_depth
                        st.session_state["n_folds"] = n_folds 

                        st.markdown("---") 
                        st.write(f"🔹Mô hình được chọn để đánh giá: `{model_option}`")
                        st.write("🔹 Tham số mô hình:")
                        st.write(f"- **Độ sâu tối đa**: `{max_depth}`")
                        st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")
                        st.write(f"✅ **Độ chính xác trung bình từ K-Fold Cross-Validation ({n_folds} folds):** `{mean_cv_accuracy:.4f} ± {std_cv_accuracy:.4f}`")
                        st.write(f"✅ **Độ chính xác trên tập validation:** `{accuracy_dt:.4f}`")
                    
                    mlflow.end_run()

            elif model_option == "SVM":
                st.subheader("🌀 Support Vector Machine (SVM)")
                            
                # Lựa chọn tham số cho SVM
                kernel = st.selectbox("Chọn kernel:", ["linear", "poly", "rbf", "sigmoid"])
                C = st.slider("Chọn giá trị C (điều chỉnh mức độ regularization):", min_value=0.1, max_value=10.0, value=1.0)
                n_folds = st.slider("Chọn số folds cho K-Fold Cross-Validation:", min_value=2, max_value=10, value=5)

                if st.button("🚀 Huấn luyện mô hình"):
                    with mlflow.start_run():
                        # Khởi tạo mô hình SVM
                        svm_model = SVC(kernel=kernel, C=C, random_state=42)

                        # Thực hiện K-Fold Cross-Validation
                        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                        cv_scores = []
                        progress_bar = st.progress(0)  # Thanh trạng thái bắt đầu từ 0%
                        progress_text = st.empty()    # Vùng hiển thị % tiến trình
                        total_steps = n_folds * 100   # Tổng số bước (mỗi fold có 100 bước nhỏ)

                        current_step = 0  # Biến đếm bước hiện tại
                        for i, (train_index, val_index) in enumerate(kf.split(X_train)):
                            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                            # Huấn luyện trên fold hiện tại
                            svm_model.fit(X_train_fold, y_train_fold)
                            y_val_pred_fold = svm_model.predict(X_val_fold)
                            fold_accuracy = accuracy_score(y_val_fold, y_val_pred_fold)
                            cv_scores.append(fold_accuracy)

                            # Giả lập tiến trình mượt mà trong mỗi fold
                            for sub_step in range(100):  # 100 bước nhỏ trong mỗi fold
                                current_step += 1
                                progress_percent = int((current_step / total_steps) * 100)  # Tính % hoàn thành
                                progress_bar.progress(progress_percent / 100)              # Cập nhật thanh trạng thái
                                progress_text.text(f"Tiến trình huấn luyện: {progress_percent}%")
                                time.sleep(0.01)  # Delay nhỏ để tạo hiệu ứng mượt mà

                        # Đảm bảo tiến trình đạt 100%
                        progress_bar.progress(1.0)
                        progress_text.text("Tiến trình huấn luyện: 100% - Đã hoàn thành!")

                        # Tính độ chính xác trung bình từ cross-validation
                        mean_cv_accuracy = np.mean(cv_scores)
                        std_cv_accuracy = np.std(cv_scores)

                        # Huấn luyện mô hình trên toàn bộ X_train
                        svm_model.fit(X_train, y_train)
                        y_val_pred_svm = svm_model.predict(X_val)
                        accuracy_svm = accuracy_score(y_val, y_val_pred_svm)

                        # Ghi log vào MLflow
                        mlflow.log_param("model_type", "SVM")
                        mlflow.log_param("kernel", kernel)
                        mlflow.log_param("C_value", C)
                        mlflow.log_param("n_folds", n_folds)
                        mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)
                        mlflow.log_metric("std_cv_accuracy", std_cv_accuracy)
                        mlflow.log_metric("accuracy", accuracy_svm)
                        mlflow.sklearn.log_model(svm_model, "svm_model")

                        # Lưu vào session_state
                        st.session_state["selected_model_type"] = "SVM"
                        st.session_state["trained_model"] = svm_model  
                        st.session_state["X_train"] = X_train
                        st.session_state["svm_kernel"] = kernel
                        st.session_state["svm_C"] = C
                        st.session_state["n_folds"] = n_folds

                        st.markdown("---") 
                        st.write(f"🔹Mô hình được chọn để đánh giá: `{model_option}`")
                        st.write("🔹 **Tham số mô hình:**")
                        st.write(f"- Kernel: `{kernel}`")
                        st.write(f"- C (Regularization): `{C}`")
                        st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")
                        st.write(f"✅ **Độ chính xác trung bình từ K-Fold Cross-Validation ({n_folds} folds):** `{mean_cv_accuracy:.4f} ± {std_cv_accuracy:.4f}`")
                        st.write(f"✅ **Độ chính xác trên tập validation:** `{accuracy_svm:.4f}`")
                    
                    mlflow.end_run()

                        
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
                # st.write(f"✅ Độ chính xác trên tập kiểm tra: `{st.session_state.get('test_accuracy', 'N/A'):.4f}`")

                # Lấy các tham số từ session_state để hiển thị
                if best_model_name == "Decision Tree":
                    criterion = st.session_state.get("dt_criterion", "entropy")
                    max_depth = st.session_state.get("dt_max_depth", 5)  # Giá trị mặc định là 5
                    n_folds = st.session_state.get("n_folds", 5)  # Giá trị mặc định là 5
                    st.write("🔹 **Tham số mô hình Decision Tree:**")
                    st.write(f"- **Tiêu chí phân nhánh**: `{criterion}`")
                    st.write(f"- **Độ sâu tối đa**: `{max_depth}`")
                    st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")
                elif best_model_name == "SVM":
                    kernel = st.session_state.get("svm_kernel", "linear")
                    C = st.session_state.get("svm_C", 1.0)
                    n_folds = st.session_state.get("n_folds", 5)  # Giá trị mặc định là 5
                    st.write("🔹 **Tham số mô hình SVM:**")
                    st.write(f"- **Kernel**: `{kernel}`")
                    st.write(f"- **C (Regularization)**: `{C}`")
                    st.write(f"- **Số folds trong Cross-Validation**: `{n_folds}`")

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
            experiment_name = "Classification"

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

            if runs:
                # Lọc các run theo loại mô hình đã chọn
                selected_model_type = st.session_state.get("selected_model_type", "Decision Tree")
                run_options = {run.info.run_id: f"{run.data.tags.get('mlflow.runName', 'Unnamed')} - {run.info.run_id}"
                            for run in runs if run.data.params.get("model_type") == selected_model_type}

                if not run_options:
                    st.info(f"Chưa có Run nào cho mô hình {selected_model_type}.")
                else:
                    # Chọn Run để xem chi tiết
                    st.subheader("Danh sách các Run đã log")
                    selected_run_id = st.selectbox("Chọn Run để xem chi tiết:", 
                                                options=list(run_options.keys()), 
                                                format_func=lambda x: run_options[x])

                    selected_run = client.get_run(selected_run_id)
                    st.write(f"**Run ID:** {selected_run_id}")
                    st.write(f"**Run Name:** {selected_run.data.tags.get('mlflow.runName', 'Unnamed')}")

                    # Hiển thị tham số theo loại mô hình
                    st.markdown("### Tham số đã log")
                    if selected_model_type == "Decision Tree":
                        params = {
                            "model_type": selected_run.data.params.get("model_type", "N/A"),
                            "max_depth": selected_run.data.params.get("max_depth", "N/A"),
                            "n_folds": selected_run.data.params.get("n_folds", "N/A")
                        }
                    elif selected_model_type == "SVM":
                        params = {
                            "model_type": selected_run.data.params.get("model_type", "N/A"),
                            "kernel": selected_run.data.params.get("kernel", "N/A"),
                            "C_value": selected_run.data.params.get("C_value", "N/A"),
                            "n_folds": selected_run.data.params.get("n_folds", "N/A")
                        }
                    st.json(params)

                    # Hiển thị chỉ số
                    st.markdown("### Chỉ số đã log")
                    metrics = {
                        "mean_cv_accuracy": selected_run.data.metrics.get("mean_cv_accuracy", "N/A"),
                        "std_cv_accuracy": selected_run.data.metrics.get("std_cv_accuracy", "N/A"),
                        "accuracy": selected_run.data.metrics.get("accuracy", "N/A")
                    }
                    st.json(metrics)

                    # Đổi tên Run
                    st.subheader("Đổi tên Run")
                    new_run_name = st.text_input("Nhập tên mới cho Run:", 
                                                value=run_options[selected_run_id].split(" - ")[0])
                    if st.button("Cập nhật tên Run"):
                        if new_run_name.strip():
                            client.set_tag(selected_run_id, "mlflow.runName", new_run_name.strip())
                            st.success(f"Đã cập nhật tên Run thành: {new_run_name.strip()}")
                        else:
                            st.warning("Vui lòng nhập tên mới cho Run.")

                    # Xóa Run
                    st.subheader("Xóa Run")
                    if st.button("Xóa Run", key="delete_run"):
                        client.delete_run(selected_run_id)
                        st.success(f"Đã xóa Run {run_options[selected_run_id]} thành công!")
                        st.experimental_rerun()

                    # Nút mở MLflow UI
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
    # ClassificationMinst.
    



    ## thay vì decision tree là gini và entropy thì -> chỉ còn entropy với chọn độ sâu của cây
    ## bổ sung thêm Chọn số folds (KFold Cross-Validation) ở cả 2 phần decsion tree và svms
    ## cập nhật lại phần demo , vì nó đang không sử dụng dữ liệu ở phần huấn luyện
  
