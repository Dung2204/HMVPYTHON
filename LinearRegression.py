import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor,LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import mlflow
from mlflow.tracking import MlflowClient

def run_LinearRegression_app():

    mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]
    mlflow_username = st.secrets["MLFLOW_TRACKING_USERNAME"]
    mlflow_password = st.secrets["MLFLOW_TRACKING_PASSWORD"]
    
    # Thiết lập biến môi trường
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    
    # Thiết lập MLflow (Đặt sau khi mlflow_tracking_uri đã có giá trị)
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Thiết lập biến môi trường
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

    # Thiết lập MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)


    # Khởi tạo session_state nếu chưa có
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'data_split' not in st.session_state:
        st.session_state.data_split = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'X_val' not in st.session_state:
        st.session_state.X_val = None
    if 'y_val' not in st.session_state:
        st.session_state.y_val = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = None
    if "run_name" not in st.session_state:
        st.session_state["run_name"] = None

    st.title("Dự đoán sự sống với tập dữ liệu Titanic")

    # Tạo các tab
    tabs = st.tabs([
        "Tập dữ liệu",
        "Thông tin",
        "Huấn luyện mô hình",
        "Dự đoán",
        "Thông tin huấn luyện & MLflow UI"
    ])
    tab_analysis,tab_note, tab_train, tab_predict, tab_mlflow = tabs

    with tab_analysis:
        st.header("Phân tích và xử lý dữ liệu")
        with st.expander("📥 Tải dữ liệu", expanded=True):  
            uploaded_file = st.file_uploader("Tải file CSV (Titanic dataset)", type=["csv"])
            if uploaded_file is not None:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.write("Dữ liệu đã được tải lên:")
                st.write(st.session_state.df.head(10))
                st.session_state.data_loaded = True


        with st.expander("📥 Thông tin của tập dữ liệu", expanded=True):
            if st.session_state.get("data_loaded", False):
                st.markdown("""
                    - `PassengerId` : ID của hành khách  
                    - `Survived` : Biến mục tiêu (0 = Không sống sót, 1 = Sống sót).  
                    - `Pclass` : Hạng vé (1, 2, 3).  
                    - `Name` : Tên hành khách.  
                    - `Sex` : Giới tính (male, female).  
                    - `Age` : Tuổi  
                    - `SibSp` : Số anh chị em hoặc vợ/chồng đi cùng.  
                    - `Parch` : Số cha mẹ hoặc con cái đi cùng.  
                    - `Ticket` : Số vé.  
                    - `Fare` : Giá vé  
                    - `Cabin` : Số phòng  
                    - `Embarked` : Cảng lên tàu (C = Cherbourg, Q = Queenstown, S = Southampton).  
                """, unsafe_allow_html=True)    
                

        with st.expander("🔍 Kiểm tra dữ liệu"):
            if st.session_state.get("data_loaded", False):
                
                df = st.session_state.df
                missing_values = df.isnull().sum()                                 
                outlier_count = {
                            col: (abs(zscore(df[col], nan_policy='omit')) > 3).sum()
                            for col in df.select_dtypes(include=['number']).columns
                }
                error_report = pd.DataFrame({
                            'Cột': df.columns,
                            'Giá trị thiếu': missing_values,
                            'Outlier': [outlier_count.get(col, 0) for col in df.columns]
                })

                st.table(error_report)

        with st.expander("⚙️ Xử lý dữ liệu"):
            if st.session_state.get("data_loaded", False):
                df = st.session_state.df
                st.write("**Xử lý dữ liệu:** Loại bỏ các cột không cần thiết, điền giá trị thiếu, mã hóa biến phân loại, và chuẩn hóa dữ liệu.")
                dropped_cols = st.multiselect("**1️⃣ Chọn cột cần loại bỏ:**", df.columns.tolist(), default=["PassengerId", "Name", "Ticket", "Cabin"])
                df.drop(columns=dropped_cols, errors='ignore', inplace=True)
                # st.write(f"Đã loại bỏ các cột: {', '.join(dropped_cols)}")
                st.write("**2️⃣ Điền giá trị thiếu**:")
                missing_cols = df.columns[df.isnull().sum() > 0]  # Chỉ lấy các cột có giá trị thiếu
                fill_missing_cols = st.multiselect("Chọn cột để điền giá trị thiếu:", missing_cols.tolist())
                for col in fill_missing_cols:
                    if df[col].isnull().sum() > 0:  # Kiểm tra lại nếu cột vẫn có giá trị thiếu
                        method = st.selectbox(f"**Phương pháp điền cho cột** {col}:", 
                                            options=["trung vị (median)", "trung bình (mean)", "mode", "loại bỏ"], 
                                            key=f"fill_{col}")
                        if df[col].dtype in ['float64', 'int64']:
                            if method == "trung vị (median)":
                                df[col].fillna(df[col].median(), inplace=True)
                            elif method == "trung bình (mean)":
                                df[col].fillna(df[col].mean(), inplace=True)
                            elif method == "loại bỏ":
                                df.dropna(subset=[col], inplace=True)
                        else:
                            if method == "mode":
                                df[col].fillna(df[col].mode()[0], inplace=True)
                            elif method == "loại bỏ":
                                df.dropna(subset=[col], inplace=True)
                st.write("**3️⃣ Mã hóa các biến phân loại**:")
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                cols_to_encode = st.multiselect("Chọn cột để mã hóa:", categorical_cols)
                for col in cols_to_encode:
                    df[col] = df[col].astype('category').cat.codes
                    # st.write(f"Đã mã hóa cột {col}.")
                st.write("4️⃣ Chuẩn hóa dữ liệu số:")
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                if "Survived" in numeric_cols:
                    numeric_cols.remove("Survived")
                norm_method = st.selectbox("Chọn phương pháp chuẩn hóa:", ["Min-Max Scaling", "Standard Scaling"], key="norm_method")
                if norm_method == "Min-Max Scaling":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    if st.toggle("Hiển thị thông tin MinMaxScaler"):
                        st.write("Chuẩn hóa dữ liệu về khoảng [0,1] hoặc [-1,1], giúp duy trì tỷ lệ giữa các giá trị gốc.")

                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    if st.toggle("Hiển thị thông tin Standard Scaling"):
                        st.write("Biến đổi dữ liệu về trung bình 0 và độ lệch chuẩn 1, phù hợp với dữ liệu có phân phối chuẩn.")
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                # st.write(f"Đã chuẩn hóa các cột số: {', '.join(numeric_cols)}")
                st.success("Dữ liệu đã được xử lý!")
                st.write(df.head(10))
                st.session_state.df = df
                st.session_state.data_processed = True
            else:
                st.warning("Vui lòng tải dữ liệu trước.")

    with tab_note:
    # Tiêu đề ứng dụng
        option = st.selectbox("Chọn loại hồi quy ", ["Hồi quy tuyến tính đơn", "Hồi quy tuyến tính bội", "Hồi quy đa thức"])
        # Giải thích lý thuyết bằng st.markdown()
        if option == "Hồi quy tuyến tính đơn":
            st.header("Hồi quy tuyến tính đơn(Simple Linear Regression)")
            st.markdown("---")
            st.markdown("""
            ###  **Lý thuyết**
            - **Hồi quy tuyến tính đơn (Simple Linear Regression)**: 
                - **Simple Linear Regression** là một kỹ thuật thống kê và học máy cơ bản dùng để mô hình hóa mối quan hệ tuyến tính giữa một biến độc lập (X) và một biến phụ thuộc (Y). 
                - Đây là dạng đơn giản nhất của hồi quy tuyến tính, với mục tiêu dự đoán giá trị của Y dựa trên X thông qua một đường thẳng.
            - **Công thức tổng quát**:
            $$
            Y = w_0 + w_1X + \\epsilon
            $$
            **Trong đó**:
            - ( Y ): Biến phụ thuộc (giá trị cần dự đoán).
            - ( X ): Biến độc lập.
            -  $$( w_0 ) $$: Hệ số chặn (intercept).
            -  $$( w_1 ) $$: Hệ số góc (slope).
            - $$( epsilon )$$: Sai số ngẫu nhiên.
            """)
            st.markdown("---")  
            st.markdown("""            
            ### 📊 **Ví dụ đơn giản**
            Giả sử bạn muốn dự đoán giá nhà (\\( Y \\)) dựa trên diện tích (\\( X \\)):
            $$
            Y = 50 + 10X
            $$
            - $$( w_0 = 50) $$: Giá trị cơ bản (intercept).
            - $$( w_1 = 10) $$: Hệ số cho thấy mỗi đơn vị diện tích tăng thêm làm giá nhà tăng 10 đơn vị.
            """)

            st.markdown("### 📊 **Ví dụ minh họa với dữ liệu chiều cao và cân nặng**")
            X = np.array([[147], [150], [153], [158], [163], [165], [168], [170], [173], [175], [178], [180], [183]])
            y = np.array([[49], [50], [51], [54], [58], [59], [60], [62], [63], [64], [66], [67], [68]])

            # Thêm cột 1 vào X để tính intercept (w_0)
            one = np.ones((X.shape[0], 1))
            Xbar = np.concatenate((one, X), axis=1)

            # Tính toán các trọng số (weights) của đường hồi quy
            A = np.dot(Xbar.T, Xbar)
            b = np.dot(Xbar.T, y)
            w = np.dot(np.linalg.pinv(A), b)

            # Chuẩn bị đường hồi quy
            w_0 = w[0][0]
            w_1 = w[1][0]
            x0 = np.linspace(145, 185, 2)
            y0 = w_0 + w_1 * x0

            # Vẽ biểu đồ
            fig, ax = plt.subplots()
            ax.plot(X.T, y.T, 'ro')  # Dữ liệu thực tế
            ax.plot(x0, y0, label='Đường hồi quy')   # Đường hồi quy
            ax.set_xlim(140, 190)
            ax.set_ylim(45, 75)
            ax.set_xlabel('Chiều cao (cm)')
            ax.set_ylabel('Cân nặng (kg)')
            ax.legend()

            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

            # In kết quả w bên dưới biểu đồ
            st.markdown("**Kết quả trọng số (weights):**")
            st.write(f"w = [{w_0}, {w_1}]")


                        
            
        elif option == "Hồi quy tuyến tính bội":
            st.header(" Hồi quy tuyến tính bội (Multiple Linear Regression)")
            st.markdown("---") 
            st.markdown("""
            ###  **Lý thuyết**
            - **Hồi quy tuyến tính bội (Multiple Linear Regression)**:
                - **Multiple Linear Regression** là một kỹ thuật thống kê và học máy mở rộng từ **Simple Linear Regression**,
                - Được sử dụng để mô hình hóa mối quan hệ tuyến tính giữa một biến phụ thuộc (dependent variable) và nhiều biến độc lập (independent variables). 
                - Mục tiêu là dự đoán giá trị của biến phụ thuộc dựa trên sự kết hợp của nhiều biến độc lập.
            - **Công thức tổng quát**:
            $$
            Y = w_0 + w_1X_1 + w_2X_2 + ... + w_nX_n + \\epsilon
            $$
            **Trong đó:**   
            - ( Y ): Biến phụ thuộc (giá trị cần dự đoán, ví dụ: giá nhà).
            - $$( X_1, X_2, ..., X_n)$$: Các biến độc lập (ví dụ: diện tích, số phòng).
            - $$( w_0 )$$: Hệ số chặn (intercept).
            - $$( w_1, w_2, ..., w_n )$$: Các hệ số hồi quy (weights).
            - $$(epsilon)$$: Sai số ngẫu nhiên.
            """)
            
            st.markdown("---") 
            st.markdown("### 📊 **Ví dụ minh họa: Dự đoán giá nhà**")
            st.markdown("""
            **Dữ liệu mẫu:**
            - Diện tích ($$X_1$$ (m²)): 
                - [40, 50, 60, 70, 80, 90]
            - Số phòng $$(X_2)$$: 
                - [1, 2, 3, 4, 5, 6]
            - Giá nhà (Y, triệu VNĐ): 
                - [150, 200, 250, 300, 350, 400]
            """)

            # Dữ liệu mẫu
            X1 = np.array([[40],[50], [60], [70], [80], [90]])  # Diện tích
            X2 = np.array([[1],[2], [3], [4], [5], [6]])     # Số phòng
            Y = np.array([[150],[200], [250], [300], [350], [400]])  # Giá nhà

            # Kết hợp các biến độc lập thành ma trận
            one = np.ones((X1.shape[0], 1))  # Cột 1 cho intercept
            Xbar = np.concatenate((one, X1, X2), axis=1)  # Ma trận [1, X1, X2]

            # Tính toán các trọng số (weights)
            A = np.dot(Xbar.T, Xbar)
            b = np.dot(Xbar.T, Y)
            w = np.dot(np.linalg.pinv(A), b)

            # Trích xuất w_0, w_1, w_2
            w_0 = w[0][0]
            w_1 = w[1][0]
            w_2 = w[2][0]

            # Chuẩn bị dữ liệu để vẽ biểu đồ 3D
            X1_grid, X2_grid = np.meshgrid(np.linspace(40, 90, 10), np.linspace(1, 6, 10))
            Y_pred = w_0 + w_1 * X1_grid + w_2 * X2_grid

            # Vẽ biểu đồ 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X1, X2, Y, c='r', marker='o', label='Dữ liệu thực tế')
            ax.plot_surface(X1_grid, X2_grid, Y_pred, alpha=0.5, cmap='viridis', label='Mặt phẳng hồi quy')
            ax.set_xlabel('Diện tích (m²)')
            ax.set_ylabel('Số phòng')
            ax.set_zlabel('Giá nhà (triệu VNĐ)')
            ax.legend()

            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

            # In kết quả w
            st.markdown("**Kết quả trọng số (weights):**")
            st.write(f"w = [{w_0:.4f}, {w_1:.4f}, {w_2:.4f}]")
            st.markdown("""
            - $$( w_0 )$$: Hệ số chặn (intercept).
            - $$( w_1 )$$: Hệ số cho biến diện tích $$(X_1)$$.
            - $$( w_2 )$$: Hệ số cho biến số phòng $$(X_2)$$.
            """)

            
            st.markdown("### 📊 **Ví dụ: Dự đoán điểm số học tập**")
            st.markdown("""
            **Dữ liệu mẫu:**
            - Số giờ học mỗi ngày $$(X_1)$$: 
                - [1, 2, 3, 4, 5, 6, 7, 8]
            - Số giờ ngủ mỗi đêm $$(X_2)$$: 
                - [5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 8.5, 9.0]
            - Điểm số trung bình (Y): 
                - [ 65, 70, 75, 80, 85, 90, 95, 100]
            """)

            # Dữ liệu mẫu
            X1 = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  # Số giờ học
            X2 = np.array([[5.0], [5.5], [6.0], [6.5], [7.0], [8.0], [8.5], [9.0]])  # Số giờ ngủ
            Y = np.array([[65], [70], [75], [80], [85], [90], [95], [100]])  # Điểm số

            # Kết hợp các biến độc lập thành ma trận
            one = np.ones((X1.shape[0], 1))  # Cột 1 cho intercept
            Xbar = np.concatenate((one, X1, X2), axis=1)  # Ma trận [1, X1, X2]

            # Tính toán các trọng số (weights)
            A = np.dot(Xbar.T, Xbar)
            b = np.dot(Xbar.T, Y)
            w = np.dot(np.linalg.pinv(A), b)

            # Trích xuất w_0, w_1, w_2
            w_0 = w[0][0]
            w_1 = w[1][0]
            w_2 = w[2][0]

            # Chuẩn bị dữ liệu để vẽ biểu đồ 3D
            X1_grid, X2_grid = np.meshgrid(np.linspace(1, 6, 10), np.linspace(5, 9, 10))
            Y_pred = w_0 + w_1 * X1_grid + w_2 * X2_grid

            # Vẽ biểu đồ 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X1, X2, Y, c='r', marker='o', label='Dữ liệu thực tế')
            ax.plot_surface(X1_grid, X2_grid, Y_pred, alpha=0.5, cmap='viridis', label='Mặt phẳng hồi quy')
            ax.set_xlabel('Số giờ học mỗi ngày')
            ax.set_ylabel('Số giờ ngủ mỗi đêm')
            ax.set_zlabel('Điểm số trung bình')
            ax.legend()

            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

            # In kết quả w
            st.markdown("**Kết quả trọng số (weights):**")
            st.write(f"w = [{w_0:.4f}, {w_1:.4f}, {w_2:.4f}]")
            st.markdown("""
            -  $$( w_0 ) $$: Hệ số chặn (intercept).
            -  $$( w_1 ) $$: Hệ số cho số giờ học  $$(X_1) $$.
            -  $$( w_2 ) $$: Hệ số cho số giờ ngủ  $$(X_2) $$.
            """)



        elif option == "Hồi quy đa thức":
            st.header("Hồi quy đa thức (Polynomial Regression)")
            st.markdown("---") 
            st.markdown("""
            ### **Lý thuyết**
            - **Hồi quy đa thức (Polynomial Regression)n**:
                - **Polynomial Regression** là một kỹ thuật hồi quy mở rộng từ **Linear Regression.**
                - Được sử dụng để mô hình hóa mối quan hệ phi tuyến tính giữa một hoặc nhiều biến độc lập (X) và một biến phụ thuộc (Y) thông qua một hàm đa thức (polynomial). 
                - Mặc dù quan hệ giữa 𝑋 và Y là phi tuyến, mô hình vẫn tuyến tính với các tham số (coefficients), do đó được gọi là "hồi quy tuyến tính đa thức."
            - **Công thức**:
            $$
            Y = w_0 + w_1X + w_2X^2 + ... + w_nX^n + \\epsilon
            $$
            **Trong đó:**
            - $$( Y )$$: Biến phụ thuộc (giá trị cần dự đoán).
            - $$( X )$$: Biến độc lập.
            - $$( w_0, w_1, ..., w_n )$$: Hệ số hồi quy.
            - $$( n )$$: Bậc của đa thức.
            - $$( epsilon )$$: Sai số ngẫu nhiên.
            """)
            st.markdown("---") 
            st.markdown("### 📊 **Ví dụ: Dự đoán tốc độ xe**")
            st.markdown("""
            **Dữ liệu mẫu:**
            - Thời gian (X, giây): [0, 2, 4, 6, 8, 10]
            - Tốc độ (Y, km/h): [0, 20, 40, 60, 80]
            """)

            # Dữ liệu mẫu
            X = np.array([0, 2, 4, 6, 8]).reshape(-1, 1)  # Thời gian (reshape để thành 2D)
            Y = np.array([0, 20, 60, 80, 70])  # Tốc độ

            # Tạo các đặc trưng đa thức (bậc 2)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            # Huấn luyện mô hình Linear Regression trên dữ liệu đa thức
            model = LinearRegression()
            model.fit(X_poly, Y)

            # Lấy các hệ số
            w_0 = model.intercept_  # Hệ số chặn
            w = model.coef_  # Hệ số (w_1, w_2, ...)
            w_1, w_2 = w[1], w[2]  # Trích xuất w_1 và w_2 (bậc 2)

            # Chuẩn bị dữ liệu để vẽ
            X_smooth = np.linspace(0, 10, 100).reshape(-1, 1)
            X_poly_smooth = poly.transform(X_smooth)
            Y_pred = model.predict(X_poly_smooth)

            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, Y, color='red', label='Dữ liệu thực tế')
            ax.plot(X_smooth, Y_pred, color='blue', label='Đường cong đa thức (bậc 2)')
            ax.set_xlabel('Thời gian (giây)')
            ax.set_ylabel('Tốc độ (km/h)')
            ax.set_title('Polynomial Regression: Tốc độ xe theo thời gian')
            ax.legend()

            # Hiển thị biểu đồ trên Streamlit
            st.pyplot(fig)

            # In kết quả w
            st.markdown("**Kết quả trọng số (weights):**")
            st.write(f"w_0 = {w_0:.4f}, w_1 = {w_1:.4f}, w_2 = {w_2:.4f}")
            st.markdown("""
            - $$( w_0 )$$: Hệ số chặn.
            - $$( w_1 )$$: Hệ số cho $$( X )$$.
            - $$( w_2 )$$: Hệ số cho $$( X^2 )$$.
            """)

         # ---------------- Huấn luyện & Kiểm thử mô hình ----------------
    with tab_train:
        st.header("Huấn luyện & Kiểm thử mô hình")
        with st.expander("📊 Chia dữ liệu"):
            if st.session_state.get("data_processed", False):
                df = st.session_state.df.copy()
                if "Survived" not in df.columns:
                    st.error("Không tìm thấy cột mục tiêu 'Survived'.")
                else:
                    st.write("**Nhập tỷ lệ chia tập dữ liệu:**")
                    with mlflow.start_run():
                        test_pct = st.slider("** Chọn % tỷ lệ tập test** (%)", 0, 50, 15)
                        valid_pct = st.slider("** Chọn % tỷ lệ tập validation (trong phần train)** (%)", 0, 50, 15)
                        train_pct = 100 - (test_pct + valid_pct)
                        total = test_pct + valid_pct + train_pct
                        st.markdown(f"""
                        Tỷ lệ phân chia bao gồm:
                        - **Train:** {train_pct}%  
                        - **Test:** {test_pct}%  
                        - **Validation:** {valid_pct}%  
                        """)
                        if total != 100:
                            st.warning("Tổng các tỉ lệ phải bằng 100%! Vui lòng điều chỉnh lại các giá trị.")
                        else:
                            if train_pct < 30:
                                st.warning("Tỉ lệ tập Train quá thấp (<30%).")
                            if test_pct < 5:
                                st.warning("Tỉ lệ tập Test quá thấp (<5%).")
                            if valid_pct < 5:
                                st.warning("Tỉ lệ tập Validation quá thấp (<5%).")
                            X = df.drop(columns=["Survived"])
                            y = df["Survived"]  
                            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)
                            valid_size = valid_pct / (valid_pct + train_pct) if (valid_pct + train_pct) > 0 else 0
                            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)
                            st.markdown(f"""
                            Số lượng mẫu sau khi chia:
                            - **Train:** {X_train.shape[0]} mẫu  
                            - **Validation:** {X_val.shape[0]} mẫu  
                            - **Test:** {X_test.shape[0]} mẫu  
                            """)
                    mlflow.end_run()
                    min_samples = 10
                    if X_train.shape[0] < min_samples:
                        st.warning("Số mẫu tập Train quá ít.")
                    if X_val.shape[0] < min_samples:
                        st.warning("Số mẫu tập Validation quá ít.")
                    if X_test.shape[0] < min_samples:
                        st.warning("Số mẫu tập Test quá ít.")
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    st.session_state.X_val = X_val
                    st.session_state.y_val = y_val
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.data_split = True
            else:
                st.warning("Vui lòng xử lý dữ liệu trước.")


        with st.expander("Huấn luyện mô hình"):   
        #<--------------------------------Test---------------------------->
            if st.session_state.get("data_split", False):
                # 1) Chọn mô hình
                col_model, col_model_tip = st.columns([0.8, 0.2])
                with col_model:
                    model_choice_to_train = st.selectbox("Chọn mô hình để huấn luyện:", 
                                                        ["Hồi quy đa biến (Multiple Regression) ", "Hồi quy đa thức (Polynomial Regression) "])
            
                col_lr, col_lr_tip = st.columns([0.8, 0.2])
                with col_lr:
                    lr_method = "constant"  # Gán cố định giá trị "constant"

                col_eta, col_eta_tip = st.columns([0.8, 0.2])
                with col_eta:
                    eta0 = st.number_input("Chọn tốc độ học (learning rate):", 
                            value=0.01, min_value=0.0001, max_value=1.0, 
                            step=0.0001, format="%.4f")
                poly_degree = 1
                if model_choice_to_train == "Hồi quy Đa thức":
                    col_poly, col_poly_tip = st.columns([0.8, 0.2])
                    with col_poly:
                        poly_degree = st.number_input("Chọn bậc của đa thức:", 
                                                    min_value=1, max_value=10, value=2)
                col_fold, col_fold_tip = st.columns([0.8, 0.2])
                with col_fold:
                    num_folds = st.number_input("Chọn số folds (KFold Cross-Validation):", 
                                                min_value=2, max_value=20, value=5, step=1)
                if st.button("Huấn luyện mô hình"):
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    X_val = st.session_state.X_val
                    y_val = st.session_state.y_val
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    with st.spinner("Đang huấn luyện mô hình với Cross Validation..."):
                        # Tự động tạo run_name
                        run_name = f"{model_choice_to_train}_Run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                        with mlflow.start_run(run_name=run_name) as run:
                            # Tham số cố định của mô hình
                            max_iter = 1000
                            tol = 1e-3
                            
                            # Cập nhật các tham số quan trọng vào dictionary
                            params = {
                                "model_choice": model_choice_to_train,
                                "learning_rate_method": lr_method,
                                "max_iter": max_iter,
                                "num_folds": num_folds,
                                "train_samples": X_train.shape[0],
                                "validation_samples": X_val.shape[0],
                                "test_samples": X_test.shape[0]
                            }
                            if lr_method == "constant" and eta0 is not None:
                                params["eta0"] = eta0
                            else:
                                params["eta0"] = "N/A"

                            if model_choice_to_train == "Hồi quy Đa thức":
                                params["poly_degree"] = poly_degree

                            # Log các tham số quan trọng
                            for key, value in params.items():
                                mlflow.log_param(key, value)

                            # Khởi tạo mô hình
                            if model_choice_to_train == "Hồi quy Đa biến":
                                if lr_method == "constant" and eta0 is not None:
                                    model = SGDRegressor(learning_rate=lr_method, eta0=eta0, max_iter=max_iter, tol=tol)
                                else:
                                    model = SGDRegressor(learning_rate=lr_method, max_iter=max_iter, tol=tol)
                            else:
                                if lr_method == "constant" and eta0 is not None:
                                    model = Pipeline([
                                        ('poly', PolynomialFeatures(degree=poly_degree)),
                                        ('sgd', SGDRegressor(learning_rate=lr_method, eta0=eta0, max_iter=max_iter, tol=tol))
                                    ])
                                else:
                                    model = Pipeline([
                                        ('poly', PolynomialFeatures(degree=poly_degree)),
                                        ('sgd', SGDRegressor(learning_rate=lr_method, max_iter=max_iter, tol=tol))
                                    ])

                            # Cross Validation
                            cv_scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='r2')
                            # Log thêm các chỉ số phụ quan trọng từ CV
                            mlflow.log_metric("mean_cv_score", np.mean(cv_scores))
                            mlflow.log_metric("cv_scores_std", np.std(cv_scores))
                            mlflow.log_metric("max_cv_score", np.max(cv_scores))
                            
                            model.fit(X_train, y_train)
                            # Dự đoán trên tập Validation
                            y_pred_val = model.predict(X_val)
                            mse_val = mean_squared_error(y_val, y_pred_val)
                            r2_val = r2_score(y_val, y_pred_val)
                            y_pred_val_class = [1 if pred >= 0.5 else 0 for pred in y_pred_val]
                            accuracy_val = accuracy_score(y_val, y_pred_val_class)
                            # Dự đoán trên tập Test
                            y_pred_test = model.predict(X_test)
                            mse_test = mean_squared_error(y_test, y_pred_test)
                            r2_test = r2_score(y_test, y_pred_test)
                            y_pred_test_class = [1 if pred >= 0.5 else 0 for pred in y_pred_test]
                            accuracy_test = accuracy_score(y_test, y_pred_test_class)

                            # Log các chỉ số đánh giá
                            mlflow.log_metric("validation_mse", mse_val)
                            mlflow.log_metric("validation_r2", r2_val)
                            mlflow.log_metric("validation_accuracy", accuracy_val)
                            mlflow.log_metric("test_mse", mse_test)
                            mlflow.log_metric("test_r2", r2_test)
                            mlflow.log_metric("test_accuracy", accuracy_test)
                            mlflow.sklearn.log_model(model, "model")

                            # Lưu thông tin vào session_state
                            st.session_state["run_id"] = run.info.run_id
                            st.session_state["run_name"] = run_name
                            st.session_state["accuracy_val"] = accuracy_val
                            st.session_state["accuracy_test"] = accuracy_test
                            st.session_state["params"] = params
                            st.session_state["model"] = model
                            st.session_state["models_trained"] = True
                        
                        # Hiển thị kết quả
                        results_df = pd.DataFrame({
                        "Metric": ["Cross Validation Scores (R²)", "Mean CV Score (R²)", "Validation MSE", "Validation R²", "Validation Accuracy", "Test MSE", "Test R²", "Test Accuracy"],
                        "Value": [
                            ', '.join([f"{score:.2e}" for score in cv_scores]),  # Chuỗi các giá trị R²
                            f"{np.mean(cv_scores):.2e}",
                            f"{mse_val:.2e}",
                            f"{r2_val:.2e}",
                            f"{accuracy_val:.2%}",
                            f"{mse_test:.2e}",
                            f"{r2_test:.2e}",
                            f"{accuracy_test:.2%}"
                        ]
                        })
                        # Hiển thị bảng
                        st.markdown("### 📊 Kết quả đánh giá mô hình")
                        st.table(results_df)


                        st.markdown("---") 
                        # Giải thích kết quả ngắn gọn hơn
                        st.markdown("### ℹ️ Giải thích kết quả")
                        st.markdown("""
                        - **Cross Validation Scores (R²):** Đánh giá hiệu suất mô hình trên từng tập huấn luyện (fold).  
                        - **Mean CV Score (R²):** Trung bình R² của các fold, giá trị càng gần 1 càng tốt.  
                        - **MSE (Mean Squared Error):** Sai số bình phương trung bình, càng nhỏ càng tốt.  
                        - **R² (R-squared):** Đo lường độ phù hợp của mô hình, gần 1 là tốt.  
                        - **Accuracy (ngưỡng 0.5):** Tỷ lệ dự đoán đúng khi áp dụng ngưỡng 0.5.  
                        """)
                        
                        metrics = ["Mean CV Score (R²)", "Validation MSE", "Validation R²", "Validation Accuracy", "Test MSE", "Test R²", "Test Accuracy"]
                        values = [np.mean(cv_scores), mse_val, r2_val, accuracy_val, mse_test, r2_test, accuracy_test]

                        # Vẽ biểu đồ cột cho các chỉ số chính
                        # st.markdown("### 📊 **Biểu đồ so sánh các chỉ số chính**")

                        # fig, ax = plt.subplots(figsize=(8, 5))
                        # sns.barplot(x=values, y=metrics, ax=ax, palette="coolwarm")

                        # # Gán nhãn giá trị lên cột
                        # for i, v in enumerate(values):
                        #     ax.text(v, i, f"{v:.2e}", color='black', va='center')

                        # ax.set_xlabel("Giá trị")
                        # ax.set_ylabel("Chỉ số")
                        # ax.set_title("So sánh các chỉ số chính")

                        # st.pyplot(fig)

                        # # Vẽ biểu đồ line cho Cross Validation Scores (R²)
                        # st.markdown("### 📉 **Biểu đồ Cross Validation Scores (R²)**")

                        # fig, ax = plt.subplots(figsize=(8, 4))
                        # sns.lineplot(x=range(len(cv_scores)), y=cv_scores, marker='o', ax=ax, color='blue')
                        # ax.axhline(np.mean(cv_scores), linestyle="--", color="red", label="Mean R²")  # Đường trung bình

                        # ax.set_xlabel("Fold")
                        # ax.set_ylabel("R² Score")
                        # ax.set_title("Cross Validation Scores (R²)")
                        # ax.legend()

                        # st.pyplot(fig)
            else:
                st.warning("Vui lòng chia tập dữ liệu trước.")

    # ---------------- Tab 4: Dự đoán ----------------
    with tab_predict:
        st.header("Demo Dự đoán")
        if st.session_state.get("models_trained", False):
            st.write("Nhập thông tin hành khách:")
            df = st.session_state.df
            features = df.drop(columns=["Survived"]).columns.tolist()
            input_values = []
            for feature in features:
                if np.issubdtype(df[feature].dtype, np.number):
                    default_value = abs(float(df[feature].median()))
                    value = st.number_input(f"{feature}:", value=default_value)
                else:
                    options = list(sorted(df[feature].unique()))
                    value = st.selectbox(f"{feature}:", options)
                input_values.append(value)
            if st.button("Dự đoán"):
                input_array = np.array(input_values).reshape(1, -1)
                prediction = st.session_state.model.predict(input_array)[0]
                result = "Sống" if prediction >= 0.5 else "Không sống"
                st.write(f"**Dự đoán:** {result}")
        else:
            st.warning("Vui lòng huấn luyện mô hình trước.")

   
    # ---------------- Tab 5: Thông tin huấn luyện & MLflow UI ----------------
    with tab_mlflow:
        st.header("Thông tin Huấn luyện & MLflow UI")
        try:
            client = MlflowClient()
            experiment_name = "LinearRegression"
    
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
    run_LinearRegression_app()


    # cd "c:/Users/Dell/OneDrive/Pictures/Documents/Code/python/OpenCV/HMVPYTHON/App" 
