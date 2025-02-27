import streamlit as st
import pandas as pd
import numpy as np
# import mlflow
# import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

# Preprocess data
def preprocess_data(df):
    df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return X, y

X, y = preprocess_data(df)

# User input for data split
st.sidebar.header("Tùy chỉnh tỷ lệ dữ liệu")
train_size = st.sidebar.slider("Tập train (%)", 50, 80, 70)
valid_size = st.sidebar.slider("Tập valid (%)", 10, 30, 15)
test_size = 100 - train_size - valid_size

st.sidebar.write(f"Train: {train_size}%  |  Valid: {valid_size}%  |  Test: {test_size}%")

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(valid_size+test_size)/100, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(valid_size+test_size), random_state=42)

# User selects regression type
regression_type = st.sidebar.radio("Chọn loại hồi quy", ('Multiple Regression', 'Polynomial Regression'))
degree = st.sidebar.slider("Bậc của hồi quy đa thức", 2, 5, 2) if regression_type == 'Polynomial Regression' else None

# Khái niệm về hồi quy
st.write("## Khái niệm về hồi quy")

# Multiple Regression information
if regression_type == 'Multiple Regrßession':
    with st.expander("Hồi quy tuyến tính bội (Multiple Regression)"):
        st.write("Hồi quy tuyến tính bội là mô hình hồi quy tuyến tính mở rộng cho nhiều biến độc lập. Công thức tổng quát:")
        st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon")
        st.write("Trong đó, β0 là hằng số, β1 đến βn là hệ số hồi quy, x1 đến xn là các biến độc lập.")
        
        # Visualization: Scatter Plot & Regression Line
        st.write("### Biểu đồ hồi quy tuyến tính bội")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x=X['Age'], y=y, ax=ax, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
        ax.set_title("Biểu đồ hồi quy tuyến tính bội giữa Age và Survived")
        ax.set_xlabel("Age")
        ax.set_ylabel("Survived")
        st.pyplot(fig)
        
        st.write("### Giải thích:")
        st.write("Biểu đồ trên thể hiện mối quan hệ giữa biến 'Age' và 'Survived'. Đường hồi quy đỏ là kết quả của mô hình hồi quy tuyến tính bội. Mối quan hệ giữa các biến độc lập sẽ được tính toán tương tự.")

# Polynomial Regression information
if regression_type == 'Polynomial Regression':
    with st.expander("Hồi quy đa thức (Polynomial Regression)"):
        st.write("Hồi quy đa thức là một dạng mở rộng của hồi quy tuyến tính, trong đó biến độc lập có thể có bậc cao hơn 1 để mô hình hóa quan hệ phi tuyến.")
        st.latex(r"y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3 + ... + \beta_nx^n + \epsilon")
        st.write("Bậc của mô hình hồi quy càng cao, khả năng mô hình hóa mối quan hệ phi tuyến càng tốt, nhưng có thể dễ bị overfitting.")
        
        # Visualization: Polynomial Regression Curve
        st.write("### Biểu đồ hồi quy đa thức")
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X[['Age']])
        
        # Train the polynomial regression model
        poly_regressor = LinearRegression()
        poly_regressor.fit(X_poly, y)
        
        # Create predictions for plotting
        X_range = np.linspace(X['Age'].min(), X['Age'].max(), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        y_range_pred = poly_regressor.predict(X_range_poly)
        
        # Plot the polynomial regression
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X['Age'], y, color='blue', label='Dữ liệu thật')
        ax.plot(X_range, y_range_pred, color='red', label=f"Đường hồi quy bậc {degree}")
        ax.set_title(f"Biểu đồ hồi quy đa thức bậc {degree} giữa Age và Survived")
        ax.set_xlabel("Age")
        ax.set_ylabel("Survived")
        ax.legend()
        st.pyplot(fig)
        
        st.write(f"### Giải thích:")
        st.write(f"Biểu đồ trên thể hiện mối quan hệ giữa 'Age' và 'Survived'. Đường cong đỏ là mô hình hồi quy đa thức bậc {degree}. Với hồi quy đa thức, đường cong có thể uốn lượn để phản ánh mối quan hệ phi tuyến giữa các biến.")



# MLFlow tracking
# mlflow.set_experiment("Titanic Survival Prediction")
# with mlflow.start_run():
if regression_type == 'Multiple Regression':
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
else:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree)),
        ('regressor', LinearRegression())
    ])
    
    # Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    # mlflow.log_param("Model Type", regression_type)
    # if degree:
    #     mlflow.log_param("Polynomial Degree", degree)
    # mlflow.log_metric("Cross-Validation R2", scores.mean())
    
    # # Train final model
model.fit(X_train, y_train)
    # mlflow.sklearn.log_model(model, "Titanic_Model")
    
    # Evaluate
train_score = model.score(X_train, y_train)
valid_score = model.score(X_valid, y_valid)
test_score = model.score(X_test, y_test)
    
    # mlflow.log_metric("Train R2", train_score)
    # mlflow.log_metric("Valid R2", valid_score)
    # mlflow.log_metric("Test R2", test_score)
    
    # Visualization
st.write(f"### Kết quả mô hình {regression_type}")
st.write(f"- Train R²: {train_score:.4f}")
st.write(f"- Valid R²: {valid_score:.4f}")
st.write(f"- Test R²: {test_score:.4f}")


data_selection = st.selectbox("Chọn dữ liệu để phân tích:", ["Train", "Valid", "Test"])
    # Cập nhật dữ liệu dựa trên lựa chọn của người dùng
if data_selection == "Train":
    X_data = X_train
    y_data = y_train
elif data_selection == "Valid":
    X_data = X_valid
    y_data = y_valid
else:
    X_data = X_test
    y_data = y_test

    # Kiểm tra dữ liệu đã chọn
st.write(f"Dữ liệu hiện tại đang sử dụng: {data_selection}")
st.write(f"Số lượng dữ liệu: {len(y_data)}")

    # Tạo biểu đồ cho dữ liệu đã chọn
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 24))  # Sắp xếp theo chiều dọc

    # Biểu đồ phân bố của dữ liệu (sử dụng y_data)
sns.histplot(y_data, bins=10, kde=True, ax=ax[0])
ax[0].set_title(f"Phân bố của dữ liệu {data_selection}", fontsize=25)
ax[0].set_xlabel("Survived", fontsize=18)
ax[0].set_ylabel("Số lượng", fontsize=18)

    # Biểu đồ dự đoán vs thực tế
sns.scatterplot(x=y_data, y=model.predict(X_data), ax=ax[1])
ax[1].set_title(f"Dự đoán với Thực tế ({data_selection})", fontsize=25)
ax[1].set_xlabel("Giá trị thực", fontsize=18)
ax[1].set_ylabel("Giá trị dự đoán", fontsize=18)

    # Biểu đồ phân phối dữ liệu theo các biến (Pclass, Age, Fare) cho tập dữ liệu đã chọn
sns.boxplot(data=df[['Pclass', 'Age', 'Fare']], ax=ax[2])
ax[2].set_title(f"Phân phối dữ liệu theo các biến ({data_selection})", fontsize=25)

    # Chỉ gọi `st.pyplot()` một lần duy nhất
st.pyplot(fig)

    # Phần giải thích
st.write("### Giải thích:")
st.write("🔹 **Phân bố dữ liệu:** Biểu đồ trên thể hiện phân bố của dữ liệu đã chọn theo biến `Survived`. Nó giúp quan sát số lượng hành khách sống sót (1) và không sống sót (0). Đường cong KDE cho thấy mật độ phân bố của dữ liệu.")
st.write("🔹 **Dự đoán với Thực tế:** Biểu đồ so sánh giá trị dự đoán của mô hình với giá trị thực tế. Nếu các điểm nằm trên đường chéo, mô hình dự đoán chính xác. Nếu phân tán quá nhiều, mô hình có thể chưa phù hợp.")
st.write("🔹 **Phân phối dữ liệu:** Biểu đồ trên thể hiện phân bố giá trị của các biến như Pclass, Age, Fare.")


if regression_type == 'Multiple Regression':
    st.write("### Kết quả mô hình Hồi quy tuyến tính bội")
        
        # Train the model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
        
        # Fit the model with the training data
    model.fit(X_train, y_train)
        
        # Predict on train, validation, and test data
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)
        
        # Plot the Linear Regression (for Train, Valid, Test)
    fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot train, valid, and test predictions
    ax.scatter(y_train, y_train_pred, color='blue', label='Train Data', alpha=0.6)
    ax.scatter(y_valid, y_valid_pred, color='green', label='Validation Data', alpha=0.6)
    ax.scatter(y_test, y_test_pred, color='red', label='Test Data', alpha=0.6)
        
        # Plot regression line
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label="Perfect Prediction Line")
        
    ax.set_title("Hồi quy tuyến tính bội: Dự đoán với Thực tế (Train, Valid, Test)", fontsize=16)
    ax.set_xlabel("Giá trị thực tế", fontsize=14)
    ax.set_ylabel("Giá trị dự đoán", fontsize=14)
    ax.legend()
        
    st.pyplot(fig)
        
        # Explanation
    st.write("### Giải thích:")
    st.write("🔹 **Dữ liệu Train (Xanh Dương)**: Dự đoán trên tập huấn luyện.")
    st.write("🔹 **Dữ liệu Valid (Xanh Lá)**: Dự đoán trên tập kiểm tra (tập valid).")
    st.write("🔹 **Dữ liệu Test (Đỏ)**: Dự đoán trên tập kiểm tra (tập test).")
    st.write("🔹 **Đường Chấm Đen**: Đường hồi quy lý tưởng, mô tả mối quan hệ giữa giá trị thực tế và giá trị dự đoán.")
    st.write("🔹 Biểu đồ thể hiện mối quan hệ giữa giá trị thực tế và giá trị dự đoán trên các tập train, valid, và test. Nếu các điểm dữ liệu gần với đường chấm đen, mô hình dự đoán chính xác.")

    

    # Hiển thị kết quả của mô hình Polynomial Regression
if regression_type == 'Polynomial Regression':
    st.subheader(f"Kết quả hồi quy đa thức bậc {degree}")

        # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X[['Age']], y, test_size=0.2, random_state=42)
        
        # Tạo đặc trưng đa thức
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

        # Huấn luyện mô hình
    poly_regressor = LinearRegression()
    poly_regressor.fit(X_train_poly, y_train)

        # Dự đoán trên tập kiểm tra
    y_pred = poly_regressor.predict(X_test_poly)

        # Tính toán lỗi RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    st.write(f"🔹 **Lỗi RMSE trên tập kiểm tra:** {rmse:.4f}")

        # Hiển thị biểu đồ thực tế vs. dự đoán
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_test, y_test, color='blue', label='Thực tế', alpha=0.6)
    ax.scatter(X_test, y_pred, color='red', label='Dự đoán', alpha=0.6)
        
        # Vẽ đường hồi quy đa thức
    X_range = np.linspace(X['Age'].min(), X['Age'].max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = poly_regressor.predict(X_range_poly)
    ax.plot(X_range, y_range_pred, color='green', label=f"Đường hồi quy bậc {degree}", linewidth=2)

    ax.set_title(f"Hồi quy đa thức bậc {degree}: Dự đoán với Thực tế")
    ax.set_xlabel("Age")
    ax.set_ylabel("Survived")
    ax.legend()
    st.pyplot(fig)

    st.write("📌 **Giải thích:**")
    st.write(f"🔹 **Đường màu xanh**: là giá trị thực tế từ tập kiểm tra.")
    st.write(f"🔹 **Đường màu đỏ**: là giá trị mô hình dự đoán.")
    st.write(f"🔹 **Đường màu xanh lá**: là đường hồi quy đa thức bậc {degree} khớp với dữ liệu.")




# Prediction demo
st.sidebar.subheader("Dự đoán thử nghiệm")
input_data = np.array([
    st.sidebar.number_input("Pclass", 1, 3, 2),
    st.sidebar.number_input("Age", 1, 100, 30),
    st.sidebar.number_input("SibSp", 0, 10, 1),
    st.sidebar.number_input("Parch", 0, 10, 0),
    st.sidebar.number_input("Fare", 0.0, 500.0, 50.0)
]).reshape(1, -1)

if st.sidebar.button("Dự đoán sống sót"):
    prediction = model.predict(input_data)
    st.sidebar.write(f"Dự đoán sống sót: {'Có' if prediction[0] > 0.5 else 'Không'}")





# # # cd "C:\Users\Dell\OneDrive\Pictures\Documents\Code\python\OpenCV\HMVPYTHON\BaiThucHanh2"
# streamlit run app.py





## nạp dữ liệu lên
## tải dữ liệu (5 giá trị)
## kiểm tra dữ liệu ( có bị lỗi hoặc thiếu ,trực quan hóa dữ liệu
#  )
## nếu thiếu thì bổ sung , nếu lỗi thì in ra và xử lí ở tiền xử lí dữ liệu
## tiền xử lí dữ liệu (  xử lí giá trị thiếu, điền các giá trị thiếu cho các cột, giá trị nhiễu , xóa dữ liệu , 
#                       loại bỏ các cột không cần thiết, xoá các cột đã chọn , các cột đã bị trùng lập , 
#                       chuyển đổi kiểu dữ liệu ví dụ nam nam là 1 nữ là 0, chuẩn hóa dữ liệu số (tất cả),
## sau khi xử lí dữ liệu xong thì sẽ cho chọn các giá trị train , test , val.
## huấn luyện mô hình Multiple / Polynomial Regression và xuâts ra các thông tin cũng như biểu đồ (có giải thích)
## demo không để bên trái mà để ngay sau khi đã chọn dữ liệu xong
## thêm độ chính xác khi nhận diện demo
## thêm phần logging




##note: dữ liêụ sau khi tiền xử lí xong thì sẽ được sử dụng tiếp
