import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title('Ứng dụng dự đoán đơn giản với Streamlit')

st.write("Đây là một ứng dụng Streamlit mẫu để minh họa việc dự đoán với mô hình Machine Learning.")

# Tạo lại dữ liệu mẫu và mô hình (trong ứng dụng Streamlit)
# Trong một ứng dụng thực tế, mô hình sẽ được lưu và tải lại.
np.random.seed(42)
X_streamlit = np.random.rand(100, 1) * 10
y_streamlit = 2 * X_streamlit + 1 + np.random.randn(100, 1) * 2

model_streamlit = LinearRegression()
model_streamlit.fit(X_streamlit, y_streamlit)

st.subheader('Nhập giá trị để dự đoán')

# Input widget
user_input = st.slider('Chọn một giá trị cho Feature', 0.0, 10.0, 5.0)

# Make prediction
prediction = model_streamlit.predict(np.array([[user_input]]))[0][0]

st.write(f'Giá trị Feature bạn đã chọn: **{user_input:.2f}**')
st.write(f'Kết quả dự đoán (Target): **{prediction:.2f}**')

st.subheader('Mô hình đã huấn luyện')
st.write(f'Hệ số (Coefficient): **{model_streamlit.coef_[0][0]:.2f}**')
st.write(f'Hệ số chặn (Intercept): **{model_streamlit.intercept_[0]:.2f}**')
