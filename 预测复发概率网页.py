import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载训练好的逻辑回归模型
model = joblib.load('logistic_regression_model.joblib')
# 创建 Streamlit 应用标题和描述
st.title('Cure Rate Prediction App')
st.write('Please Enter Value for Below Parameter')
# 用户输入
pth = st.number_input('PTH(pg/mL):', min_value=0.0, max_value=10000.0, value=3.0, step=0.1)
sex = st.selectbox('Sex:', [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
phosphorus = st.number_input('Phosphorus(mmol/L):', min_value=0.0, max_value=10000.0, value=3.5, step=0.1)
# 创建输入数据框
scaler = joblib.load('scaler.save')
input_data = pd.DataFrame({
    'PTH': [pth],
    'Sex': [sex],
    'Phosphorus': [phosphorus],
    "Symptom": [1],
    "Volume": [1.01],
    "N. of nodules": [1],
    "Calcium": [2.09],
    "ALP": [96],
    "Maximal diameter": [1.3],
    "Age": [62],
})
input_data = input_data[['Sex', 'Age', 'Symptom', 'Volume', 'N. of nodules', 'Maximal diameter', 'PTH', 'Calcium', 'Phosphorus', 'ALP']]
input_data_scaler = scaler.transform(input_data)
input_data_scaler = pd.DataFrame(input_data_scaler,columns = input_data.columns)
input_data_scaler = input_data_scaler[["PTH","Sex","Phosphorus"]]
# 预测并显示结果
if st.button('Predict'):
    prediction_proba = model.predict_proba(input_data_scaler)[0][1]  # 获取复发概率
    prediction_percentage = f'{prediction_proba * 100:.2f}%'
    st.write(f'The Probability of Cure is : {prediction_percentage}')

