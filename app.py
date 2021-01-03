from pycaret.regression import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost
import shap
import streamlit.components.v1 as components
from PIL import Image

model = load_model('deployment_30112020')
@st.cache
def predict(model, input_df):
    prediction_df = predict_model(estimator=model, data=input_df)
    prediction = prediction_df['Label'][0]
    return prediction

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
    

def ABS_SHAP(df_shap,df):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")

#main function
@st.cache
def main():
    
    st.sidebar.info('This app is created to predict CO2 Solubility in Brine')
    st.sidebar.success('https://www.pycaret.org')
    
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", 
        ("Single value prediction", "Multiple value prediction"))
    
    st.title("CO2 Solubility in Brine Prediction App")
    st.subheader("Created by: Khoirrashif")
    image_CCS = Image.open('CCS.jpg')
    st.image(image_CCS, use_column_width=False)
    st.text("(Image source: Global CCS Institute)")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    

    
##Single value Prediction
    if add_selectbox == 'Single value prediction':
    
        mNaCl = st.number_input('mNaCl (mol/kg) | min = 0.016 mol/kg, max = 6.14 mol/kg',value=3.25, min_value=0.016, max_value=6.14) #input mNaCl
        
        Pressure = st.number_input('Pressure (bar) | min = 0.98 bar, max = 1400.00 bar',value=500.00, min_value=0.98, max_value=1400.00) #input Pressure
        
        Temperature = st.number_input('Temperature (K) | min = 273.15 K, max = 723.15 K', value=425.00,min_value=273.15, max_value=723.15) #input Temperature
        
    
        output=""
    
        input_dict = {'mNaCl (mol/kg)': mNaCl, 'Pressure (bar)': Pressure, 'Temperature (K)': Temperature}
        input_df = pd.DataFrame([input_dict])
    
        if st.button("Predict"):
            output = predict(model = model, input_df = input_df)
            output = str(output) + 'mol/kg'
    
        st.success('The CO2 solubility is {}'.format(output))

    
##Multiple value Prdiction
    if add_selectbox == 'Multiple value prediction':
    
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
    
        if file_upload is not None:
            data = pd.read_csv(file_upload)   
            prediction = predict_model(estimator=model, data=data)
            st.write(prediction)
            
            shap.initjs()

            # train catBoost model
            X,y = data, prediction['Label']
            mod = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
    
            # explain the model's predictions using SHAP
            # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
            explainer = shap.TreeExplainer(mod)
            shap_values = explainer.shap_values(X)
            
            st.title("Feature Importance and Prediction Explanation based on the SHAP values")
            st.write("For a complete explanation about SHAP (SHapley Additive exPlanations) values and their impacts on machine learning models interpretability please refer to  Lundberg and Lee (2016), and their GitHub (https://github.com/slundberg/shap/blob/master/README.md)")
            
            st.header("Total distribution of observations based on the SHAP values, colored by Target Value")
            st.write("The plot below sorts features by the sum of SHAP value magnitudes all over samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The colour represents the feature value (e.g: red shows high impact, while blue shows negative impact.")
            #st.write("and uses SHAP values to show the distribution of the impacts each feature has on the model output.")
            #st.write("The colour represents the feature value (e.g: red shows high impact, while blue shows negative impact.")
            shap.summary_plot(shap_values, X)
            st.pyplot()
            plt.clf()
            
            st.header("Feature Importance according to the SHAP values (simplified plot)")
            st.write("The following plot is the simplified version of the plot above which is basically built by taking the mean absolute value of the SHAP value for each feature. It also shows the feature importance in descending order and highlights the correlation in colours.")
            #st.write("for each feature to get a standard bar plot.")
            ABS_SHAP(shap_values, X)
            st.pyplot()
            plt.clf()
            
            st.header("Prediction explanation for a single observation")
            st.write("The following plots are the Individual Force Plots. Each of them shows how each feature affects the model output from the base value for a single prediction. Features pushing the prediction higher are shown in red, while those pushing the prediction lower are in blue. A set of samples are provided below from the 3rd, 7th, and 10th observation from the dataset.")
            st.subheader("Example on the 3rd observation")
            shap.force_plot(explainer.expected_value, shap_values[3,:], X.iloc[3,:], matplotlib=True, show=False, figsize=(16,5))
            st.pyplot()
            plt.clf()
            
            st.subheader("Example on the 7th observation")
            shap.force_plot(explainer.expected_value, shap_values[7,:], X.iloc[7,:], matplotlib=True, show=False, figsize=(16,5))
            st.pyplot()
            plt.clf()
            
            st.subheader("Example on the 10th observation")
            shap.force_plot(explainer.expected_value, shap_values[10,:], X.iloc[10,:], matplotlib=True, show=False, figsize=(16,5))
            st.pyplot()
            plt.clf()
            
            st.header("Prediction explanation for the entire dataset")
            st.write("The plot below is the Collective Force Plot. It is built by rotating the individual force plot 90 degrees, and stack them horizontally for the entire dataset.")
            st_shap(shap.force_plot(explainer.expected_value, shap_values, X), 400)
            
            st.header("Dependence plots for each feature")
            st.write("The following plots are the partial dependence plots which each of them shows the marginal effect one or two features have on the predicted outcome of a machine learning model (J.H. Friedman, 2001). The partial dependence plot tells wether the relationship between the target and a feature is linear, monotonic or more complex.")
            st.subheader("Pressure")
            shap.dependence_plot("Pressure (bar)",shap_values,X,show=False)
            st.pyplot()
            plt.clf()
            
            st.subheader("Temperature")
            shap.dependence_plot("Temperature (K)",shap_values,X,show=False)
            st.pyplot()
            plt.clf()
            
            st.subheader("mNaCl")
            shap.dependence_plot("mNaCl (mol/kg)",shap_values,X,show=False)
            st.pyplot()
            plt.clf()
            
            
              
            
if __name__ == "__main__":
  main()
