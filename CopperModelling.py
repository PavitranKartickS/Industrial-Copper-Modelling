import streamlit as st
from streamlit_option_menu import option_menu
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import re

#--------------------------------------------------------------------------------------------------------------------------------------


# Page Configuration
icon = Image.open("icon.jpg")
st.set_page_config(page_title= "Industrial Copper Modelling",
                   page_icon= icon,
                   initial_sidebar_state= "collapsed",
                   layout= "wide",)

st.markdown("<h1 style='text-align: center; color: #FF9549;'>GUVI Capstone Project 5</h1>",unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #FF9549;'>Industrial Copper Modelling</h1>",unsafe_allow_html=True)


selected = option_menu(None, ['Home',"Price Prediction","Status Prediction","Conclusion"],
            icons=["house",'cash-coin','trophy',"check-circle"],orientation='horizontal',default_index=0)



# Home Page
if selected=='Home':

    c1, c2 = st.columns(2)
    with c1:
        st.write('## **PROBLEM STATEMENT:**')
        st.write('* The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.')
        st.write('* This Project utilizes a ML Regression model to predict the :violet[**‘Selling_Price’**] of copper.')
        st.write('* This Project also utilizes a ML Classification model to predict the status of the data point to be either :green[**WON**] or :red[**LOST**].')
        st.write('## Modules and Technlogies used:')
        st.write('* Python \n * Streamlit \n * NumPy \n * Pandas \n * Scikit-learn \n * Matplotlib \n * Seaborn \n * Pickle \n * Streamlit-Option-Menu')
    

    with c2:
            st.write('## **MACHINE LEARNING COMPONENTS :**')

            st.write('#### REGRESSION - ***:orange[DecisionTreeRegressor]***')
            st.write('- A decision tree regressor is a supervised learning algorithm that is used for solving regression problems, where the goal is to predict continuous-valued outputs instead of discrete outputs. It works by creating a tree-like model of decisions based on the features of the input data, with each branch representing a possible decision and the leaves representing the predicted output values.')
            st.write('#### CLASSIFICATION - ***:blue[DecisionTreeClassifier]***')
            st.write('- A decision tree classifier is a type of supervised learning algorithm used for both classification and regression tasks. It has a hierarchical tree structure consisting of a root node, branches, internal nodes, and leaf nodes. The decision tree is built by recursively splitting the data based on the most significant features, with the goal of creating homogeneous groups within each leaf node.')


if selected=='Price Prediction':
     # Define the possible values for the dropdown menus
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    # Define the widgets for user input
    with st.form("my_form"):
        
        st.write(' ')
        status = st.selectbox("Status", status_options,key=1,)
        item_type = st.selectbox("Item Type", item_type_options,key=2)
        country = st.selectbox("Country", sorted(country_options),key=3)
        application = st.selectbox("Application", sorted(application_options),key=4)
        product_ref = st.selectbox("Product Reference", product,key=5)
        st.write("                                                                                 ")           
        st.write( f'<h5 style="color:rgb(173, 250, 169, 1);">NOTE: The minimum and maximum values are for reference and are inclusive</h5>', unsafe_allow_html=True )
        quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
        thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
        width = st.text_input("Enter width (Min:1, Max:2990)")
        customer = st.text_input("customer ID (Min:12458, Max:30408185)")
        submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #16AC40;
                color: white;
                width: 100%;
            }
            </style>
        """, unsafe_allow_html=True)

        flag=0 
        pattern = "^(?:\d+|\d*\.\d+)$"
        for i in [quantity_tons,thickness,width,customer]:             
            if re.match(pattern, i):
                pass
            else:                    
                flag=1  
                break
        
    if submit_button and flag==1:
        if len(i)==0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ",i)  
            
    if submit_button and flag==0:
        
        import pickle
        with open(r"rmodel.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        with open(r'rscaler.pkl', 'rb') as f:
            scaler_loaded = pickle.load(f)

        with open(r"t.pkl", 'rb') as f:
            t_loaded = pickle.load(f)

        with open(r"s.pkl", 'rb') as f:
            s_loaded = pickle.load(f)

        #New Sample Price Prediction:
        new_sample= np.array([[np.log(float(quantity_tons)),application,np.log(float(thickness)),float(width),country,float(customer),int(product_ref),item_type,status]])
        new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
        new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scaler_loaded.transform(new_sample)
        new_pred = loaded_model.predict(new_sample1)[0]
        st.write('## :green[Predicted selling price:] ', np.exp(new_pred))


if selected=='Status Prediction':
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product=['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
    with st.form("my_form1"):
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)") 
                 
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options,key=21)
            ccountry = st.selectbox("Country", sorted(country_options),key=31)
            capplication = st.selectbox("Application", sorted(application_options),key=41)  
            cproduct_ref = st.selectbox("Product Reference", product,key=51)           
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")
            st.markdown("""
                <style>
                div.stButton > button:first-child {
                    background-color: #16AC40;
                    color: white;
                    width: 100%;
                }
                </style>
            """, unsafe_allow_html=True)
    
            cflag=0 
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons,cthickness,cwidth,ccustomer,cselling]:             
                if re.match(pattern, k):
                    pass
                else:                    
                    cflag=1  
                    break
            
            if csubmit_button and cflag==1:
                if len(k)==0:
                    st.write("please enter a valid number, blank space is not allowed")
                else:
                    st.write("You have entered an invalid value: ",k)  
                
            if csubmit_button and cflag==0:
                import pickle
                with open(r"model/cmodel.pkl", 'rb') as file:
                    cloaded_model = pickle.load(file)

                with open(r'model/cscaler.pkl', 'rb') as f:
                    cscaler_loaded = pickle.load(f)

                with open(r"model/ct.pkl", 'rb') as f:
                    ct_loaded = pickle.load(f)

                # New sample Status Prediction:
                new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication, np.log(float(cthickness)),float(cwidth),ccountry,int(ccustomer),int(product_ref),citem_type]])
                new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
                new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,7]], new_sample_ohe), axis=1)
                new_sample = cscaler_loaded.transform(new_sample)
                new_pred = cloaded_model.predict(new_sample)
                #st.write(new_pred)
                if new_pred.all()==1:
                    st.write('## :green[The Status is Won] ')
                else:
                    st.write('## :red[The status is Lost] ')


if selected == 'Conclusion': 
    st.markdown("## :orange[**SUMMARY :**]")  

    st.write('### From this Project, we are able to study the factors affecting the price and status of industrial copper :')
    st.write("* Copper was one of the first metals ever extracted and used by humans, and it has made vital contributions to sustaining and improving society since the dawn of civilization. Increased demand for copper typically indicates a growing economy, just as a drop in copper demand can suggest an economic slowdown")     
    st.write('* :red[**Infrastructure**] development such as Railways, Electrical grids, Telecommunications ,Water supply, Healthcare and construction rely on copper due to its efficiency and performance.')      
    st.write('* Global Copper production and consumption has reached an all time high of   :orange[**20 million metric tons**]. ')
    st.write('* Global Copper trade has its fair share  of complications when it comes to :red[**illegal copper cartel**] in developing nations such as South America, where in Chile is the highest copper producer. Such global phenomenon can greatly affect market prices for copper trade.  ')
    st.write('* With the use of Advanced :violet[**Machine Learning**] techniques we can equip ourselves with the tools necessary to combat growing demands and fluctuating trends in the global copper market.  ')