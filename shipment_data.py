import pandas as pd
import numpy as np
#machine learning
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import streamlit as st
#plotting
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mtick
import plotly.tools as tls
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
from PIL import Image #for images

# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler

#load data and save in cache for quick lookup
st.set_page_config(layout="wide")
@st.cache(persist=True)
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/LeenMo/ShipmentViz/main/shipment.csv')
    data.rename(columns={'Reached.on.Time_Y.N': 'On_time'}, inplace=True)
    data.loc[(data['Discount_offered']<5), 'Discount_offered']=0
    return data

raw_data = load_data()

#data.info()    #examine the dataframe; check for missing values
#some missing values for gender and product importance
#remove missing values since they are about 40 rows only

#drop rows with na from gender
data = raw_data.dropna(subset=['Gender', 'Product_importance'])

#convert categorical columns into numeric
cat_cols = ['Warehouse_block', 'Mode_of_Shipment','Gender' ]
dummies= pd.get_dummies(data[cat_cols])
data= data.join(dummies)


#encode product importances using ordinal encoder
ordinal_encoding = OrdinalEncoder(categories=[['low', 'medium','high']])
data['Product_importance_cat'] = data['Product_importance']
data[['Product_importance']] = ordinal_encoding.fit_transform(data[['Product_importance']])



navigate = st.sidebar.selectbox(
    "Navigate to",
    ("Home Page", "Get Insights",  "Predict Shipping Time"))

if navigate=="Home Page":

	# LAYING OUT THE TOP SECTION OF THE APP
	row1_1, row1_2 = st.beta_columns((2,3))

	with row1_1:
		st.markdown("# *Shipping* Insights for $~~~~~~$*E-commerce* Platform ")
    
	with row1_2:
		st.markdown("> Analyzing the shipping data of our growing international E-commerce company is essential to evaluate the overall performance and discover areas of improvement.")
		st.markdown(">Navigate through the sidebar for different actions.")

	#insert pic

	st.image ('https://www.x-cart.com/wp-content/uploads/2019/05/how_2_choose_shipping_carrier-1.jpg')



if navigate=="Get Insights":
	
	row1_1, row1_2, row1_3, row1_4= st.beta_columns((0.4,2,0.2,3))

	with row1_2:
		st.markdown("# Overview of Shipping Performance ")
    
	with row1_4:
		st.markdown("### Examine key factors affecting Shipment Performance")
		st.markdown(">  Delays, Discounts, Mode of shipment, Product's level of importance, Parcel weight in grams, and Warehouse block")

	row2_1, row2_2, row2_3, row2_4 = st.beta_columns((1,1,1,1))

	with row2_3:
		go_to_overview = st.button('Overview')

	with row2_4:
		go_to_delays = st.button('Drill Down To Delays')

	if go_to_overview:

		data_discount= data.copy()
		data_discount.loc[(data_discount['Customer_rating']==0) | (data_discount['Customer_rating']==1) , 'Customer_rating_cat']='BelowAverage'
		data_discount.loc[(data_discount['Customer_rating']==2) | (data_discount['Customer_rating']==3) , 'Customer_rating_cat']='Average'
		data_discount.loc[(data_discount['Customer_rating']==4) | (data_discount['Customer_rating']==5) , 'Customer_rating_cat']='AboveAverage'
		data_discount.loc[(data_discount['On_time']==0), 'On_time']='OnTime'
		data_discount.loc[(data_discount['On_time']==1), 'On_time']='Delayed'

		mean_time_disc_df = data_discount.groupby('On_time', as_index=False)['Discount_offered'].mean()
		count_time_ship_df = data_discount.groupby('Mode_of_Shipment', as_index=False)['On_time'].count()
		mean_disc_rating = data_discount.groupby('Customer_rating_cat', as_index=False)['Discount_offered'].mean()
		mean_calls_rating = data_discount.groupby('Customer_rating_cat', as_index=False)['Prior_purchases'].mean()
		mean_time_weight_df = data_discount.groupby('On_time', as_index=False)['Weight_in_gms'].mean()
		count_time_importance_df = data_discount.groupby('Product_importance_cat', as_index=False)['On_time'].count()
		count_gender_df = data_discount.groupby('Warehouse_block', as_index=False)['ID'].count()


		row2_1, row2_2 = st.beta_columns((0.1,2))

		with row2_2:
			fig8 = make_subplots(rows=4, cols=2, 
								subplot_titles=("Customer Rating and Discounts", "Busiest Warehouse Blocks", "Shipping Delays and Discounts","Weight and Delays","Product Importance and Delays", 'Discounts and Delays',"Mode of Shipment and Delays"),
								vertical_spacing = 0.05, horizontal_spacing = 0.1,
								specs=[[{}, {"type": "pie"}], [{"colspan": 2}, None], [{}, {}] , [{}, {}] ],

								)
			fig8.add_trace(
				go.Bar(x=mean_disc_rating['Customer_rating_cat'], y=mean_disc_rating['Discount_offered'], marker_color=['#00CC94','#636EFA','#EF553B' ],showlegend=False, width=[0.6, 0.6, 0.6]),
				row=1, col=1
				)

			fig8.add_trace(go.Pie( values=count_gender_df['ID'], labels=count_gender_df['Warehouse_block'],marker=dict(colors=['#00CC94','#00CC94','#636EFA','#EF553B','#636EFA']), hole=0.5,showlegend=False),
	              row=1, col=2)


			fig8.add_trace(
				go.Bar(x=mean_time_disc_df['Discount_offered'], y=mean_time_disc_df['On_time'], marker_color=['#EF553B','#00CC94'],showlegend=False,width=[0.6, 0.6, 0.6], orientation='h'),
				row=2, col=1
				)
			

			fig8.add_trace(
				go.Bar(x=mean_time_weight_df['On_time'], y=mean_time_weight_df['Weight_in_gms'], marker_color=['#EF553B','#00CC94'],showlegend=False, width=[0.4, 0.4]),
				row=3, col=1
				)


			fig8.add_trace(
				go.Scatter(x=count_time_importance_df['Product_importance_cat'],y=count_time_importance_df['On_time'],
				mode="markers+text",text=count_time_importance_df['Product_importance_cat'], textposition="top center",showlegend=False),
				row=3, col=2
				)

			fig8.add_trace(
				go.Bar(x=mean_time_disc_df['On_time'], y=mean_time_disc_df['Discount_offered'], marker_color=['#EF553B','#00CC94'],width=[0.4, 0.4], showlegend=False),
				row=4, col=1
				)

			fig8.add_trace(
				go.Scatter(x=count_time_ship_df['Mode_of_Shipment'],y=count_time_ship_df['On_time'],
				mode="markers+text",text=count_time_ship_df['Mode_of_Shipment'], textposition="top center",showlegend=False),
				row=4, col=2
				)


			fig8.update_layout(height=1100, width=980, template = 'plotly',)
			st.write(fig8)

		row4_1, row4_2 = st.beta_columns((0.3,2))

		with row4_2:
			st.markdown("> To delve deeper into the **causes** of delays, navigate to 'Causes of Delay' in the sidebar")

	if go_to_delays:
	
	
		data_delay= data.copy()
		data_delay.loc[(data_delay['On_time']==0), 'On_time']='OnTime'
		data_delay.loc[(data_delay['On_time']==1), 'On_time']='Delayed'
		mean_time_disc_df = data_delay.groupby('On_time', as_index=False)['Discount_offered'].mean()
		mean_time_weight_df = data_delay.groupby('On_time', as_index=False)['Weight_in_gms'].mean()
		count_time_importance_df = data.groupby('Product_importance_cat', as_index=False)['On_time'].count()
		mean_time_cost_df = data_delay.groupby('On_time', as_index=False)['Cost_of_the_Product'].mean()
		mean_time_calls_df = data_delay.groupby('On_time', as_index=False)['Customer_care_calls'].mean()
		count_time_ship_df = data.groupby('Mode_of_Shipment', as_index=False)['On_time'].count()
		count_time_warehouse_df = data.groupby('Warehouse_block', as_index=False)['On_time'].count()

		#create a dataframe with maximum average value of the delays/ontimes
		max_mean_discount = mean_time_disc_df.loc[mean_time_disc_df['Discount_offered'] == mean_time_disc_df['Discount_offered'].max()] 
		#get shipping method with highest delays
		delayed_ship_method = count_time_ship_df.loc[count_time_ship_df['On_time'] == count_time_ship_df['On_time'].max()] 
		#get parcel importance level with highest delays
		max_delayed_imp_level = count_time_importance_df.loc[count_time_importance_df['On_time'] == count_time_importance_df['On_time'].max()]
		#get warehouse with highest delays
		max_delayed_block = count_time_warehouse_df.loc[count_time_warehouse_df['On_time'] == count_time_warehouse_df['On_time'].max()]
		#get max average value of the delays/ontimes
		max_mean_calls = mean_time_calls_df.loc[mean_time_calls_df['Customer_care_calls'] == mean_time_calls_df['Customer_care_calls'].max()] 
		#get max average cost of items with delays/ontimes
		max_mean_cost = mean_time_cost_df.loc[mean_time_cost_df['Cost_of_the_Product'] == mean_time_cost_df['Cost_of_the_Product'].max()] 
		#get max average parcel weight of the delays/ontimes
		max_mean_weight = mean_time_weight_df.loc[mean_time_weight_df['Weight_in_gms'] == mean_time_weight_df['Weight_in_gms'].max()] 
		

		row1_1, row1_2 = st.beta_columns((0.15,2))

		with row1_2:
			st.markdown("# *Causes* and *Implications* of *Delays* in Shipment ")
	    
		fig7 = make_subplots(rows=2, cols=3, 
								subplot_titles=("Average Discounts Offered", "Average Parcel Weight(g)", "Average Cost of The Product", "Average Customer Care Calls","Product Importance Level","Shipment Mode"),
								vertical_spacing = 0.15,

								)		
		fig7.add_trace(
				
			go.Bar(x=mean_time_disc_df['On_time'], y=mean_time_disc_df['Discount_offered'], marker_color=['#EF553B','#00CC94'],showlegend=False, width=[0.6, 0.6, 0.6]),
			row=1, col=1
			)

		fig7.add_trace(
			go.Bar(x=mean_time_weight_df['On_time'], y=mean_time_weight_df['Weight_in_gms'], marker_color=['#EF553B','#00CC94'],showlegend=False,width=[0.6, 0.6, 0.6]),
			row=1, col=2
			)

		fig7.add_trace(
			go.Bar(x=mean_time_cost_df['On_time'], y=mean_time_cost_df['Cost_of_the_Product'], marker_color=['#EF553B','#00CC94'],showlegend=False,width=[0.6, 0.6, 0.6]),
			row=1, col=3
			)

		fig7.add_trace(
			go.Bar(x=mean_time_calls_df['On_time'], y=mean_time_calls_df['Customer_care_calls'], marker_color=['#EF553B','#00CC94'],showlegend=False,width=[0.6, 0.6, 0.6]),
			row=2, col=1
			)

		fig7.add_trace(
			go.Scatter(x=count_time_ship_df['Mode_of_Shipment'],y=count_time_ship_df['On_time'],
				mode="markers+text",text=count_time_ship_df['Mode_of_Shipment'], textposition="top center",showlegend=False),
			row=2, col=3
			)
		fig7.add_trace(
			go.Scatter(x=count_time_importance_df['Product_importance_cat'],y=count_time_importance_df['On_time'],
				mode="markers+text",text=count_time_importance_df['Product_importance_cat'], textposition="top center",showlegend=False),
			row=2, col=2
			)



		fig7.update_layout(height=760, width=1000, template = 'plotly',)
		#,mode="markers+text",text=mean_time_importance_df['Product_importance_cat']
		st.write(fig7)

		cap_row1_1, cap_row1_2, cap_row1_3, cap_row1_4,cap_row1_5, cap_row1_6 = st.beta_columns((0.15,0.7,0.05,0.7,0.05,0.8))

		with cap_row1_2:
			#print if discounted shipments are delayed or on time most of thetime
			st.markdown('### Based on the average discount on products')
			disc= np.floor(max_mean_discount.iloc[0]['Discount_offered'])
			if max_mean_discount.iloc[0]['On_time'] == 'Delayed':
				st.markdown('> Discounted products are being **delayed**. A possible approach to solve this is **increasing the expected delivery times** during sales seasons.')
			else:
				st.markdown('> Discounted items are **not being delayed**; shipments during sales periods are **running smoothly**.')

		with cap_row1_4:
			#weight
			st.markdown('### Based on the average weight of parcels')
			if max_mean_weight.iloc[0]['On_time'] != 'Delayed':
				st.markdown("> **Lighter** parcels are being **delayed**, thus further examination of the root cause is necessary.")
			else: st.markdown("> **Heavier** parcels are being **delayed**.")

		with cap_row1_6:
			#cost
			st.markdown('### Based on the average cost of products')
			if max_mean_cost.iloc[0]['On_time'] != 'Delayed':
				st.markdown("> The cost of a product is not a determining factor in delays, but more expensive products are being **shipped on time** compared to less expensive products.")
			else: st.markdown("> The cost of a product is not a determining factor in delays, but more expensive products are being **delayed** compared to less expensive products.")

		cap_row2_1, cap_row2_2, cap_row2_3, cap_row2_4,cap_row2_5, cap_row2_6 = st.beta_columns((0.15,0.7,0.05,0.7,0.05,0.8))


		with cap_row2_2:
			#calls
			st.markdown('### Based on average customer-care calls')
			if max_mean_calls.iloc[0]['On_time'] == 'Delayed':
				st.markdown('> More Customers are calling the warehouses when there are delays in shipment, which might be **burdening** our customer-care employees and **jeopardizing** our relationship with customers.')
			else: st.markdown('> More **satisfied customers** are calling when deliveries are shipped on time.')

		with cap_row2_4:
			#importance
			st.markdown('### Based on level of importance ')
			st.markdown('> Parcels with')
			st.write(max_delayed_imp_level.iloc[0]['Product_importance_cat'])
			st.markdown('> importance experience the **most delays**. Further analysis of the cause problem must be made.')
		
		

		with cap_row2_6:
			#mode of shipment
			st.markdown('### Based on mode of shipment')
			st.markdown('> Parcels shipped by')
			st.write(delayed_ship_method.iloc[0]['Mode_of_Shipment'])
			st.markdown( '> experience **more delays** compared to other shipment methods')
		
	
if navigate=="Predict Shipping Time":

	row1_1, row1_2, row1_3, row1_4 ,row1_5 = st.beta_columns((0.01,1,0.2,2,0.1))

	with row1_2:
		st.markdown("# Predict Whether or not a Shipment will be Delayed")
    
	with row1_4:
		st.markdown("### Knowing if a parcel will be delivered late to a customer beforehand, gives the company the chance to make changes in shipment before the parcel is sent out")
		st.markdown("> Input details about the parcel, warehouse and customer to get the prediction.")



	#-------------------------MACHINE LEARNING PART--------------------------------------------------
	data_ml = data.copy()
	data_ml= data_ml.drop(['ID','Warehouse_block', 'Mode_of_Shipment','Gender', 'Product_importance_cat' ], axis=1)
	X_data_ml = data_ml.drop('On_time', axis=1)
	y_data_ml = data_ml[['On_time']]

	#train, test split
	X_train, X_test, y_train, y_test = train_test_split(X_data_ml, y_data_ml ,test_size=0.2, stratify=data_ml['On_time'])
	#stadardscaler

	#------------------------commenting these out for streamlit purposes-----------------
	#logistic regression
	# log_reg= LogisticRegression()
	# log_scores= cross_val_score(log_reg, X_train, y_train, cv=5, scoring= "accuracy" )
	#log_scores array([0.65219874, 0.62078812, 0.62078812, 0.61279269, 0.64877213])

	#gradient boosting classifier
	# gbc = GradientBoostingClassifier()

	# gbc_parameters = {'min_samples_split': [130, 150] ,'max_depth': [10,15, 20],'max_features': [ 'auto', 'sqrt', None],'n_estimators':[100,150]} #
	# gbc_grid_search = GridSearchCV(gbc, gbc_parameters, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1, verbose=5)

	# gbc_grid_search.fit(X_train, y_train)
	#best score: 0.6720731010850942
	#best_param: {'max_depth': 10, 'max_features': 'sqrt','min_samples_split': 150,'n_estimators': 150}

	#decision tree
	# dtc = DecisionTreeClassifier()

	# dtc_parameters = {'splitter': ['best', 'random'] , 'max_depth': [5,7,8], 'max_features': ["auto", "sqrt", "log2", None]} 
	# dtc_grid_search = GridSearchCV(dtc, dtc_parameters, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1, verbose=5)

	# dtc_grid_search.fit(X_train, y_train)
	#0.6778983438035409 {'max_depth': 5, 'max_features': None, 'splitter': 'best'}
	#need more relevant data
	#--> use decision tree classifier

	# FINAL MODEL

	dtc_model = DecisionTreeClassifier(max_depth= 5, max_features= None, splitter= 'best')
	dtc_model.fit(X_train, y_train)
	predictions = dtc_model.predict(X_test)
	pred_acc = accuracy_score(y_test, predictions) #0.6756509821836455

	imp_features = dtc_model.feature_importances_

	#plot feature importance
	data_columns = list(X_train)

	#y_pos = range(len(data_columns))
	#plt.xticks(y_pos, data_columns, rotation=90)
	fig = plt.bar(data_columns, imp_features)

	

	# for column in X_train:
	# 	#column_name = X_train[col]
	# 	st.write('input value for',column )
	# 	st.text_input(label='your input')

	care_calls_in =st.text_input(label='Input number of customer care calls')
	customer_rating_in = st.text_input(label="Input product's rating")
	Cost_of_the_Product_in = st.text_input(label='Input the cost of the product in dollars')
	Prior_purchases_in =st.text_input(label="Input the Customer's prior purchases")
	Product_importance_in = st.text_input(label="Input the product importance: 1 for low, 2 for medium and 3 for high")
	Discount_offered_in =st.text_input(label="Input the discounted amount in dollars")
	Weight_in_gms_in = st.text_input(label="Input the weight of the product in grams")
	Warehouse_block_A_in = st.text_input(label='Input 1 if shipped from warehouse block A, and 0 otherwise')
	Warehouse_block_B_in = st.text_input(label='Input 1 if shipped from warehouse block B, and 0 otherwise')
	Warehouse_block_C_in = st.text_input(label='Input 1 if shipped from warehouse block C, and 0 otherwise')
	Warehouse_block_D_in = st.text_input(label='Input 1 if shipped from warehouse block D, and 0 otherwise')
	Warehouse_block_F_in =st.text_input(label='Input 1 if shipped from warehouse block F, and 0 otherwise')
	Mode_of_Shipment_Flight_in = st.text_input(label='Input 1 if mode of shipment is flight, 0 otherwise')
	Mode_of_Shipment_Road_in = st.text_input(label='Input 1 if mode of shipment is road, 0 otherwise')
	Mode_of_Shipment_Ship_in = st.text_input(label='Input 1 if mode of shipment is by ship, 0 otherwise')
	Gender_female_in = st.text_input(label="Input 1 if the customer's gender is Female")
	Gender_male_in = st.text_input(label="Input 1 if the customer's gender is male")

	predict_go = st.button('Get Prediction')
	if predict_go:
		
		user_array = [[care_calls_in, customer_rating_in, Cost_of_the_Product_in, Prior_purchases_in, Product_importance_in,Discount_offered_in,Weight_in_gms_in,
		  Warehouse_block_A_in, Warehouse_block_B_in, Warehouse_block_C_in, Warehouse_block_D_in, Warehouse_block_F_in,
		  Mode_of_Shipment_Flight_in,Mode_of_Shipment_Road_in, Mode_of_Shipment_Ship_in,Gender_female_in,Gender_male_in  ]]
		user_df = pd.DataFrame( user_array, columns=X_train.columns)

		user_pred= dtc_model.predict(user_df)

		if user_pred[0] == 1:
			st.markdown("# <div style='color: red; font-size: 35px'>Given the shipment details, the shipment will be delayed :-1: Appropriate measures should be taken! (prediction accuracy of 70%) </div>", unsafe_allow_html=True)
		else: st.markdown("# <div style='color: green; font-size: 35px'>Given the shipment details, the shipment will be delivered on time! :metal: No further actions to be taken. (prediction accuracy of 70%) </div>", unsafe_allow_html=True)
		

		











	#------using pipelines---------------------------------------------------------------------------------------------------------------------
	#features with most predictive power are discounts, weight and cost of product

	# y_train = train[['On_time']]
	# X_train = train.drop('On_time', axis=1)

	# y_test = test[['On_time']]
	# X_test = test.drop('On_time', axis=1)

	# train_num = train[['Customer_care_calls','Customer_care_calls', 'Cost_of_the_Product', 'Prior_purchases', 'Discount_offered', 'Weight_in_gms']]
	# train_cat_ordinal =train['Product_importance']


	# num_pipeline = Pipeline([ ('num_imputer', SimpleImputer(strategy='median')), ('std_scaler' , StandardScaler())])

	# ordinal_pipeline = Pipeline([ ('ordinal_imputer', SimpleImputer(strategy='most_frequent')),('ordinal_encoder', OrdinalEncoder(categories=[['low', 'medium','high']]))])


	# #specify columns only
	# num_attribs = list(train_num)
	# ordinal_attribs = list(train_cat_ordinal)

	# full_pipeline = ColumnTransformer([ ('numeric',num_pipeline, num_attribs ),('ordinal',ordinal_pipeline, ordinal_attribs )])

	# X_train_final = full_pipeline.fit_transform(X_train)
