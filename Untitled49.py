#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


df = pd.read_csv(r'C:\Users\Lankala Vinay Kumar\client_data.csv')
df1 = pd.read_csv(r"C:\Users\Lankala Vinay Kumar\price_data.csv")
df.head()


# In[13]:


df1.head()


# In[14]:


df.info()


# In[15]:


df.isnull().sum()


# In[16]:


df1.isnull().sum()


# In[21]:


df.describe()


# In[23]:


df.dtypes


# In[26]:


sns.lineplot(x="cons_12m", y ='cons_last_month', data=df)


# In[28]:


sns.lineplot(x="cons_last_month",y="forecast_cons_12m",data=df)


# In[35]:


sns.pairplot(df,hue ='churn' )


# In[45]:


sns.countplot(data=df, x='cons_12m')


# In[38]:


df.columns


# In[47]:


date_columns = ['date_activ','date_modif_prod','date_end','date_renewal']
df[date_columns]= df[date_columns].apply(pd.to_datetime,format="%Y-%m-%d")


# # Relation Between variables 

# In[51]:


def plot_stacked_bar(dataframe, title_, size=(18,10), rot_=0, legend_='upper right'):
    ax = dataframe.plot(
        kind='bar',
        stacked=True,
        figsize=size,
        rot=rot_,
        title=title_
    )
    
    annotate_stacked_bars(ax, textsize=14)  # Assuming this function is defined elsewhere

    plt.legend(['Retention', 'Churn'], loc=legend_)
    plt.ylabel('Company base (%)')
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, color='white', textsize=13):
    for p in ax.patches:
        value = str(round(p.get_height(), 2))
        if value == '0.0':
            continue
        ax.annotate(
            value,
            (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height() / 2),
            ha='center', va='center', color=color, size=textsize
        )


# Code Explanation
# Function Definition:
# 
# plot_stacked_bar: This function takes a DataFrame and creates a stacked bar plot. The main arguments include:
# dataframe: The DataFrame containing the data to plot.
# title_: The title of the plot.
# size: The size of the plot (default is (18, 10)).
# rot_: The rotation angle for the x-axis labels (default is 0).
# legend_: The position of the legend (default is 'upper right').
# Plotting the Stacked Bar Chart:
# 
# )
# dataframe.plot(...): This line creates a stacked bar chart using the provided DataFrame.
# kind='bar': Specifies that the plot type is a bar chart.
# stacked=True: Indicates that the bars should be stacked on top of each other.
# figsize=size_: Sets the size of the figure.
# rot=rot_: Rotates the x-axis labels by the specified angle.
# title=title_: Sets the title of the plot.
# Annotation Function Call:
# 
# python
# Copy code
# annotate_stacked_bars(ax, textsize=14)
# This calls a function annotate_stacked_bars (not defined in this snippet) to add annotations to the bars. The textsize=14 sets the size of the text annotations.
# Legend and Labels:
# 
# python
# Copy code
# plt.legend(['Retention', 'Churn'], loc=legend_)
# plt.ylabel('Company base (%)')
# plt.show()
# plt.legend(...): This line sets the legend for the plot, labeling the stacked segments as "Retention" and "Churn".
# plt.ylabel(...): Sets the label for the y-axis.
# plt.show(): Displays the plot.
# Manual Annotations (Redundant and Incorrect):
# 
# python
# Copy code
# for p in ax.patches:
#     value = str(round(p.get_height(), 2))
#     if value == '0.0':
#         continue
#     ax.annotate(
#         value,
#         ((p.get_x() + p.get_width() / 2) * pad - 0.06, (p.get_y() + p.get_height() / 2) * pad),
#         color=colour,
#         size=textsize
#     )
# Iteration over Patches: for p in ax.patches: iterates over each bar segment in the plot.
# Calculate Annotation: value = str(round(p.get_height(), 2)) calculates the height of each bar segment, rounds it to two decimal places, and converts it to a string.
# Check for Zero Values: if value == '0.0': continue skips the annotation if the value is zero.
# Annotate the Bars: ax.annotate(...) places the annotation text on the plot, centered in the bar segment.

# In[55]:


churn = df[['id','churn']]
churn.columns = ['Company','churn']
churn_total = churn.groupby('churn').count()
churn_pct = (churn_total / churn_total.sum()) * 100
churn_pct.transpose()


# In[60]:


plot_stacked_bar(churn_pct.transpose(), 'Churning status', (8,6), legend_=1)
print("\n ---Value counts -- \n")
print(churn['churn'].value_counts())


# Insights
# Majority of Companies Have Not Churned:
# 
# A total of 13,187 companies have not churned (churn = 0).
# This represents the majority of the dataset, indicating that most companies are being retained.
# Smaller Proportion Have Churned:
# 
# A total of 1,419 companies have churned (churn = 1).
# This is a much smaller number compared to those that have not churned.
# Churn Rate:
# 
# You can calculate the churn rate as the proportion of churned companies out of the total number of companies.
# Churn Rate = 
# 1419
# 13187
# +
# 1419
# ×
# 100
# ≈
# 9.7
# %
# 13187+1419
# 1419
# ​
#  ×100≈9.7%
# This means that about 9.7% of the companies in your dataset have churned.
# Implications
# Low Churn Rate: A churn rate of around 9.7% suggests that the company’s retention strategies might be effective, as a large majority of companies are not churning.
# 
# Focus Areas: While the churn rate is relatively low, understanding the characteristics of the 1,419 companies that did churn could provide insights into potential risk factors or areas where improvements in customer satisfaction or product offerings might be needed.
# 
# Comparison: It might be useful to compare these churn rates across different segments (e.g., by product type, region, customer size) to identify specific areas with higher churn and address those issues directly.
# 
# 
# 
# 
# 
# 
# 

# In[65]:


channel =  df[['id','channel_sales','churn']]
channel = channel.groupby(['channel_sales',"churn"])['id'].count().unstack(level=1).fillna(0)
channel_churn = (channel.div(channel.sum(axis=1),axis=0)*100).sort_values(by=[1],ascending=0)
channel_churn


# In[67]:


plot_stacked_bar(channel_churn,'Sales channel', rot_=60)


# Code Explanation
# 
# Data Selection and Grouping:
# 
# 
# channel = client[['id', 'channel_sales', 'churn']]
# This line selects the relevant columns (id, channel_sales, and churn) from the client DataFrame.
# 
# channel = channel.groupby(['channel_sales', 'churn'])['id'].count().unstack(level=1).fillna(0)
# This groups the data by channel_sales and churn, counting the number of ids (customers) for each combination of channel_sales and churn.
# unstack(level=1) pivots the churn values to columns, resulting in a DataFrame where columns represent churn categories (0 and 1).
# fillna(0) replaces any missing values with 0, ensuring no NaNs in the DataFrame.
# Calculate Churn Rates:
# 
# 
# channel_churn = (channel.div(channel.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)
# channel.sum(axis=1) computes the total number of customers for each channel_sales category.
# channel.div(channel.sum(axis=1), axis=0) divides each value in the DataFrame by the total number of customers for that channel, effectively calculating the proportion of churning and non-churning customers for each channel.
# Multiplying by 100 converts these proportions to percentages.
# sort_values(by=[1], ascending=False) sorts the DataFrame by the churn rate for category 1 (churning) in descending order, so channels with higher churn rates appear first.
# Resulting channel_churn DataFrame
# Rows: Each row represents a different sales channel.
# 
# Columns:
# 
# One column for the churn rate of churn_0 (non-churning customers).
# One column for the churn rate of churn_1 (churning customers).
# Values: The values in the DataFrame represent the percentage of churning and non-churning customers for each sales channel.
# 
# Insight
# The channel_churn DataFrame allows you to understand the distribution of churn rates across different sales channels. By analyzing this DataFrame, you can identify which sales channels have higher churn rates and how the churn rates compare between channels. This can provide valuable insights for targeting customer retention strategies and improving overall sales channel performance.
# 
# 
# 
# 
# 
# 
# 

# In[68]:


consumptions = df[['id','cons_12m','cons_gas_12m','cons_last_month','imp_cons','has_gas','churn']]
consumptions.head()


# In[70]:


import pandas as pd
import matplotlib.pyplot as plt

def plot_distribution(dataframe, column, ax, bins_=50):
    """
    Plot variable distribution in a stacked histogram of churned or retained company.
    
    Parameters:
    - dataframe: DataFrame containing the data.
    - column: The column name for which to plot the histogram.
    - ax: Matplotlib Axes object to plot on.
    - bins_: Number of bins for the histogram.
    """
    # Create a temporary DataFrame with the data to plot
    temp = pd.DataFrame({
        'Retention': dataframe[dataframe['churn'] == 0][column],
        'Churn': dataframe[dataframe['churn'] == 1][column]
    })
    
    # Plot the histogram
    temp.plot(kind='hist', bins=bins_, ax=ax, stacked=True, alpha=0.7)
    
    # Set the x-axis label
    ax.set_xlabel(column)
    
    # Format the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')

# Create subplots
fig, ax = plt.subplots(nrows=4, figsize=(18, 25))

# Plot distributions
plot_distribution(consumptions, 'cons_12m', ax[0])
plot_distribution(consumptions[consumptions['has_gas'] == 't'], 'cons_gas_12m', ax[1])
plot_distribution(consumptions, 'cons_last_month', ax[2])
plot_distribution(consumptions, 'imp_cons', ax[3])

# Adjust layout
plt.tight_layout()
plt.show()


# In[74]:


fig, ax = plt.subplots(nrows=4,figsize=(18,25))
sns.boxplot(x=consumptions['cons_12m'],ax=ax[0])
sns.boxplot(x=consumptions[consumptions['has_gas']=='t']['cons_gas_12m'],ax=ax[1])
sns.boxplot(x=consumptions['cons_last_month'],ax=ax[2])
sns.boxplot(x=consumptions['imp_cons'],ax=ax[3])



# In[75]:


forecast= df[['id',
    "forecast_cons_12m",
    "forecast_cons_year",
    "forecast_discount_energy",
    "forecast_meter_rent_12m",
    "forecast_price_energy_off_peak",
    "forecast_price_energy_peak",
    "forecast_price_pow_off_peak",
    "churn"]]
forecast.head()


# In[79]:


fig, ax= plt.subplots(nrows=7, figsize=(18,50))

plot_distribution(df,"forecast_cons_12m",ax[0])
plot_distribution(df,"forecast_cons_year",ax[1])
plot_distribution(df,"forecast_discount_energy",ax[3])
plot_distribution(df,"forecast_meter_rent_12m",ax[4])
plot_distribution(df,"forecast_price_energy_off_peak",ax[5])
plot_distribution(df,"forecast_price_energy_peak",ax[6])
plot_distribution(df,"forecast_price_pow_off_peak",ax[2])




# In[81]:


contract = df[['id','has_gas','churn']]
contract = df.groupby(['has_gas','churn'])['id'].count().unstack(level=1)
contract_pct = (contract.div(contract.sum(axis=1),axis=0)*100).sort_values(by=0,ascending=0)
contract_pct


# In[82]:


plot_stacked_bar(contract_pct,'Contract Type (with gas)')


# In[85]:


margin = df[['id','margin_gross_pow_ele','margin_net_pow_ele','net_margin']]

fig,ax = plt.subplots(nrows=3,figsize=(18,20))
sns.boxplot(x=margin["margin_gross_pow_ele"], ax=ax[0])
sns.boxplot(x=margin["margin_net_pow_ele"],ax=ax[1])
sns.boxplot(x=margin["net_margin"], ax=ax[2])


# In[87]:


power = df[['id','pow_max','churn']]

fig, ax = plt.subplots(nrows=1, figsize=(18, 10))
plot_distribution(power, 'pow_max', ax)


# In[89]:


other_cols = df[['id','nb_prod_act','num_years_antig','origin_up','churn']]
products = other_cols.groupby(['churn','nb_prod_act'])['id'].count().unstack(level=0).fillna(0)
products_churn = (products.div(products.sum(axis=1), axis=0)*100)
products_churn


# In[90]:


plot_stacked_bar(products_churn,'Number of Products')


# In[98]:


years_antig = other_cols.groupby(['churn','num_years_antig'])['id'].count().unstack(level=0).fillna(0)
years_antig = (years_antig.div(years_antig.sum(axis=1),axis=0)*100)
years_antig


# In[100]:


plot_stacked_bar(years_antig,'Number of years')


# In[101]:


origin = other_cols(['id','churn','origin_up'])


# In[102]:


plot_stacked_bar(origin,'Origin contract/offer', rot_=30)


# # Price dataset
# 

# In[104]:


df1.info()


# In[105]:


df1.head(n=14)


# In[107]:


df1.isnull().sum()


# In[108]:


df1.dtypes


# In[111]:


df1['price_date'] = pd.to_datetime(df1['price_date'], format='%Y-%m-%d')


# In[114]:


#create average data
mean_year = df1.groupby('id').mean().reset_index()
mean_6m = df1[df1['price_date']>'2015-06-30'].groupby('id').mean().reset_index()
mean_3m = df1[df1['price_date']>'2015-09-30'].groupby('id').mean().reset_index()


# In[115]:


#rename the columns of mean year
mean_year = mean_year.rename(
    columns = {
        'price_off_peak_var':'mean_year_price_off_peak_var',
        'price_peak_var': 'mean_year_price_peak_var',
        "price_mid_peak_var": "mean_year_price_mid_peak_var",
        "price_off_peak_fix": "mean_year_price_off_peak_fix",
        "price_peak_fix": "mean_year_price_peak_fix",
        "price_mid_peak_fix": "mean_year_price_mid_peak_fix"
    }
)


# In[116]:


mean_year['mean_year_price_off_peak'] = mean_year['mean_year_price_off_peak_var'] + mean_year['mean_year_price_off_peak_fix']
mean_year['mean_year_price_peak'] = mean_year["mean_year_price_peak_var"] + mean_year["mean_year_price_peak_fix"]
mean_year["mean_year_price_mid_peak"] = mean_year["mean_year_price_mid_peak_var"] + mean_year["mean_year_price_mid_peak_fix"]


# In[117]:


#rename columns of mean 6 month
mean_6m = mean_6m.rename(

    columns={
        "price_off_peak_var": "mean_6m_price_off_peak_var",
        "price_peak_var": "mean_6m_price_peak_var",
        "price_mid_peak_var": "mean_6m_price_mid_peak_var",
        "price_off_peak_fix": "mean_6m_price_off_peak_fix",
        "price_peak_fix": "mean_6m_price_peak_fix",
        "price_mid_peak_fix": "mean_6m_price_mid_peak_fix"
    }
)

mean_6m["mean_6m_price_off_peak"] = mean_6m["mean_6m_price_off_peak_var"] + mean_6m["mean_6m_price_off_peak_fix"]
mean_6m["mean_6m_price_peak"] = mean_6m["mean_6m_price_peak_var"] + mean_6m["mean_6m_price_peak_fix"]
mean_6m["mean_6m_price_mid_peak"] = mean_6m["mean_6m_price_mid_peak_var"] + mean_6m["mean_6m_price_mid_peak_fix"]
#rename the columns of mean 3 month
mean_3m = mean_3m.rename(

    columns={
        "price_off_peak_var": "mean_3m_price_off_peak_var",
        "price_peak_var": "mean_3m_price_peak_var",
        "price_mid_peak_var": "mean_3m_price_mid_peak_var",
        "price_off_peak_fix": "mean_3m_price_off_peak_fix",
        "price_peak_fix": "mean_3m_price_peak_fix",
        "price_mid_peak_fix": "mean_3m_price_mid_peak_fix"
    }
)

mean_3m["mean_3m_price_off_peak"] = mean_3m["mean_3m_price_off_peak_var"] + mean_3m["mean_3m_price_off_peak_fix"]
mean_3m["mean_3m_price_peak"] = mean_3m["mean_3m_price_peak_var"] + mean_3m["mean_3m_price_peak_fix"]
mean_3m["mean_3m_price_med_peak"] = mean_3m["mean_3m_price_mid_peak_var"] + mean_3m["mean_3m_price_mid_peak_fix"]
#merge into 1 dataframe
price_faetures = pd.merge(mean_year, mean_6m, on='id')
price_features = pd.merge(price_faetures, mean_3m, on='id')
price_features


# In[119]:


#Now lets merge in the churn data and see whether price sensitivity has any correlation with churn
price_analysis = pd.merge(price_features, df[['id','churn']], on='id')
price_analysis.drop(['id','price_date_x','price_date_y','price_date'],axis=1,inplace=True)
#Checking correlation
corr = price_analysis.corr()

#ploting correlation
plt.figure(figsize=(28,21))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,
            annot=True, annot_kws={'size':'10'})
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


# In[122]:


#Now we will merge the client data with price churn data for modeling in the next move
price_churn = pd.merge(price_features, df[['id','churn']], on='id')
price_churn.drop(['price_date_x','price_date_y','price_date'],axis=1,inplace=True)
churn_data = pd.merge(df.drop(columns=['churn']), price_churn, on='id')
churn_data.head()


# In[123]:


churn_data.to_csv('churn_data_modeling.csv')

