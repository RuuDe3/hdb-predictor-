import pandas as pd
df=pd.read_csv('./data/resale_transactions.csv')




df.head()



df.info()



df.isnull().sum()



df.describe().T



df.duplicated().sum()



df[df.duplicated(keep=False)].sort_values(by=list(df.columns))



df.drop_duplicates(inplace=True)


df.tail()



df.duplicated().sum()


len(df)


num_duplicates = 88688 - 84465
print(num_duplicates)


df.isnull().sum()





import seaborn as sns
import matplotlib.pyplot as plt



sns.set_style('whitegrid')


plt.figure(figsize=(10,6))
count_plot = sns.countplot(x='flat_type', data=df)
plt.title('Count of Different Flat Types')
plt.xlabel('Flat Type')
plt.ylabel('Count')
plt.xticks(rotation=45)


for p in count_plot.patches:
    count_plot.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 9),
                        textcoords = 'offset points')
plt.show()


df['flat_type'] = df['flat_type'].replace('FOUR ROOM', '4 ROOM')


print(df['flat_type'].value_counts())


sns.set_style("whitegrid")


plt.figure(figsize=(10, 6))
sns.histplot(df['resale_price'], kde=True)
plt.title('Distribution of Resale Price')
plt.xlabel('Resale Price')
plt.ylabel('Frequency')
plt.show()


sns.set_style("whitegrid")


month_counts = df['month'].value_counts().sort_index()


plt.figure(figsize=(14, 7))
line_plot = sns.lineplot(x=month_counts.index, y=month_counts.values, marker='o')


for x, y in zip(month_counts.index, month_counts.values):
    plt.text(x, y, f'{y}', ha='center', va='bottom')


plt.title('Trend of Number of Resale Listings Over the Months')
plt.xlabel('Month')
plt.ylabel('Number of Resale Listings')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




sns.set_style('whitegrid')


plt.figure(figsize=(14, 8))
sns.boxplot(x='flat_type', y='resale_price', data=df, hue='flat_type', palette='Set2')
plt.title('Resale Price by Flat Type')
plt.xlabel('Flat Type')
plt.ylabel('Resale Price')
plt.xticks(rotation=0)
plt.show()


sns.set_style('whitegrid')


fig, axes = plt.subplots(2, 1, figsize=(10, 12))


sorted_town_name = df['town_name'].value_counts().index


count_plot_1 = sns.countplot(y='town_name', data=df, order=sorted_town_name, ax=axes[0])
axes[0].set_title('Count of town_name')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('town_name')


for p in count_plot_1.patches:
    count_plot_1.annotate(format(p.get_width(), '.0f'),
                          (p.get_width(), p.get_y() + p.get_height() / 2),
                          ha='center', va='center',
                          xytext=(15, 0),
                          textcoords='offset points')
    

sorted_town_id = df['town_id'].value_counts().index


count_plot_2 = sns.countplot(y='town_id', data=df, order=sorted_town_id, ax=axes[1])
axes[1].set_title('Count of town_id')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('town_id')

for p in count_plot_2.patches:
     count_plot_2.annotate(format(p.get_width(), '.0f'),
                           (p.get_width(), p.get_y() + p.get_height() / 2),
                           ha='center', va='center',
                           xytext=(15, 0),
                           textcoords='offset points')
     

plt.tight_layout()

plt.show()



miss_yishun = 5111 - 4870
miss_bedok = 5009 - 4753
miss_punggol = 4707 - 4463
total_miss_town = miss_yishun + miss_bedok + miss_punggol
print(total_miss_town)





sns.set_style('whitegrid')


fig, axes = plt.subplots(2,1, figsize=(10,12))


sorted_flatm_name = df['flatm_name'].value_counts().index


count_plot_1 = sns.countplot(y='flatm_name', data=df, order=sorted_flatm_name, ax=axes[0])
axes[0].set_title ('Count of flatm_name')
axes[0].set_xlabel ('Count')
axes[0].set_ylabel ('flatm_name')



for _ in count_plot_1.patches:
    count_plot_1.annotate(format(_.get_width(), '.0f'),
                          (_.get_width(), _.get_y() + _.get_height() / 2),
                          ha='center', va='center',
                          xytext=(15,0),
                          textcoords='offset points')


sorted_flatm_id = df['flatm_id'].value_counts().index


count_plot_2 = sns.countplot(y='flatm_id', data=df, order=sorted_flatm_id, ax=axes[1])
axes[1].set_title ('Count of flatm_id')
axes[1].set_xlabel ('Count')
axes[1].set_ylabel ('flatm_id')



for i in count_plot_2.patches:
    count_plot_2.annotate(format(i.get_width(), '.0f'),
                          (i.get_width(), i.get_y() + i.get_height() / 2),
                          ha='center', va='center',
                          xytext=(15,0),
                          textcoords='offset points')


plt.tight_layout() 


plt.show




miss_simpl = 3887 - 3690
miss_apart = 3382 - 3223
miss_masion = 2506 - 2433
miss_stand = 2433 - 2374
total_miss_flat = miss_simpl + miss_apart + miss_masion + miss_stand
print(total_miss_flat)

def fill_missing_names(df: pd.DataFrame, id_column:str, name_column:str) ->pd.DataFrame:
    """
    Fills missing values in the 'name_column
 using the 'id_column'.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to be filled.
        id_columns (str): The name of the column containing the IDs.
        name_column (str): The name of the column containing the names to be filled.

    Returns:
        pd.DataFrame: The DataFrame with missing values in 'name_column' filled.
    """
  
    missing_names= df[name_column].isna()

   
    name_mapping = df[[id_column, name_column]].dropna().drop_duplicates().set_index(id_column)[name_column].to_dict()


    df.loc[missing_names, name_column] = df.loc[missing_names, id_column].map(name_mapping)

    return df


df = fill_missing_names (df=df, id_column='town_id', name_column='town_name')


df = fill_missing_names(df=df, id_column='flatm_id', name_column='flatm_name')





sns.set_style('whitegrid')
fig, axes = plt.subplots(2,1,figsize=(10,12))
sorted_flatm_name = df['flatm_name'].value_counts().index


count_plot_1 = sns.countplot(y='flatm_name', data=df, order=sorted_flatm_name, ax=axes[0])
axes[0].set_title('Count of flatm_name')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('faltm_name')



for _ in count_plot_1.patches:
    count_plot_1.annotate(format(_.get_width(), '.0f'),
                          (_.get_width(), _.get_y() + _.get_height() / 2),
                          ha='center', va='center',
                          xytext=(15,0),
                          textcoords='offset points')
    

count_plot_2 = sns.countplot(y='town_name', data=df, order=sorted_town_name, ax=axes[1])
axes[1].set_title('Count of town_name')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('town_name')


for _ in count_plot_2.patches:
    count_plot_2.annotate(format(_.get_width(), '.0f'),
                          (_.get_width(), _.get_y() + _.get_height() / 2),
                          ha='center', va='center',
                          xytext=(15,0),
                          textcoords='offset points')
    
plt.show()


df.isnull().sum()


sns.set_style("whitegrid")


plt.figure(figsize=(12, 8))
sns.scatterplot(x='floor_area_sqm', y='resale_price', hue='flat_type', data=df, palette='Set2', alpha=0.7)
plt.title('Scatter Plot of Floor Area vs Resale Price by Flat Type')
plt.xlabel('Floor Area (sqm)')
plt.ylabel('Resale Price')
plt.legend(title='Flat Type')
plt.show()


outliers = df[(df['flat_type'] == '3 ROOM') &
              (df['floor_area_sqm'] > 150) &
              (df['resale_price'] > 800000)]



outliers




terrace_outliers = df[(df['flat_type'] == '3 ROOM') &
                      (df['flatm_name'] == "Terrace")]

terrace_outliers.describe().T


df['lease_commence_date'] = df['lease_commence_date'].abs()



df.describe().T


continuous_vars_all = df[['floor_area_sqm', 'lease_commence_date', 'resale_price']]


corr_matrix_pear = continuous_vars_all.corr(method='pearson')




corr_matrix_pear


plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pear, annot=True, cmap='coolwarm', vmin=0, center=0.5, vmax=1, linewidths=0.5, fmt=".2f")
plt.title('Pearson Correlation Heatmap')
plt.show()


def convert_storey_range(storey_range: str) ->float:
    """
    Converts a storey range string into its average numerical value.

    The function takes a storey range in the format 'XX TO YY', splits it into two parts, 
    converts these parts to integers, and returns the average of these integers.

    Args:
        storey_range(str): A string representing a range of storeys, in the format
        'XX TO YY'
    
    Returns:
        float: The average of the two storey in the range.

    Example:
        convert_storey_range('07 TO 09') -> 8.0   
    
    """

    range_values = storey_range.split(' TO ')
    return (int(range_values[0]) + int(range_values[1])) / 2
df['storey_range'] = df['storey_range'].apply(convert_storey_range)

spearman_vars = df[['storey_range', 'floor_area_sqm', 'lease_commence_date', 'resale_price']]

spearman_corr = spearman_vars.corr(method='spearman')

spearman_corr


plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, vmin=0, vmax=1, linewidths=0.5, fmt=".2f")
plt.title('Spearman Correlation Matrix Heatmap')
plt.show()



df = df.drop(columns=['town_id', 'id', 'flatm_id'])



df['year_month'] = pd.to_datetime(df['month'], format='%Y-%m')


df['year'] = df['year_month'].dt.year
df['month'] = df['year_month'].dt.month


df.head()


df = df.drop(columns=['year_month'])

df.head()


import re

def extract_lease_info(lease_str : str) -> int:

    """
    Convert lease information from a string format to total months

    this function takes a string representing the remaining lease period, which
    may include year and months in various formats (e.g "70 years 3 months",
    "85 years", "67"), and converts it to the total number of months.

    args:
        lease_str(str) : the remaining lease period as a string

    Returns:
        int : the total number of months, or None if the input is NaN.
    """

    if pd.isna(lease_str):
        return None
    
    
    years_match = re.search(r'(\d+)\s*years?', lease_str)
    months_match = re.search(r'(\d+)\s*months?', lease_str)
    number_match = re.search(r'^\d+$', lease_str.strip())

    if years_match:
        years = int(years_match.group(1))
    elif number_match: # if only a number is present, assume it's in years
        years = int(number_match.group(0))
    else:
        years = 0

    months = int(months_match.group(1)) if months_match else 0

 
    total_months = years * 12 + months
    return total_months



df['remaining_lease_months'] = df['remaining_lease'].apply(extract_lease_info)

df[['remaining_lease', 'remaining_lease_months']].sample(5)


