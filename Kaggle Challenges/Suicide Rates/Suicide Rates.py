import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import random
# Seed so we always get same results when filling missing values
random.seed()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
######################################################################################################
# https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
######################################################################################################
# Get data
data = pd.read_csv('data/master.csv')
data = data.drop(['country-year'], axis=1)
# print(data.head(15))
# Columns: ['country', 'year', 'sex', 'age', 'suicides_no', 'population', 'suicides/100k pop', 'HDI for year', ' gdp_for_year ($) ', 'gdp_per_capita ($)', 'generation']

def show_all_available_countries():
      '''
      ['Albania' 'Antigua and Barbuda' 'Argentina' 'Armenia' 'Aruba' 'Australia'
       'Austria' 'Azerbaijan' 'Bahamas' 'Bahrain' 'Barbados' 'Belarus' 'Belgium'
       'Belize' 'Bosnia and Herzegovina' 'Brazil' 'Bulgaria' 'Cabo Verde'
       'Canada' 'Chile' 'Colombia' 'Costa Rica' 'Croatia' 'Cuba' 'Cyprus'
       'Czech Republic' 'Denmark' 'Dominica' 'Ecuador' 'El Salvador' 'Estonia'
       'Fiji' 'Finland' 'France' 'Georgia' 'Germany' 'Greece' 'Grenada'
       'Guatemala' 'Guyana' 'Hungary' 'Iceland' 'Ireland' 'Israel' 'Italy'
       'Jamaica' 'Japan' 'Kazakhstan' 'Kiribati' 'Kuwait' 'Kyrgyzstan' 'Latvia'
       'Lithuania' 'Luxembourg' 'Macau' 'Maldives' 'Malta' 'Mauritius' 'Mexico'
       'Mongolia' 'Montenegro' 'Netherlands' 'New Zealand' 'Nicaragua' 'Norway'
       'Oman' 'Panama' 'Paraguay' 'Philippines' 'Poland' 'Portugal'
       'Puerto Rico' 'Qatar' 'Republic of Korea' 'Romania' 'Russian Federation'
       'Saint Kitts and Nevis' 'Saint Lucia' 'Saint Vincent and Grenadines'
       'San Marino' 'Serbia' 'Seychelles' 'Singapore' 'Slovakia' 'Slovenia'
       'South Africa' 'Spain' 'Sri Lanka' 'Suriname' 'Sweden' 'Switzerland'
       'Thailand' 'Trinidad and Tobago' 'Turkey' 'Turkmenistan' 'Ukraine'
       'United Arab Emirates' 'United Kingdom' 'United States' 'Uruguay'] '''
      print(data.country.unique())

def show_count_and_average_suicides_for_a_country(country_name="Germany", plot=False):
      # Get average suicides per year and total number for a given country
      # Plot allsuicides

      # Some data from Country
      country = data.loc[data['country'] == country_name]
      country_cya = country[['country', 'year', 'suicides_no']]

      # Year / Sum Suicides
      country_cya_aggr = country_cya.groupby(['year'], as_index=False).agg({'suicides_no': "sum"})
      # print(country_cya_aggr)
      # Average
      print("From ", country_cya_aggr['year'].min(), " until ", country_cya_aggr['year'].max(), " there were ",
            country_cya_aggr['suicides_no'].sum(), " suicides in " , country_name)
      print("That's ", round(country_cya_aggr['suicides_no'].mean(), 2), " on average per year.")

      if plot == True:
            country_cya_aggr.plot(x='year', y='suicides_no', kind='bar', title=str(country_name), label='Number of Suicides', color='r')
            plt.show()

def which_country_has_highest_suicide_to_population_ratio(plot=False):
      # For which year do most countries have values?
      # Group by year -> Get number of unique values for countries -> sort descending
      # years = data[['year','country']].groupby(['year'], as_index=True)['country'].nunique().sort_values(ascending=False)
      # The first value is the year we pick
      # year_with_most_countries = years.reset_index(level=0).iloc[0]['year'] # 2009 <----------------------
      # print(year_with_most_countries)

      # All Data for 2009
      all_data_2009 = data[['country','suicides_no','population']].loc[data['year'] == 2009]
      all_data_2009 = all_data_2009.groupby(['country']).agg({'suicides_no':'sum','population':'sum'})
      all_data_2009['ratio'] = (all_data_2009['suicides_no'] / all_data_2009['population'])
      all_data_2009 = all_data_2009.drop(['population', 'suicides_no'], axis=1).sort_values(ascending=True, by='ratio' )
      if plot == True:
            all_data_2009.plot.bar(y='ratio', figsize=(13, 7))
            plt.show()
      print(all_data_2009)

