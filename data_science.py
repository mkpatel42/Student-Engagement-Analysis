#read files
from gettext import install

import pandas

import inline as inline
import matplotlib
import pip

import unicodecsv
from matplotlib import pylab
from matplotlib.pyplot import hist


def readfile(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)
enrollments = readfile('C:/Users/Swapneel/Desktop/study/anacondafiles/enrollments (1).csv')
daily_engagement = readfile('C:/Users/Swapneel/Desktop/study/anacondafiles/daily_engagement (1).csv')
project_submissions = readfile('C:/Users/Swapneel/Desktop/study/anacondafiles/project_submissions (1).csv')

print(enrollments[0])
#no of recorcrd in enrollment
len(enrollments)

#change the key name in daily eng
for d in daily_engagement:
    d['account_key'] = d['acct']
    del[d['acct']]
#convert data types as eery value is string change it in date and int
#fixing data types  (strptime)
from datetime import datetime as dt
def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%Y-%m-%d')
def parse_maybe_int(i):
    if i == '':
        return None
    else:
        return int(i)
#cleanupdata in engagement table
for e in enrollments:
    e['cancel_date'] = parse_date(e['cancel_date'])
    e['days_to_cancel'] = parse_maybe_int(e['days_to_cancel'])
    e['is_canceled'] = e['is_canceled'] == 'True'
    e['is_udacity'] = e['is_udacity'] == 'True'
    e['join_date'] = parse_date(e['join_date'])
for engagement_record in daily_engagement:
    engagement_record['lessons_completed'] = int(float(engagement_record['lessons_completed']))
    engagement_record['num_courses_visited'] = int(float(engagement_record['num_courses_visited']))
    engagement_record['projects_completed'] = int(float(engagement_record['projects_completed']))
    engagement_record['total_minutes_visited'] = float(engagement_record['total_minutes_visited'])
    engagement_record['utc_date'] = parse_date(engagement_record['utc_date'])

daily_engagement[0]



# Clean up the data types in the submissions table
for submission in project_submissions:
    submission['completion_date'] = parse_date(submission['completion_date'])
    submission['creation_date'] = parse_date(submission['creation_date'])

project_submissions[0]

# students who are enrolled but not engadge
#unique student in enrollment
unique_stud = set()
for f in enrollments:
    unique_stud.add(f['account_key'])
print(len(unique_stud))

#in project sub
unique_stud3 = set()
for f in project_submissions:
    unique_stud3.add(f['account_key'])
print(len(unique_stud3))
#in daily eng
unique_stud2 = set()
for f in daily_engagement:
    unique_stud2.add(f['account_key'])
len(unique_stud2)

daily_engagement[0]['account_key']
# students who are enrolled but not engadge
for e in enrollments:
    student = e['account_key']
    if student not in unique_stud2:
        print(e)
        break
#enrolled for just one day
prob_stud = 0
for e in enrollments:
    student = e['account_key']
    if student not in unique_stud2 \
        and e['join_date'] != e['cancel_date']:
        prob_stud +=1
        print(e)
print(prob_stud)
#check for is udacity
uda_test = set()
for e in enrollments:
    if e['is_udacity']:
        uda_test.add(e['account_key'])
print(len(uda_test))

#remove that data correspond to IS_Udacity
def removedata(data):
    non_uda = []
    for d in data:
        if d['account_key'] not in uda_test:
            non_uda.append(d)
    return non_uda

print(len(enrollments))
print(len(daily_engagement))
print(len(project_submissions))
uda_enr = removedata(enrollments)
uda_eng = removedata(daily_engagement)
uda_sub = removedata(project_submissions)
print(len(uda_enr))
print(len(uda_eng))
print(len(uda_sub))
print('-------------------')
#enrolled for a week
paid_dict = {}
for e in uda_enr:
    if not e['is_canceled'] or e['days_to_cancel']>7:
        key = e['account_key']
        value = e['join_date']
        paid_dict[key] = value
        if key not in paid_dict or \
            value > paid_dict[key]:
            paid_dict[key] = value

print(len(paid_dict))
print('-----------------')
#function to return free trial student record wich is within 7 days
def within_one_week(join_date, engagement_date):
    time_delta = engagement_date - join_date
    return time_delta.days < 7 and time_delta.days >=0
#function to remove thoes student
def remove_free_trial_cancels(data):
    new_data = []
    for data_point in data:
        if data_point['account_key'] in paid_dict:
            new_data.append(data_point)
    return new_data

paid_enrollments = remove_free_trial_cancels(uda_enr)
paid_engagement = remove_free_trial_cancels(uda_eng)
paid_submissions = remove_free_trial_cancels(uda_sub)

print(len(paid_enrollments))
print(len(paid_engagement))
print(len(paid_submissions))
print('-------------')

# visited = 1 non visited = 0
# usage: to cont total no of days visited by student
for engagement_record in paid_engagement:
    if engagement_record['num_courses_visited'] >0:
        engagement_record['has_visited'] = 1
    else:
        engagement_record['has_visited'] = 0



##stud eng in first week
paid_engagement_in_first_week = []
for engagement_record in uda_eng:
    account_key = engagement_record['account_key']
    if account_key in paid_dict.keys() and within_one_week(paid_dict[account_key],engagement_record['utc_date']):
        paid_engagement_in_first_week.append(engagement_record)

print(len(paid_engagement_in_first_week))



#exploring student engagement
# total no of minues eng by student in first week

from collections import defaultdict
engagement_by_account = defaultdict(list)
for engagement_record in paid_engagement_in_first_week:
    account_key = engagement_record['account_key']
    engagement_by_account[account_key].append(engagement_record)
total_minutes_by_account = {}
for account_key,engagement_for_student in engagement_by_account.items():
    total_minutes = 0
    for engagement_record in engagement_for_student:
        total_minutes += engagement_record['total_minutes_visited']
    total_minutes_by_account[account_key] = total_minutes

total_minutes = list(total_minutes_by_account.values())

import numpy as np
print('min', np.min(total_minutes))
print('max', np.max(total_minutes))
print('std', np.std(total_minutes))
print('mean', np.mean(total_minutes))

#student eng with max min
#stud within first week
student_with_max_minutes = None
max_minutes = 0
#get key and values using .items()
for student, total_minutes in total_minutes_by_account.items():
    if total_minutes > max_minutes:
        max_minutes = total_minutes
        student_with_max_minutes = student

print('max min', max_minutes)
for engagement_record in paid_engagement_in_first_week:
    if engagement_record['account_key'] == student_with_max_minutes:
        print(engagement_record)
print(len(engagement_record))


#same as previous
# no of sutdent completed lesson in first week
from collections import defaultdict

def group_data(data, key_name):  #group data by key
    grouped_data = defaultdict(list)
    for data_point in data:
        key = data_point[key_name]
        grouped_data[key].append(data_point)
    return grouped_data

engagement_by_account = group_data(paid_engagement_in_first_week, 'account_key')

def sum_grouped_items(grouped_data, field_name):
    summed_data = {}
    for key, data_points in grouped_data.items():
        total = 0
        for data_point in data_points:
            total += data_point[field_name]
        summed_data[key] = total
    return summed_data

total_minutes_by_account = sum_grouped_items(engagement_by_account,'total_minutes_visited')

#%pylab inline
#import matplotlib.pyplot as plt
import numpy as np

def describe_data(data):
    sum = list(data.values())
    print('Mean:', np.mean(sum))
    print('Standard deviation:', np.std(sum))
    print('Minimum:', np.min(sum))
    print('Max:', np.max(sum))
    #plt.hist(data)

describe_data(total_minutes_by_account)

lc=sum_grouped_items(engagement_by_account, 'lessons_completed')
describe_data(lc)
print('-----------------')
# has visited in first week
dv = sum_grouped_items(engagement_by_account, 'has_visited')
describe_data(dv)

#no of student passed and disction by using these two keys

subway_project_lesson_keys = ['746169184', '3176718735']

pass_subway_project = set()

for submission in paid_submissions:
    project = submission['lesson_key']
    rating = submission['assigned_rating']

    if ((project in subway_project_lesson_keys) and
            (rating == 'PASSED' or rating == 'DISTINCTION')):
        pass_subway_project.add(submission['account_key'])

print(len(pass_subway_project))
#passing ans non passing in first week
passing_engagement = []
non_passing_engagement = []

for engagement_record in paid_engagement_in_first_week:
    if engagement_record['account_key'] in pass_subway_project:
        passing_engagement.append(engagement_record)
    else:
        non_passing_engagement.append(engagement_record)

print('passing 1st week', len(passing_engagement))
print('non passing 1st week', len(non_passing_engagement))

##
passing_engagement_by_account = group_data(passing_engagement,'account_key')
non_passing_engagement_by_account = group_data(non_passing_engagement,'account_key')
print('non passing stud:')
non_passing_minutes = sum_grouped_items(non_passing_engagement_by_account,'total_minutes_visited')
describe_data(non_passing_minutes)
print('passing stud:')
passing_minutes = sum_grouped_items(passing_engagement_by_account,'total_minutes_visited')
describe_data(passing_minutes)

#no of lessons completed in week for passinf and non passing
passing_engagement_by_account = group_data(passing_engagement,'account_key')
non_passing_engagement_by_account = group_data(non_passing_engagement,'account_key')
print('non passing stud:')
non_passing_minutes = sum_grouped_items(non_passing_engagement_by_account,'lessons_completed')
describe_data(non_passing_minutes)
print('passing stud:')
passing_minutes = sum_grouped_items(passing_engagement_by_account,'lessons_completed')
describe_data(passing_minutes)


#same for has visited
passing_engagement_by_account = group_data(passing_engagement,'account_key')
non_passing_engagement_by_account = group_data(non_passing_engagement,'account_key')
print('non passing stud:')
non_passing_minutes = sum_grouped_items(non_passing_engagement_by_account,'has_visited')
describe_data(non_passing_minutes)
print('passing stud:')
passing_minutes = sum_grouped_items(passing_engagement_by_account,'has_visited')
describe_data(passing_minutes)



#histogram
#data = [1, 2, 1, 3, 3, 1, 4, 2]



#import matplotlib.pyplot as plt
#plt.hist(data)
#plt.show()


#Using pandas build in : red_cvs and find unique rows in file
import pandas as pd
daily_engagement = pd.read_csv('daily_engagement_full (1).csv')
print(len(daily_engagement['acct'].unique()))

import numpy as np
# First 20 countries with employment data
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])

# Employment data in 2007 for those 20 countries
employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])

# Change False to True for each block of code to see what it does

# Accessing elements
if True:
    print(countries[0])
    print(countries[3])

# Slicing
if True:
    print(countries[0:3]) #0 1 2
    print(countries[:3])  # 0, 1 2
    print(countries[17:])  # 17 18 19 20
    print(countries[:])  # all


# Element types
if True:
    print(countries.dtype)
    print(employment.dtype)
    print(np.array([0, 1, 2, 3]).dtype)
    print(np.array([1.0, 1.5, 2.0, 2.5]).dtype)
    print(np.array([True, False, True]).dtype)
    print(np.array(['AL', 'AK', 'AZ', 'AR', 'CA']).dtype)

# Looping
if True:
    for country in countries:
        print('Examining country {}'.format(country))

    for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]
        print('Country {} has employment {}'.format(country,
                country_employment))

# Numpy functions
if True:
    print(employment.mean())
    print(employment.std())
    print(employment.max())
    print(employment.sum())

def max_employment(countries, employment):
    '''
    Fill in this function to return the name of the country
    with the highest employment in the given employment
    data, and the employment in that country.
    '''
    max_country = None      # Replace this with your code
    max_value = None   # Replace this with your code
    i = employment.argmax()
    return (max_country[i], max_value[i])


if True:
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 1, 2])

    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(a ** b)
# Logical operations with NumPy arrays
if False:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])

    print(a & b)
    print(a | b)
    print(~a)

    print(a & True)
    print(a & False)

    print(a | True)
    print(a | False)

import pandas as pd

countries = ['Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
             'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
             'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia']

life_expectancy_values = [74.7,  75. ,  83.4,  57.6,  74.6,  75.4,  72.3,  81.5,  80.2,
                          70.3,  72.1,  76.4,  68.1,  75.2,  69.8,  79.4,  70.8,  62.7,
                          67.3,  70.6]

gdp_values = [ 1681.61390973,   2155.48523109,  21495.80508273,    562.98768478,
              13495.1274663 ,   9388.68852258,   1424.19056199,  24765.54890176,
              27036.48733192,   1945.63754911,  21721.61840978,  13373.21993972,
                483.97086804,   9783.98417323,   2253.46411147,  25034.66692293,
               3680.91642923,    366.04496652,   1175.92638695,   1132.21387981]
life_expectancy = pd.Series(life_expectancy_values)
gdp = pd.Series(gdp_values)

def variable_correlation(variable1, variable2):
    num_same_direction = None        # Replace this with your code
    num_different_direction = None   # Replace this with your code

    both_above = (variable1> variable1.mean()) & \
                 (variable2 > variable2.mean())
    both_below = (variable1 < variable1.mean()) & \
                 (variable2 < variable2.mean())
    is_same_dir = both_above | both_below
    num_same_dir = is_same_dir.sum()
    num_diff_dir = len(variable1) - num_same_dir
    return(num_same_dir, num_diff_dir)

variable_correlation(life_expectancy_values, gdp_values)


import pandas as pd
import seaborn as sns

# The following code reads all the Gapminder data into Pandas DataFrames. You'll
# learn about DataFrames next lesson.

path = '/datasets/ud170/gapminder/'
employment = pd.read_csv(path + 'employment_above_15.csv', index_col='Country')
female_completion = pd.read_csv(path + 'female_completion_rate.csv', index_col='Country')
male_completion = pd.read_csv(path + 'male_completion_rate.csv', index_col='Country')
life_expectancy = pd.read_csv(path + 'life_expectancy.csv', index_col='Country')
gdp = pd.read_csv(path + 'gdp_per_capita.csv', index_col='Country')

# The following code creates a Pandas Series for each variable for the United States.
# You can change the string 'United States' to a country of your choice.

employment_us = employment.loc['United States']
female_completion_us = female_completion.loc['United States']
male_completion_us = male_completion.loc['United States']
life_expectancy_us = life_expectancy.loc['United States']
gdp_us = gdp.loc['United States']
#employment_us.show()
print(employment.index.values)













'''

from collections import defaultdict
def group_data(data, key):
    dt= defaultdict(list)
    for d in data:
        k = d['account_key']
        dt[k].append(dt)
    return dt
ed= list(group_data(paid_engagement_in_first_week, account_key))

def summ(data,name):
    sun = {}
    for key, dp in data.items():
            total = 0
            for d in dp:
                total += d[name]
            sun[key] = total
    return sun
ab = summ(engagement_by_account,'total_minutes_visited')

import numpy as np
def data(d):

    #print('m', np.mean(d))
    print('m', np.min(d))
    print('m', np.max(d))
    print('m', np.std(d))
tm = total_minutes_by_account.values()
data(tm)'''