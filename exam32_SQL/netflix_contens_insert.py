import pandas as pd
pd.set_option('display.max_columns', 20)
import pymysql
import numpy as np
import pickle
#print(np.nan)
# a = np.nan
# if a == np.nan:
#     print('is nan')
# elif a > np.nan:
#     print('a > nan')
# elif a < np.nan:
#     print('a < nan')
# elif np.isnan(a):
#     print('a is nan')
# else:
#     print(' nan is nan')
#

df = pd.read_csv('../datasets/netflix_titles.csv')
#print(df.head())
# print(df.tail())
df.info()

df.fillna('', inplace=True)
for i in range(len(df)):
    for j in range(12):
        df.iloc[i, j] = df.iloc[i, j].replace('"', '%\\"')
        df.iloc[i, j] = df.iloc[i, j].replace("'", "\\'")

# df = df.replace("'", "\\'")
# df = df.replace('"', '%\\"')
#print(df.head(20))

conn = pymysql.connect(
        user='root',
        passwd='jsl10204^^',  # 자신의 비번 입력
        host='127.0.0.1',
        port=3306,
        db='netflix',
        charset='utf8'
    )

errors = []

for i in range(len(df)):
    try:
        sql = '''insert into netflix_contents value(
                          {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});'''.format(
            '"{}"'.format(df.iloc[i, 0]),
            '"{}"'.format(df.iloc[i, 1]),
            '"{}"'.format(df.iloc[i, 2]),
            'null' if df.iloc[i, 3] == '' else '"{}"'.format(df.iloc[i, 3]),
            'null' if df.iloc[i, 4] == '' else '"{}"'.format(df.iloc[i, 4]),
            'null' if df.iloc[i, 5] == '' else '"{}"'.format(df.iloc[i, 5]),
            'null' if df.iloc[i, 6] == '' else '"{}"'.format(df.iloc[i, 6]),
            df.iloc[i,7],
            'null' if df.iloc[i, 7] == '' else '"{}"'.format(df.iloc[i, 8]),
            'null' if df.iloc[i, 8] == '' else '"{}"'.format(df.iloc[i, 9]),
            '"{}"'.format(df.iloc[i, 10]),
            '"{}"'.format(df.iloc[i, 11]))
        print(sql)
        with conn.cursor() as cursor:
            cursor.execute(sql)
            conn.commit()
    except:
        errors.append(i)
        print(i)

conn.close()
with open('./errors.pickle', 'wb') as f:
    pickle.dump(errors, f)


# a = "\'hello\' \"world\""
# print(a)

# print(df.description.head())
# #print(df.duration.unique())
# max_length = 0
# df_description = df.description.dropna(inplace=False)
# df_description.reset_index(drop=True, inplace=True)
# for i, con in enumerate(df_description):
#     if max_length < len(con):
#         max_length = len(con)
#
# # max_value = 0
# # for i, year in enumerate(df_release_year):
# #     if max_value < year:
# #         max_value = year
#
# print(max_length)
















