import pymysql
import pandas as pd

if __name__ == '__main__':
    my_db = pymysql.connect(
        user='root',
        passwd='your password',   #자신의 비번 입력
        host='127.0.0.1',
        port=3306,
        db='shopdb',
        charset='utf8'
    )
    cursor = my_db.cursor((pymysql.cursors.DictCursor))
    sql = 'select * from buytbl where userID = "KHD";'
    cursor.execute(sql)
    resp = cursor.fetchall()
    my_db.close()
    print(resp)
    df = pd.DataFrame(resp)
    print(df)
    df.info()
    print(df.describe())




