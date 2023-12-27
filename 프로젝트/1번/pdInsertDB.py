def pdInsertDB(cursor, dbcon, title):
    sql = 'select pdname from product where pdname = :1'
    cursor.execute(sql, [title])
    if not cursor.fetchall(): # 이미 해당 데이터가 DB에 있으면 조회만하고 넘어간다.
        sql = f'INSERT INTO product (pdno, pdname) VALUES (seq_product.nextval, :1)'
        cursor.execute(sql, [title])
        print('입력되었습니다')
        dbcon.commit()

        