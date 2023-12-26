CREATE OR REPLACE PROCEDURE sp_department_select
(
/** �Ķ���� ���� */
   -- ���� �Ķ���� ����
   o_result     OUT     NUMBER,
   o_recordset  OUT     SYS_REFCURSOR
)
/** SP ���ο��� ����� ���� ���� */
IS
    -- ���⼭�� ��� ����
    
/** ������ sql ���� �ۼ� */
BEGIN
    -- �а���� ��ȸ�ϱ� --> ��ȸ ����� O_RECORDSET�� �����Ѵ�.
    OPEN o_recordset FOR
        SELECT deptno, dname, loc FROM department ORDER BY deptno ASC;
    -- ������� ����(=0)���� ����
    o_result := 0;

/** ����ó�� */
EXCEPTION
    WHEN OTHERS THEN
        RAISE_APPLICATION_ERROR(-20001, SQLERRM);
        o_result := 9;
END sp_department_select;
/