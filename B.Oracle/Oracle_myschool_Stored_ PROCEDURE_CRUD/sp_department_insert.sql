create or replace PROCEDURE sp_department_insert
(
/** �Ķ���� ���� */
    -- �Ϲ� �Ķ����
    o_dname     IN      department.dname%TYPE,
    o_loc       IN      department.loc%TYPE,
   -- ���� �Ķ���� ����
   o_result     OUT     NUMBER,
   o_deptno  OUT     department.deptno%TYPE
)
/** SP ���ο��� ����� ���� ���� */
IS
    -- ���� ����
    t_input_exception EXCEPTION;

/** ������ sql ���� �ۼ� */
BEGIN
    -- ����� �Ϸù�ȣ ä���ϱ� --> ��ȸ ����� o_deptno�� �����Ѵ�.
    SELECT seq_department.NEXTVAL INTO o_deptno FROM DUAL;

    -- �Ķ���� �˻�
    IF o_dname IS NULL THEN
        o_deptno := 0;
        RAISE t_input_exception;
    END IF;
    
    
    -- �а����� �߰��ϱ�
    INSERT INTO department(deptno, dname, loc)
    VALUES(o_deptno, o_dname, o_loc);
    
    -- ������� ����(=0)���� ����
    o_result := 0;
    
    -- ��� ó���� ����Ǿ����Ƿ�, ���� ������ Ŀ���Ѵ�
    COMMIT;

/** ����ó�� */
EXCEPTION
    WHEN t_input_exception THEN
        o_result := 1;
        ROLLBACK;
    WHEN OTHERS THEN
        RAISE_APPLICATION_ERROR(-20001, SQLERRM);
        o_result := 9;
        ROLLBACK;
END sp_department_insert;
/