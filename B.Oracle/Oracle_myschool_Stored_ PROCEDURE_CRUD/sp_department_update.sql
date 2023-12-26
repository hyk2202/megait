CREATE OR REPLACE PROCEDURE sp_department_update
(
/** �Ķ���� ���� */
    -- �Ϲ� �Ķ����
    o_deptno    IN      department.deptno%TYPE,
    o_dname     IN      department.dname%TYPE,
    o_loc       IN      department.loc%TYPE,
   -- ���� �Ķ���� ����
   o_result     OUT     NUMBER,
   o_rowcount   OUT     NUMBER
)
/** SP ���ο��� ����� ���� ���� */
IS
    -- ���� ����
    t_input_exception EXCEPTION; -- �Ķ���Ͱ� �������� �������
    t_data_not_found  EXCEPTION; -- �Է�, ����, ������ ���� ���� 0�ΰ��

/** ������ sql ���� �ۼ� */
BEGIN
    -- �Ķ���� �˻�
    IF o_dname IS NULL OR o_deptno IS NULL THEN
        RAISE t_input_exception;
    END IF;

    -- �а����� �����ϱ�
    UPDATE department SET dname = o_dname, loc = o_loc WHERE deptno = o_deptno;

    -- ������ ���� ���� ��ȸ�ϱ�
    o_rowcount := SQL%ROWCOUNT;
    
    -- ������ ���� ���ٸ� ������ ���� �߻�
    IF o_rowcount < 1 THEN
        RAISE t_data_not_found;
    END IF;

    -- ������� ����(=0)���� ����
    o_result := 0;

    -- ��� ó���� ����Ǿ����Ƿ�, ���� ������ Ŀ���Ѵ�
    COMMIT;

/** ����ó�� */
EXCEPTION
    WHEN t_input_exception THEN
        o_result := 1;
        ROLLBACK;
    WHEN t_data_not_found THEN
        o_result := 2;
        ROLLBACK;
    WHEN OTHERS THEN
        RAISE_APPLICATION_ERROR(-20001, SQLERRM);
        o_result := 9;
        ROLLBACK;
        
END sp_department_update;
/