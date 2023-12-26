create or replace PROCEDURE sp_professor_update
(
/** �Ķ���� ���� */
    -- �Ϲ� �Ķ����
    o_profno        IN      professor.profno%TYPE,
    o_name          IN      professor.name%TYPE,
    o_userid         IN      professor.userid%TYPE,
    o_position      IN      professor.position%TYPE,
    o_sal              IN      professor.sal%TYPE,
    o_hiredate      IN      professor.hiredate%TYPE,
    o_comm        IN      professor.comm%TYPE,
    o_deptno       IN      professor.deptno%TYPE,

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
    IF o_profno IS NULL OR o_name IS NULL 
        OR o_userid IS NULL OR o_position IS NULL 
        OR o_sal IS NULL OR o_hiredate IS NULL 
        OR o_deptno IS NULL THEN
        RAISE t_input_exception;
    END IF;

    -- �а����� �����ϱ�
    UPDATE professor SET name = o_name, 
        userid = o_userid, position = o_position, 
        sal = o_sal, hiredate = o_hiredate, 
        comm = o_comm, deptno = o_deptno
    WHERE profno = o_profno;

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

END sp_professor_update;
/