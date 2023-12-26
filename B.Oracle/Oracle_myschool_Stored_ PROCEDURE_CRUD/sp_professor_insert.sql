create or replace PROCEDURE sp_professor_insert
(
/** �Ķ���� ���� */
    -- �Ϲ� �Ķ����
    o_name          IN      professor.name%TYPE,
    o_userid         IN      professor.userid%TYPE,
    o_position      IN      professor.position%TYPE,
    o_sal              IN      professor.sal%TYPE,
    o_hiredate      IN      professor.hiredate%TYPE,
    o_comm        IN      professor.comm%TYPE,
    o_deptno       IN      professor.deptno%TYPE,
   -- ���� �Ķ���� ����
   o_result          OUT     NUMBER,
   o_profno        OUT     professor.profno%TYPE
)
/** SP ���ο��� ����� ���� ���� */
IS
    -- ���� ����
    t_input_exception EXCEPTION;

/** ������ sql ���� �ۼ� */
BEGIN

    -- ����� �Ϸù�ȣ ä���ϱ� --> ��ȸ ����� o_deptno�� �����Ѵ�.
    SELECT seq_professor.NEXTVAL INTO o_profno FROM DUAL;
    
    -- �Ķ���� �˻�
    IF o_profno IS NULL THEN
        o_profno := 0;
        RAISE t_input_exception;
    END IF;

    -- �а����� �߰��ϱ�
    INSERT INTO professor(profno, name, userid, position, sal, hiredate, comm, deptno)
    VALUES(o_profno, o_name, o_userid, o_position, o_sal, o_hiredate, o_comm, o_deptno);

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
END sp_professor_insert;
/