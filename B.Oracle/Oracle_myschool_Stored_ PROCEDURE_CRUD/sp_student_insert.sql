create or replace PROCEDURE sp_student_insert
(
/** �Ķ���� ���� */
    -- �Ϲ� �Ķ����
    o_name          IN      student.name%TYPE,
    o_userid        IN      student.userid%TYPE,
    o_grade         IN      student.grade%TYPE,
    o_idnum         IN      student.idnum%TYPE,
    o_birthdate     IN      student.birthdate%TYPE,
    o_tel           IN      student.tel%TYPE,
    o_height        IN      student.height%TYPE,
    o_weight        IN      student.weight%TYPE,
    o_deptno        IN      student.deptno%TYPE,
    o_profno        IN      student.profno%TYPE,
   -- ���� �Ķ���� ����
   o_result         OUT     NUMBER,
   o_studno         OUT     student.studno%TYPE
)
/** SP ���ο��� ����� ���� ���� */
IS
    -- ���� ����
    t_input_exception EXCEPTION;

/** ������ sql ���� �ۼ� */
BEGIN

    -- ����� �Ϸù�ȣ ä���ϱ� --> ��ȸ ����� o_deptno�� �����Ѵ�.
    SELECT seq_student.NEXTVAL INTO o_studno FROM DUAL;
    
    -- �Ķ���� �˻�
    IF o_studno IS NULL THEN
        o_studno := 0;
        RAISE t_input_exception;
    END IF;

    -- �а����� �߰��ϱ�
    INSERT INTO student(studno, name, userid, grade, idnum, birthdate, tel, height, weight, deptno, profno)
    VALUES(o_studno, o_name, o_userid, o_grade, o_idnum, o_birthdate, o_tel, o_height, o_weight, o_deptno, o_profno);

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
END sp_student_insert;
/