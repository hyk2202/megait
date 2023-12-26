create or replace PROCEDURE sp_student_update
(
/** �Ķ���� ���� */
    -- �Ϲ� �Ķ����
    o_studno        IN      student.studno%TYPE,
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
    IF o_studno IS NULL OR o_name IS NULL 
        OR o_userid IS NULL OR o_grade IS NULL 
        OR o_idnum IS NULL OR o_birthdate IS NULL 
        OR o_tel IS NULL OR o_height IS NULL
        OR o_weight IS NULL OR o_deptno IS NULL THEN
        RAISE t_input_exception;
    END IF;

    -- �а����� �����ϱ�
    UPDATE student SET studno = o_studno, name = o_name, 
        userid = o_userid, grade = o_grade, 
        idnum = o_idnum, birthdate = o_birthdate, 
        tel = o_tel, height = o_height, 
        weight = o_weight, deptno = o_deptno, profno = o_profno
    WHERE studno = o_studno;

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

END sp_student_update;
/