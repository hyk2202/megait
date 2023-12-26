create or replace PROCEDURE sp_student_delete
(
/** �Ķ���� ���� */
   -- �Ϲ� �Ķ����
   o_studno     IN      student.studno%TYPE,
   -- ���� �Ķ���� ����
   o_result     OUT     NUMBER,
   o_rowcount   OUT     NUMBER
)
/** SP ���ο��� ����� ���� ���� */
IS
    -- ���� ����
    t_input_exception EXCEPTION;
    t_data_not_found  EXCEPTION;

/** ������ sql ���� �ۼ� */
BEGIN
    -- �Ķ���� �˻�
    IF o_studno IS NULL THEN
        RAISE t_input_exception;
    END IF;

    -- �а����� �����ϱ�
    DELETE FROM student WHERE studno = o_studno;
    
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
END sp_student_delete;
/