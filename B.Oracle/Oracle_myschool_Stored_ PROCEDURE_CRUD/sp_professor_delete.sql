create or replace PROCEDURE sp_professor_delete
(
/** �Ķ���� ���� */
    -- �Ϲ� �Ķ����
    o_profno    IN      professor.profno%TYPE,
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
    IF o_profno IS NULL THEN
        RAISE t_input_exception;
    END IF;

    -- �������� �����ϱ�
    DELETE FROM professor WHERE profno = o_profno;

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
END sp_professor_delete;
