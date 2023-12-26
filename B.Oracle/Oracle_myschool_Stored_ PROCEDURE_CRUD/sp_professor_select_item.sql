create or replace PROCEDURE sp_professor_select_item
(
/** �Ķ���� ���� */
  -- �Ϲ� �Ķ����
  o_profno        IN      NUMBER,
  -- ���� �Ķ����
  o_result        OUT     NUMBER,
  o_recordset     OUT     SYS_REFCURSOR
)

/** SP ���ο��� ����� ���� ���� */
IS
  -- ���ܸ� �����Ѵ�.
  t_input_exception  EXCEPTION;

/** ������ SQL ���� �ۼ� */
BEGIN
  -- �Ķ���͸� �˻��ؼ� �ʼ����� Null�̶�� ������ ���ܸ� �߻���Ų��.
  -- > ���ν����� ��� Exception ������� �Ѿ��.
  IF o_profno IS NULL THEN
    RAISE t_input_exception;
  END IF;

  -- �а� ��� ��ȸ�ϱ� --> ��ȸ ����� o_recordset�� �����Ѵ�.
  OPEN o_recordset FOR
    SELECT * FROM professor
    WHERE profno = o_profno;

  -- ������� ����(=0)���� ����
  o_result := 0;

/** ����ó�� */
EXCEPTION
  WHEN t_input_exception THEN
    o_result := 1;
  WHEN others THEN
    RAISE_APPLICATION_ERROR(-20001, SQLERRM);
    o_result := 9; 

END sp_professor_select_item;
/