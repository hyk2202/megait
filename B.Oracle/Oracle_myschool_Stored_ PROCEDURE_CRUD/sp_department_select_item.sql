CREATE OR REPLACE PROCEDURE sp_department_select_item
(
/** �Ķ���� ���� */
  -- �Ϲ� �Ķ����
  o_deptno        IN      NUMBER,
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
  IF o_deptno IS NULL THEN
    RAISE t_input_exception;
  END IF;

  -- �а� ��� ��ȸ�ϱ� --> ��ȸ ����� o_recordset�� �����Ѵ�.
  OPEN o_recordset FOR
    SELECT deptno, dname, loc FROM department
    WHERE deptno=o_deptno
    ORDER BY deptno ASC;

  -- ������� ����(=0)���� ����
  o_result := 0;

/** ����ó�� */
EXCEPTION
  WHEN t_input_exception THEN
    o_result := 1;
  WHEN others THEN
    RAISE_APPLICATION_ERROR(-20001, SQLERRM);
    o_result := 9; 
    
END sp_department_select_item;
/
