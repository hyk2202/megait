create or replace PROCEDURE sp_professor_select_item
(
/** 파라미터 선언 */
  -- 일반 파라미터
  o_profno        IN      NUMBER,
  -- 참조 파라미터
  o_result        OUT     NUMBER,
  o_recordset     OUT     SYS_REFCURSOR
)

/** SP 내부에서 사용할 변수 선언 */
IS
  -- 예외를 선언한다.
  t_input_exception  EXCEPTION;

/** 구현할 SQL 구문 작성 */
BEGIN
  -- 파라미터를 검사해서 필수값이 Null이라면 강제로 예외를 발생시킨다.
  -- > 프로시저의 제어가 Exception 블록으로 넘어간다.
  IF o_profno IS NULL THEN
    RAISE t_input_exception;
  END IF;

  -- 학과 목록 조회하기 --> 조회 결과를 o_recordset에 저장한다.
  OPEN o_recordset FOR
    SELECT * FROM professor
    WHERE profno = o_profno;

  -- 결과값을 성공(=0)으로 설정
  o_result := 0;

/** 예외처리 */
EXCEPTION
  WHEN t_input_exception THEN
    o_result := 1;
  WHEN others THEN
    RAISE_APPLICATION_ERROR(-20001, SQLERRM);
    o_result := 9; 

END sp_professor_select_item;
/