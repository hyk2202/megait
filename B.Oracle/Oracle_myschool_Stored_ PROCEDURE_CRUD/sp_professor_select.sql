create or replace PROCEDURE sp_professor_select
(
/** 파라미터 선언 */
   -- 참조 파라미터 선언
   o_result     OUT     NUMBER,
   o_recordset  OUT     SYS_REFCURSOR
)
/** SP 내부에서 사용할 변수 선언 */
IS
    -- 여기서는 사용 안함

/** 구현할 sql 구문 작성 */
BEGIN
    -- 학과목록 조회하기 --> 조회 결과를 O_RECORDSET에 저장한다.
    OPEN o_recordset FOR
        SELECT * FROM professor ORDER BY profno ASC;
    -- 결과값을 성공(=0)으로 설정
    o_result := 0;

/** 예외처리 */
EXCEPTION
    WHEN OTHERS THEN
        RAISE_APPLICATION_ERROR(-20001, SQLERRM);
        o_result := 9;
END sp_professor_select;
/