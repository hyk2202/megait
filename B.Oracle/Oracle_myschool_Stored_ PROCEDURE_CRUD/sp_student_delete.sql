create or replace PROCEDURE sp_student_delete
(
/** 파라미터 선언 */
   -- 일반 파라미터
   o_studno     IN      student.studno%TYPE,
   -- 참조 파라미터 선언
   o_result     OUT     NUMBER,
   o_rowcount   OUT     NUMBER
)
/** SP 내부에서 사용할 변수 선언 */
IS
    -- 예외 선언
    t_input_exception EXCEPTION;
    t_data_not_found  EXCEPTION;

/** 구현할 sql 구문 작성 */
BEGIN
    -- 파라미터 검사
    IF o_studno IS NULL THEN
        RAISE t_input_exception;
    END IF;

    -- 학과정보 삭제하기
    DELETE FROM student WHERE studno = o_studno;
    
    -- 수정된 행의 수를 조회하기
    o_rowcount := SQL%ROWCOUNT;
    
    -- 수정된 행이 없다면 강제로 에러 발생
    IF o_rowcount < 1 THEN
        RAISE t_data_not_found;
    END IF;

    -- 결과값을 성공(=0)으로 설정
    o_result := 0;

    -- 모든 처리가 종료되었으므로, 변경 사항을 커밋한다
    COMMIT;

/** 예외처리 */
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