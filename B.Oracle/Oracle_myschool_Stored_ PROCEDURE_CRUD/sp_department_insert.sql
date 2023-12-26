create or replace PROCEDURE sp_department_insert
(
/** 파라미터 선언 */
    -- 일반 파라미터
    o_dname     IN      department.dname%TYPE,
    o_loc       IN      department.loc%TYPE,
   -- 참조 파라미터 선언
   o_result     OUT     NUMBER,
   o_deptno  OUT     department.deptno%TYPE
)
/** SP 내부에서 사용할 변수 선언 */
IS
    -- 예외 선언
    t_input_exception EXCEPTION;

/** 구현할 sql 구문 작성 */
BEGIN
    -- 저장될 일련번호 채집하기 --> 조회 결과를 o_deptno에 저장한다.
    SELECT seq_department.NEXTVAL INTO o_deptno FROM DUAL;

    -- 파라미터 검사
    IF o_dname IS NULL THEN
        o_deptno := 0;
        RAISE t_input_exception;
    END IF;
    
    
    -- 학과정보 추가하기
    INSERT INTO department(deptno, dname, loc)
    VALUES(o_deptno, o_dname, o_loc);
    
    -- 결과값을 성공(=0)으로 설정
    o_result := 0;
    
    -- 모든 처리가 종료되었으므로, 변경 사항을 커밋한다
    COMMIT;

/** 예외처리 */
EXCEPTION
    WHEN t_input_exception THEN
        o_result := 1;
        ROLLBACK;
    WHEN OTHERS THEN
        RAISE_APPLICATION_ERROR(-20001, SQLERRM);
        o_result := 9;
        ROLLBACK;
END sp_department_insert;
/