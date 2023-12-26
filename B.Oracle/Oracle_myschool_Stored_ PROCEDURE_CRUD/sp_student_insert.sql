create or replace PROCEDURE sp_student_insert
(
/** 파라미터 선언 */
    -- 일반 파라미터
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
   -- 참조 파라미터 선언
   o_result         OUT     NUMBER,
   o_studno         OUT     student.studno%TYPE
)
/** SP 내부에서 사용할 변수 선언 */
IS
    -- 예외 선언
    t_input_exception EXCEPTION;

/** 구현할 sql 구문 작성 */
BEGIN

    -- 저장될 일련번호 채집하기 --> 조회 결과를 o_deptno에 저장한다.
    SELECT seq_student.NEXTVAL INTO o_studno FROM DUAL;
    
    -- 파라미터 검사
    IF o_studno IS NULL THEN
        o_studno := 0;
        RAISE t_input_exception;
    END IF;

    -- 학과정보 추가하기
    INSERT INTO student(studno, name, userid, grade, idnum, birthdate, tel, height, weight, deptno, profno)
    VALUES(o_studno, o_name, o_userid, o_grade, o_idnum, o_birthdate, o_tel, o_height, o_weight, o_deptno, o_profno);

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
END sp_student_insert;
/