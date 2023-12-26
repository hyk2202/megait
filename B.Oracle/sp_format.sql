CREATE OR REPLACE PROCEDURE 프로시져이름
(
/** 파라미터 선언 */
  -- 참조 파라미터 선언

)

/** SP 내부에서 사용할 변수 선언 */
IS
    

/** 구현할 SQL구문 작성 */
BEGIN
    

/** 예외처리 */
EXCEPTION

    WHEN others THEN
        RAISE_APPLICATION_ERROR(-20001,SQLERRM);
        -- 오라클에서는 에러가 발생해도 외부 프로그래밍언어/툴에서는 에러가 발생하지 않아서 결과값을 보고 체크한다.

END 프로시져이름t;
/