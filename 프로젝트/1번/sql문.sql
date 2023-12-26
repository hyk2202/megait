CREATE SEQUENCE seq_product START WITH 1 INCREMENT BY 1;
CREATE TABLE product(
    pdno NUMBER NOT NULL,
    pdname VARCHAR2(1200) NOT NULL,
    PRIMARY KEY(pdno)
);
CREATE TABLE review (
    rvno NUMBER NOT NULL,
    rvcontent VARCHAR2(1200) NULL,
    rvdatetime VARCHAR2(20) NOT NULL,
    rvvalue NUMBER NOT NULL,
    rvresult VARCHAR2(10) NOT NULL,
    pdno NUMBER NOT NULL,
    PRIMARY KEY (rvno),
    FOREIGN KEY (pdno) REFERENCES product(pdno)
);
create sequence seq_review START WITH 1 INCREMENT BY 1;


SELECT
  p.pdname,
  SUM(CASE WHEN r.rvresult = '긍정' THEN 1 ELSE 0 END) AS 긍정_개수,
  SUM(CASE WHEN r.rvresult = '부정' THEN 1 ELSE 0 END) AS 부정_개수,
  LISTAGG(r.rvcontent, ', ') WITHIN GROUP (ORDER BY r.rvno) AS 리뷰_내용
FROM
  review r
JOIN
  product p ON r.pdno = p.pdno
GROUP BY
  p.pdname;