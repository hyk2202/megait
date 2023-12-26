DROP SEQUENCE seq_department;
DROP SEQUENCE seq_student;
DROP SEQUENCE seq_professor;
COMMIT;

DROP TABLE student;
COMMIT;

DROP TABLE professor;
COMMIT;

DROP TABLE department;
COMMIT;

-- ----------------------------
--  Table structure for department
-- ----------------------------
CREATE SEQUENCE seq_department START WITH 300 INCREMENT BY 1;

CREATE TABLE department (
  deptno number(4) NOT NULL,
  dname varchar2(50) NOT NULL,
  loc varchar2(50) DEFAULT NULL,
  PRIMARY KEY (deptno)
);

-- ----------------------------
--  Records of department
-- ----------------------------
INSERT INTO department VALUES (101, '컴퓨터공학과', '1호관');
INSERT INTO department VALUES (102, '멀티미디어학과', '2호관');
INSERT INTO department VALUES (201, '전자공학과', '3호관');
INSERT INTO department VALUES (202, '기계공학과', '4호관');
COMMIT;

-- ----------------------------
--  Table structure for professor
-- ----------------------------
CREATE SEQUENCE seq_professor START WITH 9910 INCREMENT BY 1;

CREATE TABLE professor (
  profno number(11) NOT NULL,
  name varchar2(10) NOT NULL,
  userid varchar2(10) NOT NULL,
  position varchar2(20) NOT NULL,
  sal number(10) NOT NULL,
  hiredate date NOT NULL,
  comm number(11) DEFAULT NULL,
  deptno number(11) NOT NULL,
  PRIMARY KEY (profno),
  CONSTRAINT fk_professor_department FOREIGN KEY (deptno) REFERENCES department (deptno)
);

-- ----------------------------
--  Records of professor
-- ----------------------------
INSERT INTO professor VALUES (9901, '김도훈', 'capool', '교수', 500, TO_DATE('1982-06-12 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), 20, 101);
INSERT INTO professor VALUES (9902, '이재우', 'sweat413', '조교수', 320, TO_DATE('1995-04-12 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), null, 201);
INSERT INTO professor VALUES (9903, '성연희', 'pascal', '조교수', 360, TO_DATE('1993-03-17 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), 15, 101);
INSERT INTO professor VALUES (9904, '염일웅', 'blue77', '전임강사', 240, TO_DATE('1998-10-11 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), null, 102);
INSERT INTO professor VALUES (9905, '권혁일', 'refresh', '교수', 450, TO_DATE('1986-02-11 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), 25, 102);
INSERT INTO professor VALUES (9906, '이만식', 'pocari', '부교수', 420, TO_DATE('1988-07-11 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), null, 101);
INSERT INTO professor VALUES (9907, '전은지', 'totoro', '전임강사', 210, TO_DATE('2001-05-11 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), null, 101);
INSERT INTO professor VALUES (9908, '남은혁', 'bird13', '부교수', 400, TO_DATE('1990-10-18 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), 17, 202);
COMMIT;

-- ----------------------------
--  Table structure for student
-- ----------------------------
CREATE SEQUENCE seq_student START WITH 20105 INCREMENT BY 1;

CREATE TABLE student (
  studno number(11) NOT NULL,
  name varchar2(10) NOT NULL,
  userid varchar2(10) NOT NULL,
  grade number(11) NOT NULL,
  idnum varchar2(13) NOT NULL,
  birthdate date NOT NULL,
  tel varchar2(13) NOT NULL,
  height number(11) NOT NULL,
  weight number(11) NOT NULL,
  deptno number(11) NOT NULL,
  profno number(11) DEFAULT NULL,
  PRIMARY KEY (studno),
  CONSTRAINT fk_student_department FOREIGN KEY (deptno) REFERENCES department (deptno),
  CONSTRAINT fk_student_professor FOREIGN KEY (profno) REFERENCES professor (profno)
);

-- ----------------------------
--  Records of student
-- ----------------------------
INSERT INTO student VALUES ('10101', '전인하', 'jun123', '4', '7907021369824', TO_DATE('1979-07-02 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '051)781-2158', '176', '72', '101', '9903');
INSERT INTO student VALUES ('10102', '박미경', 'ansel414', '1', '8405162123648', TO_DATE('1984-05-16 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '055)261-8947', '168', '52', '101', null);
INSERT INTO student VALUES ('10103', '김영균', 'mandu', '3', '8103211063421', TO_DATE('1981-03-21 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '051)824-9637', '170', '88', '101', '9906');
INSERT INTO student VALUES ('10104', '지은경', 'gomo00', '2', '8004122298371', TO_DATE('1980-04-12 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '055)418-9627', '161', '42', '101', '9907');
INSERT INTO student VALUES ('10105', '임유진', 'youjin12', '2', '8301212196482', TO_DATE('1983-01-21 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '02)312-9838', '171', '54', '101', '9907');
INSERT INTO student VALUES ('10106', '서재진', 'seolly', '1', '8511291186273', TO_DATE('1985-11-29 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '051)239-4861', '186', '72', '101', null);
INSERT INTO student VALUES ('10107', '이광훈', 'huriky', '4', '8109131276431', TO_DATE('1981-09-13 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '055)736-4981', '175', '92', '101', '9903');
INSERT INTO student VALUES ('10108', '류민정', 'cleansky', '2', '8108192157498', TO_DATE('1981-08-19 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '055)248-3679', '162', '72', '101', '9907');
INSERT INTO student VALUES ('10201', '김진영', 'simply', '2', '8206062186327', TO_DATE('1982-06-06 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '055)419-6328', '164', '48', '102', '9905');
INSERT INTO student VALUES ('10202', '오유석', 'yousuk', '4', '7709121128379', TO_DATE('1977-09-12 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '051)724-9618', '177', '92', '102', '9905');
INSERT INTO student VALUES ('10203', '하나리', 'hanal', '1', '8501092378641', TO_DATE('1985-01-09 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '055)296-3784', '160', '68', '102', null);
INSERT INTO student VALUES ('10204', '윤진욱', 'samba7', '3', '7904021358671', TO_DATE('1979-04-02 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '053)487-2698', '171', '70', '102', '9905');
INSERT INTO student VALUES ('20101', '이동훈', 'dals', '1', '8312101128467', TO_DATE('1983-12-10 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '055)426-1752', '172', '64', '201', null);
INSERT INTO student VALUES ('20102', '박동진', 'ping2', '1', '8511241639826', TO_DATE('1985-11-24 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '051)742-6384', '182', '70', '201', null);
INSERT INTO student VALUES ('20103', '김진경', 'lovely', '2', '8302282169387', TO_DATE('1983-02-28 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '052)175-3941', '166', '51', '201', '9902');
INSERT INTO student VALUES ('20104', '조명훈', 'rader214', '1', '8412141254963', TO_DATE('1984-12-14 00:00:00', 'YYYY-MM-DD HH24:MI:SS'), '02)785-6984', '184', '62', '201', null);
COMMIT;





