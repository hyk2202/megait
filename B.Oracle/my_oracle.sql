select dname,deptno,loc from department;
select distinct deptno from student;
select distinct deptno, grade from student;
select dname as department_name, deptno as department_number from department;
select name,sal,sal*12+100 as 연봉 from professor;
--select name,sal,sal*12+100 연봉 from professor; 위와 동일함

select studno, name, deptno,weight from student where grade=1 and weight<=90;

-- 학생 테이블에서 학생 이름과 학생 번호를 조회하시오
select studno, name from student;

-- 교수 테이블(professor)에서 직급(position)의 종류를 조회하시오 (직급의 종류는 저장된 직급 데이터들의 중복을 제거하면 알 수 있다.)
--select distinct position from professor;
select position from professor GROUP BY POSITION;

-- department 테이블을 사용하여 deptno 를 부서 , dname를 부서명, loc 를 위치로 별명을 설정하여 출력하세요
select deptno as 부서, dname as 부서명, loc as 위치 from department;
--select deptno 부서, dname 부서명, loc 위치 from department;

-- 학생 테이블(student)에서 학생 이름(name)과 각 학생에 대한표준 체중을 조회. (단, 표준 체중은 (키(height)-110)*0.9 로 구하시오.)
select name, (height-110)*0.9 as 표준_체중 from student;

-- 학생테이블에서 학과번호가 101번인 학생들의 학번 , 이름, 학년을 출력하시오.
select studno,name,grade from student where deptno = 101;

-- 교수테이블에서 학과번호가 101번인 교수들의 교수번호, 이름,급여를 출력하시오.
select profno,name,sal from professor where deptno = 101;

-- 학생테이블에서 키가 170 이상인 학생의 학번, 이름, 학년, 학과번호, 키를 출력하시오
select studno, name,grade, deptno, height from student where height >=170;

-- 학생 테이블에서 1학년 이거나 몸무게가 70kg 이상인 학생만 검색하여 이름, 학번, 학년, 몸무게, 학과 번호를 출력하시오
select name, studno, grade, weight, deptno from student where grade =1 or weight >= 70;

select studno, name, weight from student where weight between 50 and 70;
--select studno, name, weight from student where weight >= 50 and weight <= 70;

select name, grade, deptno from student where deptno in (102,201);
--select name, grade, deptno from student where deptno = 102 or deptno = 201;

select name, grade, deptno from student where name like '진%';
select name, grade, deptno from student where name like '%진';
select name, grade, deptno from student where name like '%진%';

select name, position, comm from professor;
select name, position, comm from professor where comm is null;
select name, position, comm from professor where comm is not null;

select name, grade, deptno from student where deptno=102 and (grade =4 or grade = 1);
// 비교
select name, grade, deptno from student where deptno=102 and grade =4 or grade = 1;

select name, sal from professor where sal between 300 and 400;
select profno, name, position, deptno from professor where position in ('조교수','전임강사');
select deptno, dname, loc from department where dname like '%공학%';
select studno, name, grade, profno from student where profno is not null;
select name, grade, deptno from student where grade =1 or deptno=102 and grade =4; 

select name, grade, tel from student order by name;
select studno, name, grade, deptno, userid from student order by deptno, grade desc;
select rownum as rnum, tbl.* from(
    select deptno, dname from department order by deptno
) tbl;

select * from(
    select rownum as rnum, tbl.* from(
        select name,position,sal from professor order by sal desc
    )tbl where rownum<=3
)where rnum>0;

desc professor;

select name, grade, idnum from student order by grade desc;
select name, grade, deptno from student where deptno = 101 order by birthdate;
select name, studno, grade from student order by grade, name;

 select * from (
    select rownum as rnum, tbl.* from (
        select name, position, sal from professor order by sal desc
    )tbl where rownum <=5
 ) where rnum >4; -- where runm=5

select name, length(name) from student;
select name, substr(name,1,1) from student;
select name, substr(name,length(name)) from student;
select name, substr(name,2,1) from student;
select name, substr(name,2) from student;
select name, replace(name,'이','lee') from student;

select concat(name,grade) from student;
--select name||grade from student;

select concat(concat(concat(name,' '),grade), '학년') from student;
-- select name||' '||grade||'학년' from student;

select trim(name), ltrim(name), rtrim(name) from student;

select instr(name,'이'), name from student;
select upper(userid), lower(userid),userid from student;

select to_char(sysdate,'yyyy-mm-dd HH24:mi:ss') from dual;
select sysdate from dual;
select to_char(sysdate,'yyyymmddHH24miss') from dual;
select to_char(sysdate+100,'yyyy-mm-dd'), to_char(sysdate-7,'yyyy-mm-dd') from dual;

select name, replace(name,substr(name,2,1),'*') from student;
select name, replace(idnum,substr(idnum,length(idnum)-6),'*******') from student;
--select name, substr(idnum,1,6)||'*******' from student;
select name, birthdate from student where to_char(birthdate,'yyyy') >=1980;

select count(studno) from student where grade=3;
select comm from professor where deptno = 101;

select count(comm) from professor where deptno=101;
select count(*) from professor where deptno=101;
select max(sal) from professor;
select min(sal) from professor;
select sum(sal) from professor;
select avg(height) from student;
select avg(weight), sum(weight) from student where deptno = 101;
select deptno, name from professor order by deptno;
--select deptno, name from professor group by deptno; 에러발생(name칼럼에 대한 처리가 없음)
select deptno, count(name) from professor group by deptno;
select deptno, count(*),count(comm) from professor group by deptno;
select deptno,grade,count(*),avg(weight) from student group by deptno,grade;

select grade, count(*), avg(height) avg_height, avg(weight) avg_weight 
from student 
group by grade
order by avg_height desc;

select grade, count(*), avg(height) avg_height, avg(weight) avg_weight 
from student 
group by grade
having count(*) > 4
order by avg_height desc;

select deptno, grade, count(*), max(height), max(weight)
from student
group by deptno, grade
having count(*) >=3
order by deptno;

select max(height), min(height) 
from student
where deptno = 101;

select deptno, avg(sal),min(sal),max(sal) 
from professor
group by deptno;

select avg(weight) avg_weight, count(*)
from student
group by deptno
order by avg_weight desc;

select deptno, count(*)
from professor 
group by deptno
having count(*) <=2
order by deptno;

select p.name, d.deptno, d.dname 
from professor p, department d
where p.deptno = d.deptno and deptno = 101; -- equi join

select p.name, d.deptno, d.dname 
from professor p
inner join department d -- inner 생략가능
on p.deptno = d.deptno 
WHERE deptno = 101;-- inner join

select s.name, p.name
from student s
left outer join professor p -- outer 생략가능
on s.profno = p. profno; -- left outer join의 경우 왼쪽에 지정된 테이블의 모든 데이터 출력을 보장한다

select s.name, p.name
from student s, professor p
where s.profno=p.profno(+);

select name, position from professor
where position = (
select position from professor where name = '전은지'
);

select name, deptno, grade, height from student
where grade = 1 and height > (
select avg(height) from student
);

select name, dname from student s
join department d on s.deptno = d.deptno
where s.deptno = (select deptno from student where name = '이광훈');

select studno, grade, name from student 
where profno in (select profno from professor where sal>300);

select s.studno, s.name, s.deptno, d.dname, d.loc from student s, department d
where s.deptno = d.deptno;

select s.studno, s.name, s.deptno, d.dname, s.grade from student s, department d
where s.deptno = d.deptno and s.deptno=102;

select s.name, s.grade, p.name, p.position
from student s, professor p
where s.profno = p.profno;

select s.name, s.grade, p.name, p.position
from student s
join professor p on s.profno = p.profno;

select name, grade from student
where grade = (select grade from student where UsERID = 'jun123');

select name, deptno, weight from student
where weight < (
    select avg(weight) from student where deptno = 101
);

select s.name, s.weight, d.dname, p.name from student s
join department d on s.deptno = d.deptno
join professor p on s.profno = p.profno
where s.weight < (
    select avg(weight) from student 
    where deptno = (select deptno from student where name = '이광훈')
);    

select name, grade, height from student
where grade = (select grade from student where studno = '20101')
and height >(select height from student where studno = '20101');

select s.studno, d.dname, s.grade, s.name from student s
join department d on s.deptno = d.deptno 
where s.deptno in (select deptno from department where dname like '%공학%');
