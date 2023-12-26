select dname,deptno,loc from department;
select distinct deptno from student;
select distinct deptno, grade from student;
select dname as department_name, deptno as department_number from department;
select name,sal,sal*12+100 as ���� from professor;
--select name,sal,sal*12+100 ���� from professor; ���� ������

select studno, name, deptno,weight from student where grade=1 and weight<=90;

-- �л� ���̺��� �л� �̸��� �л� ��ȣ�� ��ȸ�Ͻÿ�
select studno, name from student;

-- ���� ���̺�(professor)���� ����(position)�� ������ ��ȸ�Ͻÿ� (������ ������ ����� ���� �����͵��� �ߺ��� �����ϸ� �� �� �ִ�.)
--select distinct position from professor;
select position from professor GROUP BY POSITION;

-- department ���̺��� ����Ͽ� deptno �� �μ� , dname�� �μ���, loc �� ��ġ�� ������ �����Ͽ� ����ϼ���
select deptno as �μ�, dname as �μ���, loc as ��ġ from department;
--select deptno �μ�, dname �μ���, loc ��ġ from department;

-- �л� ���̺�(student)���� �л� �̸�(name)�� �� �л��� ����ǥ�� ü���� ��ȸ. (��, ǥ�� ü���� (Ű(height)-110)*0.9 �� ���Ͻÿ�.)
select name, (height-110)*0.9 as ǥ��_ü�� from student;

-- �л����̺��� �а���ȣ�� 101���� �л����� �й� , �̸�, �г��� ����Ͻÿ�.
select studno,name,grade from student where deptno = 101;

-- �������̺��� �а���ȣ�� 101���� �������� ������ȣ, �̸�,�޿��� ����Ͻÿ�.
select profno,name,sal from professor where deptno = 101;

-- �л����̺��� Ű�� 170 �̻��� �л��� �й�, �̸�, �г�, �а���ȣ, Ű�� ����Ͻÿ�
select studno, name,grade, deptno, height from student where height >=170;

-- �л� ���̺��� 1�г� �̰ų� �����԰� 70kg �̻��� �л��� �˻��Ͽ� �̸�, �й�, �г�, ������, �а� ��ȣ�� ����Ͻÿ�
select name, studno, grade, weight, deptno from student where grade =1 or weight >= 70;

select studno, name, weight from student where weight between 50 and 70;
--select studno, name, weight from student where weight >= 50 and weight <= 70;

select name, grade, deptno from student where deptno in (102,201);
--select name, grade, deptno from student where deptno = 102 or deptno = 201;

select name, grade, deptno from student where name like '��%';
select name, grade, deptno from student where name like '%��';
select name, grade, deptno from student where name like '%��%';

select name, position, comm from professor;
select name, position, comm from professor where comm is null;
select name, position, comm from professor where comm is not null;

select name, grade, deptno from student where deptno=102 and (grade =4 or grade = 1);
// ��
select name, grade, deptno from student where deptno=102 and grade =4 or grade = 1;

select name, sal from professor where sal between 300 and 400;
select profno, name, position, deptno from professor where position in ('������','���Ӱ���');
select deptno, dname, loc from department where dname like '%����%';
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
select name, replace(name,'��','lee') from student;

select concat(name,grade) from student;
--select name||grade from student;

select concat(concat(concat(name,' '),grade), '�г�') from student;
-- select name||' '||grade||'�г�' from student;

select trim(name), ltrim(name), rtrim(name) from student;

select instr(name,'��'), name from student;
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
--select deptno, name from professor group by deptno; �����߻�(nameĮ���� ���� ó���� ����)
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
inner join department d -- inner ��������
on p.deptno = d.deptno 
WHERE deptno = 101;-- inner join

select s.name, p.name
from student s
left outer join professor p -- outer ��������
on s.profno = p. profno; -- left outer join�� ��� ���ʿ� ������ ���̺��� ��� ������ ����� �����Ѵ�

select s.name, p.name
from student s, professor p
where s.profno=p.profno(+);

select name, position from professor
where position = (
select position from professor where name = '������'
);

select name, deptno, grade, height from student
where grade = 1 and height > (
select avg(height) from student
);

select name, dname from student s
join department d on s.deptno = d.deptno
where s.deptno = (select deptno from student where name = '�̱���');

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
    where deptno = (select deptno from student where name = '�̱���')
);    

select name, grade, height from student
where grade = (select grade from student where studno = '20101')
and height >(select height from student where studno = '20101');

select s.studno, d.dname, s.grade, s.name from student s
join department d on s.deptno = d.deptno 
where s.deptno in (select deptno from department where dname like '%����%');
