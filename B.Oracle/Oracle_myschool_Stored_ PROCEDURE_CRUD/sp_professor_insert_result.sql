var v_result number;
var v_seq number;

execute SP_PROFESSOR_INSERT('���¿�', 'xodud1202', '��ǥ�̻�', 2000, '2017-08-16', 3000, 201, :v_result, :v_seq);

print v_result;
print v_seq;