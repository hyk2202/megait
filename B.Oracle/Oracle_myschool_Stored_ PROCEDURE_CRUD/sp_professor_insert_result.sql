var v_result number;
var v_seq number;

execute SP_PROFESSOR_INSERT('이태영', 'xodud1202', '대표이사', 2000, '2017-08-16', 3000, 201, :v_result, :v_seq);

print v_result;
print v_seq;