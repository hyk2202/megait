var v_result number;
var v_seq number;

execute SP_STUDENT_INSERT('юлеб©╣', 'xodud1202', 4, '8912111111111', '1989-12-11', '010-7111-1111', 176, 68, 101, 9901, :v_result, :v_seq);

print v_result;
print v_seq;