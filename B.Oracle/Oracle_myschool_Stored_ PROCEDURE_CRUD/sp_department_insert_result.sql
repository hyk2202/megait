var v_result number;
var v_seq number;

execute SP_DEPARTMENT_INSERT('정보통신학과', '6호관', :v_result, :v_seq);

print v_result;
print v_seq;