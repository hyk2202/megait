var v_result number;
var v_seq number;

execute SP_DEPARTMENT_INSERT('��������а�', '6ȣ��', :v_result, :v_seq);

print v_result;
print v_seq;