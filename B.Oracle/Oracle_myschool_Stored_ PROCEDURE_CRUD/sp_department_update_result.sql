var v_result number;
var v_rowcount number;

execute SP_DEPARTMENT_UPDATE(300, '������ΰ�', '�츮��', :v_result, :v_rowcount);

print v_result;
print v_rowcount;