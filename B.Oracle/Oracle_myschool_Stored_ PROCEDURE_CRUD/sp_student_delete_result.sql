var v_result number;
var v_rowcount number;

execute SP_STUDENT_DELETE(20105, :v_result, :v_rowcount);

print v_result;
print v_rowcount;