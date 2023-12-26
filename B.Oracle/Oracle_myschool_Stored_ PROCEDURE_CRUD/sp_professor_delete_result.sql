var v_result number;
var v_rowcount number;

execute SP_PROFESSOR_DELETE(9908, :v_result, :v_rowcount);

print v_result;
print v_rowcount;