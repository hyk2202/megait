var v_result number;
var v_rs refcursor;

execute SP_STUDENT_SELECT_ITEM(20104, :v_result, :v_rs);

print v_result;
print v_rs;