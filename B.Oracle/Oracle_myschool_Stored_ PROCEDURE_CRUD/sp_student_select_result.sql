var v_result number;
var v_rs refcursor;

execute SP_STUDENT_SELECT(:v_result, :v_rs);

print v_result;
print v_rs;