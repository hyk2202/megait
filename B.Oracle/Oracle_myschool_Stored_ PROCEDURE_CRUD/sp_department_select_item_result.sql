var v_result number;
var v_rs refcursor;

execute SP_DEPARTMENT_SELECT_ITEM(101, :v_result, :v_rs);

print v_result;
print v_rs;