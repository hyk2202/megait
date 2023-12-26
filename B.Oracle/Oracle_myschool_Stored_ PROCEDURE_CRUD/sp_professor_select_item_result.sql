var v_result number;
var v_rs refcursor;

execute SP_PROFESSOR_SELECT_ITEM(9902, :v_result, :v_rs);

print v_result;
print v_rs;