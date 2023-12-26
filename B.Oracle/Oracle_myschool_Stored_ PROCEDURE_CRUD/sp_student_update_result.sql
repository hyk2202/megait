var v_result number;
var v_rowcount number;

execute SP_STUDENT_UPDATE(20105, '¼Û¾ÆÁö', 'sksksk1001', 3, '1234561234567', '2017-11-11', '031-111-1111', 178, 77, 201, 9902, :v_result, :v_rowcount);

print v_result;
print v_rowcount;