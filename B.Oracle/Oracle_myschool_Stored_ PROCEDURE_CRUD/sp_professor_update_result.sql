var v_result number;
var v_rowcount number;

execute SP_PROFESSOR_UPDATE(9908, '������', 'skylove841', '�̻�', 300, '2017-08-17', 100,  201, :v_result, :v_rowcount);

print v_result;
print v_rowcount;