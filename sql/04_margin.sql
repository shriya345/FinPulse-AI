-- What fraction of annual sales became earnings?
SELECT r.ticker,r."end",100.0*n.value/NULLIF(r.value,0) AS net_margin_pct
FROM known r JOIN known n ON (r.ticker,r.start,r."end")=(n.ticker,n.start,n."end")
WHERE r.metric='revenue' AND n.metric='net_income' AND r.kind='annual';
