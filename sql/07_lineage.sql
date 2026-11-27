-- Which original sources support annual revenue?
SELECT k.ticker,k."end",k.filed,k.accession,s.url FROM known k
JOIN source_records s ON k.source=s.id WHERE k.metric='revenue' AND k.kind='annual';
