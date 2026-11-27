-- How much operating cash remained after annual capital spending?
SELECT o.ticker,o."end",o.value-c.value AS free_cash_flow_usd
FROM known o JOIN known c ON (o.ticker,o.start,o."end")=(c.ticker,c.start,c."end")
WHERE o.metric='operating_cash' AND c.metric='capex' AND o.kind='annual';
