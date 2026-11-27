-- What are the latest available filing and reporting dates for each issuer?
SELECT c.name,MAX(k.filed) AS latest_filing,MAX(k."end") AS latest_period
FROM companies c LEFT JOIN known k ON k.ticker=c.ticker GROUP BY c.ticker,c.name;
