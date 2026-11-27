-- How many distinct years are represented per issuer and measure?
SELECT ticker,metric,COUNT(DISTINCT "end") AS years FROM known
WHERE kind='annual' GROUP BY ticker,metric ORDER BY ticker,metric;
