-- Which issuer reported the largest FY2024 revenue?
SELECT ticker,"end",value,DENSE_RANK() OVER (ORDER BY value DESC) AS rank
FROM known WHERE metric='revenue' AND kind='annual' AND "end" LIKE '2024%';
