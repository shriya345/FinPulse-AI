-- How did annual sales change from the previous observed fiscal year?
SELECT ticker,"end",value,LAG("end") OVER w AS previous_end,
 value-LAG(value) OVER w AS change_usd FROM known
WHERE metric='revenue' AND kind='annual'
WINDOW w AS (PARTITION BY ticker ORDER BY "end") ORDER BY ticker,"end";
