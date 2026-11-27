-- What was known by 2024 year end? Rebuild this temporary view per SQL session.
CREATE TEMP VIEW known AS
WITH ranked AS (
 SELECT f.*, ROW_NUMBER() OVER (PARTITION BY ticker,metric,start,"end" ORDER BY filed DESC, accession DESC) AS revision
 FROM financial_facts f WHERE filed <= '2024-12-31'
) SELECT * FROM ranked WHERE revision=1;
SELECT ticker, COUNT(*) AS known_facts FROM known GROUP BY ticker;
