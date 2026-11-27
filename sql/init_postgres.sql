
CREATE TABLE companies (
	ticker VARCHAR NOT NULL, 
	name VARCHAR, 
	cik VARCHAR, 
	PRIMARY KEY (ticker)
)

;

CREATE TABLE evaluation_results (
	id VARCHAR NOT NULL, 
	result TEXT, 
	PRIMARY KEY (id)
)

;

CREATE TABLE ingestion_runs (
	id VARCHAR NOT NULL, 
	ticker VARCHAR, 
	timestamp VARCHAR, 
	status VARCHAR, 
	rows_seen INTEGER, 
	rejected INTEGER, 
	error VARCHAR, 
	PRIMARY KEY (id)
)

;

CREATE TABLE research_sessions (
	id VARCHAR NOT NULL, 
	ticker VARCHAR, 
	result TEXT, 
	PRIMARY KEY (id)
)

;

CREATE TABLE source_records (
	id VARCHAR NOT NULL, 
	url VARCHAR, 
	retrieved VARCHAR, 
	capture VARCHAR, 
	raw TEXT, 
	PRIMARY KEY (id)
)

;

CREATE TABLE financial_facts (
	id VARCHAR NOT NULL, 
	ticker VARCHAR, 
	source VARCHAR, 
	metric VARCHAR, 
	start VARCHAR, 
	"end" VARCHAR, 
	filed VARCHAR, 
	kind VARCHAR, 
	unit VARCHAR, 
	value BIGINT, 
	accession VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(ticker) REFERENCES companies (ticker), 
	FOREIGN KEY(source) REFERENCES source_records (id)
)

;
CREATE INDEX ix_fact_lookup ON financial_facts (ticker, metric, "end", filed);