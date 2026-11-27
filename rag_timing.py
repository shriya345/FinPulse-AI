"""Compare warmed models with a fresh query embedding for every retrieval method."""
import json
from rag import index
from rag_docs import ROOT
from rag_retrieve import METHODS,model,query_vector

if __name__=="__main__":
    idx=index()
    model("encoder")
    model("ranker")
    idx.search("Warm up model execution","AAPL","2024-12-31",METHODS[-1])
    rows=[]
    for c in json.loads((ROOT/"eval/document-cases.json").read_text()):
        for method in METHODS:
            query_vector.cache_clear()
            _,ms=idx.search(c["question"],c["ticker"],c["cutoff"],method)
            rows.append(dict(case=c["id"],method=method,ms=ms))
    (ROOT/"eval/document-timing.json").write_text(json.dumps(rows,indent=2)+"\n")
