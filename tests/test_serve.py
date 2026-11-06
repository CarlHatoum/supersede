from fastapi.testclient import TestClient

from supersede.corpus import Doc
from supersede.lineage import LineageGraph
from supersede.serve.app import Service, create_app

DOCS = [
    Doc("RFC1", "Old protocol", year=1990, obsoleted_by=("RFC2",)),
    Doc("RFC2", "New protocol", year=2000, obsoletes=("RFC1",)),
    Doc("RFC3", "Something else", year=2001),
]


class FixedRetriever:
    def search(self, query, k=10):
        return [("RFC1", 3.0), ("RFC3", 2.0), ("RFC2", 1.0)][:k]


async def fake_model(old, new):
    return {"label": "supersedes", "confidence": 0.7, "rationale": "same scope"}


def client():
    declared = LineageGraph(DOCS)
    service = Service(
        docs={d.doc_id: d for d in DOCS},
        retriever=FixedRetriever(),
        inferred=LineageGraph.from_edges(DOCS, [("RFC1", "RFC2")]),
        declared=declared,
        pair_proba={("RFC1", "RFC2"): 0.9, ("RFC3", "RFC2"): 0.99},
        threshold=0.94,
        assessor=fake_model,
    )
    return TestClient(create_app(service))


def test_endpoints():
    with client() as c:
        assert c.get("/healthz").json()["docs"] == 3

        r = c.get("/search", params={"q": "protocol"}).json()
        ids = [x["doc_id"] for x in r["results"]]
        assert ids[:2] == ["RFC2", "RFC1"]  # current version promoted above the old one
        assert r["results"][1]["superseded"] and r["results"][1]["current_versions"] == ["RFC2"]
        none = c.get("/search", params={"q": "protocol", "lineage": "none"}).json()
        assert none["results"][0]["doc_id"] == "RFC1"
        assert c.get("/search", params={"q": "x", "lineage": "bad"}).status_code == 400

        lin = c.get("/lineage/RFC1").json()
        assert lin["inferred"]["replaced_by"] == ["RFC2"]
        assert c.get("/lineage/RFC9").status_code == 404

        # in the band: the model decides
        a = c.post("/assess", json={"old": "RFC1", "new": "RFC2"}).json()
        assert a["decided_by"] == "model" and a["label"] == "supersedes"
        # confident: the cheap scorer decides
        b = c.post("/assess", json={"old": "RFC3", "new": "RFC2"}).json()
        assert b == {"label": "supersedes", "decided_by": "cheap_scorer", "score": 0.99}
        n = c.post("/assess", json={"old": "RFC2", "new": "RFC3"}).json()
        assert n["label"] == "unrelated" and n["decided_by"] == "cheap_scorer"


def test_a2a():
    with client() as c:
        card = c.get("/.well-known/agent-card.json").json()
        assert card["skills"][0]["id"] == "assess_supersession"
        assert card["url"].endswith("/a2a")
        msg = {
            "role": "user",
            "messageId": "m1",
            "parts": [{"kind": "data", "data": {"old": "RFC3", "new": "RFC2"}}],
        }
        r = c.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": msg}},
        ).json()
        assert r["result"]["parts"][0]["data"]["label"] == "supersedes"
        bad = c.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/get"}).json()
        assert bad["error"]["code"] == -32601
