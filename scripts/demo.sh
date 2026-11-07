#!/usr/bin/env bash
# Live demo for the talk: lineage-aware search, the cascade, MCP and A2A.
# Press Enter to advance. Starts the API on $PORT if it isn't running.
#
#   scripts/demo.sh                 # local API on :8080 (Ollama needed for step 4b)
#   API=https://... scripts/demo.sh # a deployed API
set -u
PORT="${PORT:-8080}"
API="${API:-http://localhost:$PORT}"
PY="${PY:-.venv/bin/python}"
bold=$(tput bold 2>/dev/null || true); dim=$(tput dim 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)

step() { printf '\n%s== %s ==%s\n' "$bold" "$1" "$reset"; }
show() { printf '%s$ %s%s\n' "$dim" "$1" "$reset"; }
pause() { read -r -p "${dim}[enter]${reset} " _; }
results() {  # search results as "rank  RFC  year  title  [superseded -> heads]"
  jq -r '.results | to_entries[] | "\(.key + 1). \(.value.doc_id)  \(.value.year)  \(.value.title[0:48])"
    + (if .value.superseded then "   [obsolete -> \(.value.current_versions | join(", "))]" else "" end)'
}

if ! curl -sf "$API/healthz" >/dev/null; then
  echo "starting the API on :$PORT (about 20 s)..."
  "$PY" -m uvicorn supersede.serve.app:app --port "$PORT" --log-level warning \
    >/tmp/supersede-demo-api.log 2>&1 &
  until curl -sf "$API/healthz" >/dev/null; do sleep 1; done
fi
curl -s "$API/healthz" | jq -c .

step "1. Plain retrieval: which TCP congestion control document?"
show "GET /search?q=TCP Congestion Control&lineage=none"
curl -s "$API/search?q=TCP%20Congestion%20Control&k=4&lineage=none" | results
pause

step "2. Same query, with the lineage inferred from the documents (no metadata)"
show "GET /search?q=TCP Congestion Control"
curl -s "$API/search?q=TCP%20Congestion%20Control&k=4" | results
pause

step "3. What replaced RFC 2581? (inferred vs declared in the index)"
show "GET /lineage/RFC2581"
curl -s "$API/lineage/RFC2581" | jq -c '{inferred: .inferred.replaced_by, declared: .declared.replaced_by}'
pause

step "4a. A clear pair: the cheap scorer decides in under a millisecond"
show 'POST /assess {"old": "RFC2581", "new": "RFC5681"}'
curl -s -X POST "$API/assess" -H 'content-type: application/json' \
  -d '{"old":"RFC2581","new":"RFC5681"}' | jq -c .
pause

step "4b. An ambiguous pair (TLS 1.0 -> 1.1): the model reads both documents"
show 'POST /assess {"old": "RFC2246", "new": "RFC4346"}'
time curl -s -X POST "$API/assess" -H 'content-type: application/json' \
  -d '{"old":"RFC2246","new":"RFC4346"}' | jq '{label, confidence, rationale, decided_by, score}'
pause

step "5. The same tools over MCP (from Claude Code or any MCP client)"
show "claude mcp add --transport http supersede $API/mcp/"
"$PY" - "$API" <<'EOF'
import asyncio, json, sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main(api):
    async with streamablehttp_client(f"{api}/mcp/") as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", ", ".join(t.name for t in tools.tools))
            out = await s.call_tool("find_later", {"doc_id": "RFC2581", "k": 3})
            print("find_later(RFC2581):")
            for c in out.content[:3]:
                d = json.loads(c.text)
                print(f"  {d['doc_id']}  {d['year']}  {d['title']}")

asyncio.run(main(sys.argv[1]))
EOF
pause

step "6. And to other agents over A2A"
show "GET /.well-known/agent-card.json"
curl -s "$API/.well-known/agent-card.json" | jq -c '{name, url, skills: [.skills[].id]}'
show 'POST /a2a  message/send {"old": "RFC2581", "new": "RFC5681"}'
curl -s -X POST "$API/a2a" -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":1,
  "method":"message/send","params":{"message":{"role":"user","messageId":"demo",
  "parts":[{"kind":"data","data":{"old":"RFC2581","new":"RFC5681"}}]}}}' \
  | jq -c '.result.parts[0].data'
echo
