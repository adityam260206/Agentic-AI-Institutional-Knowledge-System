#!/bin/sh
# ── EduMind AI — container entrypoint ─────────────────────────────────────────
# Creates the state directory, seeds demo users once, waits for Ollama, then
# starts the API. Uses Python for HTTP checks (the slim image has no curl).

set -e
export PYTHONPATH="/app:${PYTHONPATH}"

mkdir -p "$(dirname "${LEDGER_DB_PATH:-/app/ingestion_ledger.db}")"

# seed_users.py drops all tables, so only seed when there are no users yet;
# restarts keep accounts and chat history.
python - <<'EOF'
from backend.database import Base, SessionLocal, User, engine
Base.metadata.create_all(bind=engine)
db = SessionLocal()
try:
    if db.query(User).first():
        print("[STARTUP] Users already present; not seeding.")
    else:
        from backend.auth import hash_password
        from seed_users import SEED_USERS
        for e in SEED_USERS:
            db.add(User(username=e["username"], hashed_password=hash_password(e["password"]), role=e["role"]))
        db.commit()
        print("[STARTUP] Seeded demo users.")
finally:
    db.close()
EOF

OLLAMA="${OLLAMA_BASE_URL:-http://localhost:11434}"
echo "[STARTUP] Waiting for Ollama at $OLLAMA ..."
python - "$OLLAMA" <<'EOF'
import sys, time, httpx
url = sys.argv[1]
for attempt in range(1, 31):
    try:
        httpx.get(url, timeout=3)
        print("[STARTUP] Ollama is ready.")
        sys.exit(0)
    except httpx.HTTPError:
        print(f"[STARTUP]   attempt {attempt}/30...")
        time.sleep(3)
print("[STARTUP] Ollama not reachable; starting anyway (chat will fail until it is).")
EOF

echo "[STARTUP] Starting FastAPI on :8000"
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
