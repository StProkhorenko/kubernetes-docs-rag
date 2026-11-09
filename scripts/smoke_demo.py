"""Проверяет HTTP-страницу и поиск. Подходит для Docker smoke без ключей."""

import argparse
import json
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    client = build_opener(ProxyHandler({}))
    with client.open(args.base_url + "/", timeout=10) as response:
        assert "Kube Triage" in response.read().decode()
    with client.open(args.base_url + "/health", timeout=10) as response:
        health = json.load(response)
        assert health["index_ready"] and not health["llm_configured"]
    query = json.dumps({"question": "Что означает ImagePullBackOff?", "mode": "bm25"}).encode()
    request = Request(args.base_url + "/search", data=query, headers={"Content-Type": "application/json"})
    with client.open(request, timeout=10) as response:
        result = json.load(response)["results"][0]
        assert result["section_id"] == "images:89"
        assert result["source_url"].endswith("#L91-L95")
    print("Kube Triage: page, health, BM25 and source link verified.")


if __name__ == "__main__":
    main()
