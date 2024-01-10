import os
import json
import time
import redis
import hashlib
from ecdsa import SigningKey, SECP256k1
from coincurve import PrivateKey
from labler import Labler


class Worker:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    RELAY_URL = os.getenv("RELAY_URL")
    SK_HEX = os.getenv("SECRET_KEY")
    SK = SigningKey.from_string(bytes.fromhex(SK_HEX), curve=SECP256k1)
    NOSTR_PUBKEY = SK.verifying_key.to_string("compressed").hex()[2:]
    LIMIT = int(os.getenv("PAST_EVENTS_LIMIT", 1000))

    def __init__(self):
        self.redis = redis.from_url(self.REDIS_URL)
        self.labler = Labler()

    async def on_message(self, ws, message):
        nostr_event = json.loads(message)
        command = nostr_event[0]

        if command == "EOSE":
            subid = nostr_event[1]
            if subid != "FUTURE":
                await self.next_past_page(subid, self.seen_ts(), ws)
        else:
            await self.handle_event(command, nostr_event, ws)

    async def on_connect(self, ws):
        future_req = [
            "REQ",
            "FUTURE",
            {"kinds": [1], "since": int(time.time())}
        ]
        await ws.send(json.dumps(future_req))
        last_seen_ts = self.seen_ts()
        past_req = [
            "REQ",
            f"PAST_{last_seen_ts}",
            {"kinds": [1], "limit": self.LIMIT, "until": last_seen_ts}
        ]
        await ws.send(json.dumps(past_req))

    async def handle_event(self, command, nostr_event, ws):
        if command == "EVENT":
            evt = nostr_event[-1]
            if not self.seen(evt["id"]):
                await self.handle_new_event(evt, ws)
            self.update_last_seen_at(event["created_at"], self.seen_ts())
        else:
            print(json.dumps(nostr_event))

    async def handle_new_event(self, event, ws):
        self.record_seen(event["id"])
        tags = self.labler.get_tags(event)
        if len(tags) > 0:
            label_event = self.build_signed_event(event, tags)
            await ws.send(json.dumps(["EVENT", label_event]))
            # self.update_last_seen_at(event["created_at"], self.seen_ts())

    def seen(self, id):
        return self.redis.sismember("labler.seen_ids", id)

    def record_seen(self, id):
        return self.redis.sadd("labler.seen_ids", id)

    def update_last_seen_at(self, new_ts, old_ts):
        self.redis.set("labler.last_seen_ts", min(
            int(new_ts), int(old_ts)))

    async def next_past_page(self, subid, ts, ws):
        await ws.send(json.dumps(["CLOSE", subid]))
        past_req = [
            "REQ",
            f"PAST_{ts}",
            {"kinds": [1], "limit": self.LIMIT, "until": ts}
        ]
        await ws.send(json.dumps(past_req))
        return True

    def seen_ts(self):
        return int(self.redis.get("labler.last_seen_ts") or int(time.time()))

    def build_signed_event(self, event, tags):
        payload = {
            "pubkey": self.NOSTR_PUBKEY,
            "created_at": event["created_at"],
            "kind": 1985,
            "content": "",
            "tags": tags + [
                ["e", event["id"], self.RELAY_URL],
                ["p", event["pubkey"], self.RELAY_URL]
            ]
        }

        serialized = [
            0,
            payload["pubkey"],
            payload["created_at"],
            payload["kind"],
            payload["tags"],
            payload["content"]
        ]

        t = json.dumps(serialized, separators=(',', ': ')).encode('utf-8')

        sha256 = hashlib.sha256(t).hexdigest()

        sk = PrivateKey(bytes.fromhex(self.SK_HEX))

        sig = sk.sign_schnorr(bytes.fromhex(sha256), b'').hex()

        signed_event = {
            "id": sha256,
            "sig": sig,
            **payload
        }

        return signed_event
