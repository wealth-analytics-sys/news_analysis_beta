"""Portal (S7) authentication helpers — password hashing and login throttling (BUILD_NOTES F39).

users.yaml stores only salted PBKDF2-SHA256 hashes, never passwords. Pure Python (stdlib
only) so it is unit-testable without Streamlit.

Set or reset a user's password (prints the hash to paste into users.yaml):
    python -m src.auth hash
Generate a strong random password + its hash:
    python -m src.auth new
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time

ALGO = "pbkdf2_sha256"
ITERATIONS = 600_000          # OWASP 2023 guidance for PBKDF2-SHA256

MAX_FAILURES = 5              # failed attempts per account inside the window ...
WINDOW_S = 15 * 60            # ... within 15 minutes ...
LOCKOUT_S = 15 * 60           # ... lock that account for 15 minutes


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    """-> 'pbkdf2_sha256$<iterations>$<salt b64>$<hash b64>'."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    b64 = lambda b: base64.b64encode(b).decode("ascii")  # noqa: E731
    return f"{ALGO}${iterations}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str | None) -> bool:
    """Constant-time check of a password against a stored hash. A missing or malformed
    hash never verifies (and still costs one PBKDF2 run, so response time does not reveal
    whether the account exists)."""
    try:
        algo, iters, salt_b64, hash_b64 = (stored or "").split("$")
        if algo != ALGO:
            raise ValueError
        salt, expected, n = base64.b64decode(salt_b64), base64.b64decode(hash_b64), int(iters)
    except (ValueError, TypeError):
        hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), b"dummy-salt-0000", ITERATIONS)
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, n)
    return hmac.compare_digest(dk, expected)


def generate_password(nbytes: int = 12) -> str:
    """Random URL-safe password (~16 chars, ~96 bits)."""
    return secrets.token_urlsafe(nbytes)


class LoginThrottle:
    """Per-account failed-login lockout, shared by every session in the process.

    It must live at process level (the portal holds it in st.cache_resource), NOT in
    st.session_state — an attacker just opens a new browser session to reset per-session
    state. Resets when the app process restarts; acceptable for the beta.
    """

    def __init__(self, max_failures: int = MAX_FAILURES, window_s: float = WINDOW_S,
                 lockout_s: float = LOCKOUT_S, clock=time.monotonic):
        self.max_failures, self.window_s, self.lockout_s = max_failures, window_s, lockout_s
        self._clock = clock
        self._fails: dict[str, list[float]] = {}
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(account: str) -> str:
        return (account or "").strip().lower()

    def locked_for(self, account: str) -> float:
        """Seconds left on this account's lockout (0 if not locked)."""
        with self._lock:
            until = self._locked_until.get(self._key(account), 0.0)
            return max(0.0, until - self._clock())

    def record_failure(self, account: str) -> None:
        k, now = self._key(account), self._clock()
        with self._lock:
            recent = [t for t in self._fails.get(k, []) if now - t < self.window_s]
            recent.append(now)
            if len(recent) >= self.max_failures:
                self._locked_until[k] = now + self.lockout_s
                recent = []
            self._fails[k] = recent

    def record_success(self, account: str) -> None:
        with self._lock:
            self._fails.pop(self._key(account), None)


def _cli() -> None:
    import getpass
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "hash"
    if mode == "new":
        pw = generate_password()
        print(f"password:      {pw}")
    elif mode == "hash":
        pw = getpass.getpass("New password: ")
        if len(pw) < 12:
            sys.exit("Use at least 12 characters.")
        if pw != getpass.getpass("Repeat: "):
            sys.exit("Passwords do not match.")
    else:
        sys.exit("usage: python -m src.auth [hash|new]")
    print(f"password_hash: \"{hash_password(pw)}\"")


if __name__ == "__main__":
    _cli()
