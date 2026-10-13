"""Portal auth: password hashing + login throttle (BUILD_NOTES F39)."""
from pathlib import Path

import yaml

from src.auth import LoginThrottle, generate_password, hash_password, verify_password

FAST = 1_000  # iterations — keep the suite quick; production uses auth.ITERATIONS


def test_hash_roundtrip_and_salted():
    h1, h2 = hash_password("correct horse", FAST), hash_password("correct horse", FAST)
    assert h1 != h2                                  # random salt per hash
    assert verify_password("correct horse", h1)
    assert not verify_password("wrong horse", h1)


def test_missing_or_malformed_hash_never_verifies():
    for stored in (None, "", "12345678", "md5$1$x$y", "pbkdf2_sha256$notint$x$y"):
        assert not verify_password("12345678", stored)


def test_generated_passwords_are_strong_and_unique():
    pws = {generate_password() for _ in range(50)}
    assert len(pws) == 50 and all(len(p) >= 16 for p in pws)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_throttle_locks_after_max_failures_then_expires():
    clk = _Clock()
    th = LoginThrottle(max_failures=5, window_s=900, lockout_s=900, clock=clk)
    for _ in range(4):
        th.record_failure("RM@x.com")
    assert th.locked_for("rm@x.com") == 0            # 4 < 5, case-insensitive key
    th.record_failure("rm@x.com")
    assert th.locked_for("rm@x.com") > 0
    assert th.locked_for("other@x.com") == 0         # per-account
    clk.t += 901
    assert th.locked_for("rm@x.com") == 0


def test_throttle_window_and_success_reset():
    clk = _Clock()
    th = LoginThrottle(max_failures=3, window_s=60, lockout_s=60, clock=clk)
    th.record_failure("a"); th.record_failure("a")
    clk.t += 61                                      # old failures age out of the window
    th.record_failure("a")
    assert th.locked_for("a") == 0
    th.record_failure("a"); th.record_success("a"); th.record_failure("a")
    assert th.locked_for("a") == 0


def test_users_yaml_holds_no_plaintext_passwords():
    users = yaml.safe_load((Path(__file__).resolve().parents[1] / "users.yaml")
                           .read_text(encoding="utf-8"))["users"]
    assert users
    for u in users:
        assert "password" not in u, f"plaintext password for {u.get('email')}"
        assert str(u.get("password_hash", "")).startswith("pbkdf2_sha256$")
