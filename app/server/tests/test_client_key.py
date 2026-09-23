"""Per-client limits key IPv6 by /64."""

from __future__ import annotations

from server.api.ratelimit import client_key


def test_ipv4_is_its_own_key():
    assert client_key("203.0.113.7") == "203.0.113.7"


def test_ipv6_addresses_in_one_slash_64_share_a_key():
    a = client_key("2001:db8:1:2:aaaa::1")
    b = client_key("2001:db8:1:2:ffff:ffff:ffff:ffff")
    assert a == b == "2001:db8:1:2::/64"
    assert client_key("2001:db8:1:3::1") != a


def test_ipv4_mapped_ipv6_counts_as_ipv4():
    assert client_key("::ffff:203.0.113.7") == "203.0.113.7"


def test_non_ip_hosts_pass_through():
    assert client_key("testclient") == "testclient"
