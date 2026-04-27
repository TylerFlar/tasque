"""``python -m tasque.proxy`` — start the proxy on the default host:port."""

from __future__ import annotations

from tasque.proxy.server import DEFAULT_HOST, DEFAULT_PORT, serve


def main() -> None:
    serve(host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
