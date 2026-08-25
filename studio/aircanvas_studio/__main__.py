from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aircanvas-studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run("aircanvas_studio.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
