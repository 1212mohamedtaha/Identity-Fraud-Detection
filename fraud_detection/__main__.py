"""``python -m fraud_detection`` starts the web app."""
import argparse

from .web import create_app


def main():
    parser = argparse.ArgumentParser(description="Identity fraud detection chat app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="enable the Flask debugger and reloader")
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
