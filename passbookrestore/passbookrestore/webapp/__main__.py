import argparse
import webbrowser
from threading import Timer

from .app import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the PassbookRestore AI web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    app = create_app()
    url = f"http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}"
    print(f"\n  PassbookRestore AI is running at {url}\n  Press Ctrl+C to stop.\n")
    if not a.no_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=a.host, port=a.port, debug=a.debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
