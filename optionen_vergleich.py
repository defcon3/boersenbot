"""
Lokales Tool zum Vergleich mehrerer Optionsscheine auf dasselbe Underlying.
Start:  python optionen_vergleich.py
Dann:   http://localhost:5051
"""
from flask import Flask, jsonify, render_template, request

from warrant_fetch import fetch_warrant

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("optionen_vergleich.html")


@app.route("/api/warrant/<wkn>")
def api_warrant(wkn):
    force = request.args.get("refresh") in ("1", "true", "yes")
    try:
        data = fetch_warrant(wkn, force_refresh=force)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False)
