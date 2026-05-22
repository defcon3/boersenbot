"""
Lokales Tool zum Vergleich mehrerer Optionsscheine auf dasselbe Underlying.
Start:  python optionen_vergleich.py
Dann:   http://localhost:5051
"""
from flask import Flask, jsonify, render_template, request

from warrant_fetch import fetch_warrant
from warrant_search import resolve_underlying, search_warrants

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


@app.route("/api/underlyings")
def api_underlyings():
    """Basiswert-Autocomplete: Name/ISIN/WKN -> [{name, id_notation}]."""
    q = request.args.get("q", "")
    try:
        return jsonify({"ok": True, "results": resolve_underlying(q)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _num_arg(name):
    v = (request.args.get(name) or "").strip().replace(",", ".")
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


@app.route("/api/search")
def api_search():
    """Optionsschein-Finder: Basiswert + Typ + Strike-Range + EUR-Marktpreis."""
    underlying = request.args.get("underlying", "").strip()
    id_notation = request.args.get("id_notation", "").strip()
    opt_type = request.args.get("type", "CALL").strip().upper()
    if opt_type not in ("CALL", "PUT", ""):
        opt_type = "CALL"
    if not underlying:
        return jsonify({"ok": False, "error": "Basiswert fehlt."}), 400
    target = ({"name": underlying, "id_notation": id_notation}
              if id_notation else underlying)
    try:
        res = search_warrants(
            target,
            opt_type=opt_type,
            strike_from=_num_arg("strike_from"),
            strike_to=_num_arg("strike_to"),
            price_min=_num_arg("price_min"),
            price_max=_num_arg("price_max"),
            limit=int(request.args.get("limit", 50)),
        )
        if res.get("error"):
            return jsonify({"ok": False, "error": res["error"]}), 404
        return jsonify({"ok": True, **res})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False)
