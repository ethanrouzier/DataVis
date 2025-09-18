#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, uuid, io, contextlib, importlib, types
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use("Agg")  # backend non-interactif
import matplotlib.pyplot as plt
# en haut de app.py (imports)
import logging
logging.basicConfig(level=logging.INFO)
# --- Dossiers ---
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TPL_DIR    = os.path.join(BASE_DIR, "templates")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOTS_DIR  = os.path.join(STATIC_DIR, "plots")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

ALLOWED_EXT = {".csv", ".xlsx", ".xls"}

# --- Flask ---
app = Flask(__name__, template_folder=TPL_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key")

# --- Import core autovis (à côté) ---
autovis = importlib.import_module("autovis")

# ---------- Helpers ----------


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT

def save_open_figs_and_close() -> List[str]:
    """Sauve toutes les figures ouvertes puis ferme. Retourne les URLs web."""
    urls: List[str] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        fname = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.png"
        fpath = os.path.join(PLOTS_DIR, fname)
        fig.savefig(fpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        urls.append(url_for("static", filename=f"plots/{fname}"))
    return urls

def run_autovis_and_capture(df, commands) -> Tuple[List[str], str, Optional[str]]:
    """
    Exécute autovis.compile_and_run(df, commands) en interceptant plt.show()
    pour sauver les PNG. Capture stdout pour les logs et renvoie les URLs.
    """
    real_show = plt.show
    saved_urls: List[str] = []

    def patched_show(*args, **kwargs):
        # quand autovis appelle show(), on SAUVE et on RETIENT les URLs
        saved_urls.extend(save_open_figs_and_close())

    plt.show = patched_show
    buf = io.StringIO()
    err_text: Optional[str] = None

    try:
        with contextlib.redirect_stdout(buf):
            autovis.compile_and_run(df, commands)
        # au cas où il resterait des figures ouvertes
        saved_urls.extend(save_open_figs_and_close())
    except Exception as e:
        saved_urls.extend(save_open_figs_and_close())
        err_text = f"{type(e).__name__}: {e}"
    finally:
        plt.show = real_show

    logs = buf.getvalue()
    return saved_urls, logs, err_text

def stringify_plan(commands: List[Any]) -> str:
    """Affiche la mini-langue à partir de la liste VisCommand d'autovis."""
    out = []
    for c in commands:
        params = getattr(c, "params", {}) or {}
        parts = []
        for k, v in params.items():
            if isinstance(v, list):
                parts.append(f"{k}={','.join(map(str, v))}")
            else:
                parts.append(f"{k}={v}")
        out.append(f"{getattr(c, 'kind', 'vis')}: " + "; ".join(parts))
    return "\n".join(out)


# ---- NEW: Ask LLM for 5 NL prompts tailored to the dataset ----
def mistral_suggest_nl_prompts(
    df,
    k: int = 5,
    lang: str = "fr",
    *,
    model: str = None,
    endpoint: str = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> List[str]:
    import re as _re
    model = model or getattr(autovis, "DEFAULT_MISTRAL_MODEL", "mistral-large-latest")
    endpoint = endpoint or getattr(autovis, "DEFAULT_MISTRAL_ENDPOINT", "https://api.mistral.ai/v1/chat/completions")
    key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
    if not key:
        raise RuntimeError("Missing Mistral API key")

    cols_list = [str(c) for c in df.columns]
    cols_text = "\n".join(f"- {c}" for c in cols_list)
    try: date_cands  = autovis.guess_date_columns(df)
    except Exception: date_cands = []
    try: label_cands = autovis.guess_label_columns(df)
    except Exception: label_cands = []

    # collect unique values for columns literally named category/categorie/label/labels
    def _norm(s: str) -> str:
        import unicodedata as _ud
        s = str(s).lower()
        return "".join(ch for ch in _ud.normalize("NFD", s) if _ud.category(ch) != "Mn")
    cat_vals = {}
    for c in df.columns:
        n = _norm(c)
        if n in {"category","categorie","label","labels"}:
            try:
                vals = df[c].dropna().astype(str).unique().tolist()
                cat_vals[str(c)] = vals[:50]
            except Exception: pass
    cat_text = ""
    if cat_vals:
        lines = [f"- {c}: " + ", ".join(vs[:12]) + ("…" if len(vs)>12 else "") for c,vs in cat_vals.items()]
        cat_text = "Category-like values (from 'category'/'label'):\n" + "\n".join(lines)

    sys_prompt = (
      "You are a senior data-visualization strategist. Given a dataset description, "
      "propose concise natural-language chart requests that would be useful.\n"
      f"Output EXACTLY {k} lines, no bullets/numbering. Each line (<=140 chars) must reference real column names."
      "Don't hesitate to filter by category"
      "Speak like a real human, with detailed requests"
    )
    user = (
      f"Language: {lang}\nDataset columns:\n{cols_text}\n\n"
      + (f"Date candidates:\n" + "\n".join(f"- {c}" for c in date_cands) + "\n" if date_cands else "")
      + (f"Label candidates:\n" + "\n".join(f"- {c}" for c in label_cands) + "\n" if label_cands else "")
      + (cat_text + "\n\n" if cat_text else "")
      + "Constraints & diversity: include at least one of each where possible: table preview, histogram (numeric), "
        "bar by category, timeline (if date exists), and a scatter if two numeric columns exist."
    )
    headers = {"Content-Type":"application/json", "Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":user}], "temperature": temperature}
    data = autovis._http_post_json(endpoint, headers, payload)
    content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
    lines = [ln.strip(" •-\t") for ln in content.splitlines() if ln.strip()]
    if len(lines) < k and ";" in content:
        lines = [p.strip() for p in content.split(";") if p.strip()]
    out = []
    for ln in lines:
        out.append(_re.sub(r'^\s*\d+[\).\s]+\s*', '', ln))
        if len(out) == k: break
    while len(out) < k:
        out.append(f"Aperçu (table) des colonnes principales: {', '.join(cols_list[:3])}")
    return out
# ---------- Routes ----------

@app.route("/", methods=["GET", "POST"])
def index():
    # ----- POST: traite, stocke en session, redirige -----
    if request.method == "POST":
        plots: List[str] = []
        logs: str = ""
        error: Optional[str] = None
        plan_text: str = ""
        auto_prompts: List[str] = []

        prompt = (request.form.get("prompt") or "").strip() or "table, histogram"
        sheet  = (request.form.get("sheet") or "").strip() or None

        # clé Mistral (form > env)
        mistral_key_form = (request.form.get("mistral_api_key") or "").strip()
        if mistral_key_form:
            os.environ["MISTRAL_API_KEY"] = mistral_key_form
        mistral_key = os.environ.get("MISTRAL_API_KEY")

        # Fichier
        f = request.files.get("datafile")
        if not f or f.filename == "":
            flash("Merci de sélectionner un fichier CSV/XLSX.", "error")
            return redirect(url_for("index"))
        if not allowed_file(f.filename):
            flash("Format non supporté. Utilise CSV, XLSX ou XLS.", "error")
            return redirect(url_for("index"))

        # Sauvegarde upload
        fname = secure_filename(f.filename)
        upath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{fname}")
        f.save(upath)
        # --- mémorise pour les raffinements suivants ---
        session["data_path"] = upath
        session["sheet"] = sheet
        if mistral_key_form:
            session["mistral_key"] = mistral_key_form

        try:
            # Charge dataframe via autovis
            df = autovis.load_dataframe(upath, sheet=sheet)

            # ---------- Planner : Mistral brut → parse ; sinon fallback naïf ----------
            try:
                if not mistral_key:
                    raise RuntimeError("Missing Mistral API key")

                mini_raw = autovis.mistral_minilang_from_prompt(
                    prompt, df,
                    model=getattr(autovis, "DEFAULT_MISTRAL_MODEL", "mistral-large-latest"),
                    endpoint=getattr(autovis, "DEFAULT_MISTRAL_ENDPOINT", "https://api.mistral.ai/v1/chat/completions"),
                    api_key=mistral_key,
                    temperature=0.1,
                    openai_compatible=False,
                    oa_endpoint=None,
                    oa_api_key=None,
                )
                app.logger.info("\n=== LLM RAW MINI-LANG ===\n%s\n=========================", mini_raw)
                commands = autovis.parse_plan_minilang(mini_raw)

            except Exception as e:
                app.logger.warning("[PLANNER] fallback=naive because: %s", e)
                commands = autovis.naive_planner_from_prompt(prompt, df)

            # Plan normalisé (pour affichage & log)
            plan_text = stringify_plan(commands)
            app.logger.info("\n=== MINI-LANG (normalized) ===\n%s\n==============================", plan_text)

            # Exécution et capture (sauve les PNG au passage)
            plots, logs, err_text = run_autovis_and_capture(df, commands)
            if err_text:
                error = err_text

        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        # Stocke le résultat et redirige pour éviter le "resubmit form"
        session["results"] = {
            "plots": plots,
            "logs": logs,
            "error": error,
            "plan_text": plan_text,
            "auto_prompts": [],
        }
        return redirect(url_for("index"))

    # ----- GET: lit depuis la session (si dispo) -----
    results = session.pop("results", None) or {}
    plots: List[str] = results.get("plots", [])
    logs: str = results.get("logs", "")
    error: Optional[str] = results.get("error")
    plan_text: str = results.get("plan_text", "")

    return render_template(
        "index.html",
        plots=plots,
        logs=logs,
        error=error,
        plan_text=plan_text,
        auto_prompts=(results.get("auto_prompts") or []),
    )

@app.route("/refine", methods=["POST"])
def refine():
    # 1) Récupère l’instruction utilisateur + plan précédent
    refine_prompt = (request.form.get("refine_prompt") or "").strip()
    previous_plan = (request.form.get("previous_plan") or "").strip() \
                    or (session.get("results") or {}).get("plan_text", "")

    if not refine_prompt:
        flash("Merci d'écrire une instruction pour affiner.", "error")
        return redirect(url_for("index"))

    # 2) Retrouve le dataset en session
    upath = session.get("data_path")
    sheet = session.get("sheet")
    if not upath or not os.path.exists(upath):
        flash("Aucun dataset en session. Ré-uploade un fichier pour continuer.", "error")
        return redirect(url_for("index"))

    # 3) Clé API
    mistral_key = session.get("mistral_key") or os.environ.get("MISTRAL_API_KEY") or ""

    # 4) Construit le prompt combiné (instruction + code précédent)
    combined_prompt = (
        f"{refine_prompt}\n\n"
        f"Précédent code (mini-langue) à adapter:\n{previous_plan}"
    )

    plots: List[str] = []
    logs: str = ""
    error: Optional[str] = None
    plan_text: str = ""
    auto_prompts: List[str] = []

    try:
        # 5) Recharge le DF
        df = autovis.load_dataframe(upath, sheet=sheet)

        # 6) Planner (Mistral → parse ; fallback naïf)
        try:
            if not mistral_key:
                raise RuntimeError("Missing Mistral API key")

            mini_raw = autovis.mistral_minilang_from_prompt(
                combined_prompt, df,
                model=getattr(autovis, "DEFAULT_MISTRAL_MODEL", "mistral-large-latest"),
                endpoint=getattr(autovis, "DEFAULT_MISTRAL_ENDPOINT", "https://api.mistral.ai/v1/chat/completions"),
                api_key=mistral_key,
                temperature=0.1,
                openai_compatible=False,
                oa_endpoint=None,
                oa_api_key=None,
            )
            app.logger.info("\n=== LLM RAW MINI-LANG (REFINE) ===\n%s\n=========================", mini_raw)
            commands = autovis.parse_plan_minilang(mini_raw)

        except Exception as e:
            app.logger.warning("[PLANNER/REFINE] fallback=naive because: %s", e)
            commands = autovis.naive_planner_from_prompt(combined_prompt, df)

        # 7) Plan normalisé (pour affichage & log)
        plan_text = stringify_plan(commands)
        app.logger.info("\n=== MINI-LANG PARSED (REFINE) ===\n%s\n", plan_text)

        # 8) Exécution + capture des plots
        plots, logs, err_text = run_autovis_and_capture(df, commands)
        if err_text:
            error = err_text

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    # 9) Stocke le nouveau résultat et repart sur GET /
    session["results"] = {
        "plots": plots,
        "logs": logs,
        "error": error,
        "plan_text": plan_text,
        "auto_prompts": [],
    }
    return redirect(url_for("index"))



# ---------- NEW: Auto-report route ----------
@app.route("/auto_report", methods=["POST"])
def auto_report():
    plots: List[str] = []
    logs: str = ""
    error: Optional[str] = None
    plan_text: str = ""
    auto_prompts: List[str] = []

    sheet  = (request.form.get("sheet") or "").strip() or None
    mistral_key_form = (request.form.get("mistral_api_key") or "").strip()
    if mistral_key_form:
        os.environ["MISTRAL_API_KEY"] = mistral_key_form
        session["mistral_key"] = mistral_key_form
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    lang = (request.form.get("app_lang") or "fr").strip() or "fr"

    f = request.files.get("datafile")
    if f and f.filename:
        if not allowed_file(f.filename):
            flash("Format non supporté. Utilise CSV, XLSX ou XLS.", "error")
            return redirect(url_for("index"))
        fname = secure_filename(f.filename)
        upath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{fname}")
        f.save(upath)
        session["data_path"] = upath
        session["sheet"] = sheet
    else:
        upath = session.get("data_path")
        sheet = session.get("sheet")
        if not upath or not os.path.exists(upath):
            flash("Aucun dataset. Merci d'uploader un fichier.", "error")
            return redirect(url_for("index"))

    try:
        df = autovis.load_dataframe(upath, sheet=sheet)
        if not mistral_key:
            raise RuntimeError("Missing Mistral API key")
        auto_prompts = mistral_suggest_nl_prompts(
            df, k=5, lang=lang,
            model=getattr(autovis, "DEFAULT_MISTRAL_MODEL", "mistral-large-latest"),
            endpoint=getattr(autovis, "DEFAULT_MISTRAL_ENDPOINT", "https://api.mistral.ai/v1/chat/completions"),
            api_key=mistral_key, temperature=0.2,
        )
        all_plans = []
        for idea in auto_prompts:
            mini_raw = autovis.mistral_minilang_from_prompt(
                idea, df,
                model=getattr(autovis, "DEFAULT_MISTRAL_MODEL", "mistral-large-latest"),
                endpoint=getattr(autovis, "DEFAULT_MISTRAL_ENDPOINT", "https://api.mistral.ai/v1/chat/completions"),
                api_key=mistral_key, temperature=0.1,
                openai_compatible=False, oa_endpoint=None, oa_api_key=None,
            )
            cmds = autovis.parse_plan_minilang(mini_raw)
            all_plans.append(stringify_plan(cmds))
            p_urls, run_logs, err_text = run_autovis_and_capture(df, cmds)
            plots.extend(p_urls)
            if run_logs: logs += f"\n# {idea}\n" + run_logs + "\n"
            if err_text: logs += f"[auto-report error] {err_text}\n"
        plan_text = "\n\n".join(all_plans).strip()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    session["results"] = {
        "plots": plots, "logs": logs, "error": error,
        "plan_text": plan_text, "auto_prompts": auto_prompts,
    }
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 8000)))
