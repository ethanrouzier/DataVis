#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
autovis.py — Auto data-vis pipeline (planner→compiler→tools) avec retries LLM

What it does
------------
1) Tu donnes un dataset (CSV/Excel) + une requête utilisateur (NLP) ou un plan (mini-langue/JSON).
2) Le planner (Mistral ou naïf) produit un plan en mini-langue.
3) Le compilateur résout les colonnes (robuste) et exécute les outils matplotlib (plt.show()).
4) Si ça casse, on loggue l’erreur + contexte et on demande au LLM de **replan** (jusqu’à --max-retries).

Exemples
--------
python autovis.py data.csv --planner mistral --mistral-api-key sk-... \
  --prompt "Timeline des rendez-vous médicaux par type et un histogramme des ingrédients"

python autovis.py export.xlsx --sheet export --plan "
table: limit=20
timeline: date=date_doc; label=type_doc_clinique
histogram: x=ingredients_liste; bins=20
"
"""

from __future__ import annotations

import os
import re
import json
import argparse
import unicodedata
import difflib
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Callable, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from datetime import datetime

# =================== Utilities: cleaning, matching, filtering ===================

# ====== Figure & font scaling + thème beige-green ======
import matplotlib as mpl
from cycler import cycler

# Multiplie la taille des figures et polices (configurable via env)
FIG_SCALE  = float(os.getenv("AUTOVIS_FIG_SCALE",  "1.0"))  # 2.0 => ~2x plus grand
FONT_SCALE = float(os.getenv("AUTOVIS_FONT_SCALE", "1.4"))  # 1.4 => polices plus lisibles

def set_beige_green_theme():
    base_font = 11 * FONT_SCALE
    mpl.rcParams.update({
        "figure.facecolor": "#FFFAF1",
        "axes.facecolor":   "#FFFFFF",
        "axes.edgecolor":   "#E4D6C5",
        "axes.labelcolor":  "#1f2328",
        "axes.grid": True,
        "grid.color": "#E4D6C5",
        "grid.linestyle": "-",
        "grid.alpha": 0.8,
        "xtick.color": "#6f6a60",
        "ytick.color": "#6f6a60",
        "font.size": base_font,
        "axes.titlesize": base_font * 1.2,
        "axes.titleweight": "bold",
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font * 0.95,
        "lines.linewidth": 2.0,
        "axes.prop_cycle": cycler(color=["#2FAF79", "#6AD3A6", "#1f2328", "#B13434", "#8C6B43"]),
        "legend.frameon": False,
    })

def beautify(ax, title: Optional[str] = None):
    for sp in ("top","right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_alpha(0.5)
    ax.spines["bottom"].set_alpha(0.5)
    ax.grid(axis="y", alpha=0.55)
    if title: ax.set_title(title)
    ax.figure.tight_layout()

def new_fig(w=6, h=4):
    """Crée une figure en tenant compte de FIG_SCALE."""
    return plt.subplots(figsize=(w * FIG_SCALE, h * FIG_SCALE))

set_beige_green_theme()


# --- Visual helpers ---------------------------------------------------------

_COLOR_SYNONYMS = {
    # français
    "bleu":"blue","rouge":"red","vert":"green","jaune":"gold","orange":"orange",
    "violet":"purple","rose":"pink","noir":"black","blanc":"white","gris":"gray",
    "turquoise":"turquoise","marron":"saddlebrown","beige":"#f5efe6",
    # anglais (au cas où)
    "blue":"blue","red":"red","green":"green","gold":"gold","orange":"orange",
    "purple":"purple","pink":"pink","black":"black","white":"white","gray":"gray",
    "grey":"gray","turquoise":"turquoise","brown":"saddlebrown","beige":"#f5efe6",
    "teal":"teal","cyan":"cyan","magenta":"magenta",
    "transparent":"none",
}

def _normalize_color_token(tok: str) -> str:
    t = _norm_text(tok)  # déjà dispo dans ton code, retire accents/casse
    return _COLOR_SYNONYMS.get(t, tok)  # fallback: laisse passer (#hex, nom mpl)

def _extract_visuals(kwargs: dict) -> dict:
    """Prend et retire les clés visuelles du dict paramètres."""
    vis_keys = {"bg","color","grid"}
    vis = {}
    for k in list(kwargs.keys()):
        if k in vis_keys:
            vis[k] = kwargs.pop(k)
    return vis

def _apply_visuals(fig, ax, vis: dict):
    bg   = vis.get("bg")
    grid = vis.get("grid")
    if isinstance(bg, str):
        c = _normalize_color_token(bg)
        try:
            fig.patch.set_facecolor(c)
            ax.set_facecolor(c)
        except Exception:
            pass
    if isinstance(grid, (str, bool)):
        on = (str(grid).strip().lower() in {"1","true","yes","y","on","oui"})
        ax.grid(on)

# ---------- LLM context helpers ----------

# --- Embeddings + PCA (fallback texte -> numérique) ---
import os
import numpy as np

_EMBED_MODEL = None

def _get_embedder():
    """
    Charge sentence-transformers à la demande.
    Modèle par défaut : paraphrase-multilingual-MiniLM-L12-v2 (384d).
    Désactive via AUTOVIS_EMBED_FALLBACK=0
    """
    if os.getenv("AUTOVIS_EMBED_FALLBACK", "1") in {"0","false","False"}:
        raise RuntimeError("Embedding fallback disabled by AUTOVIS_EMBED_FALLBACK=0")
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers n'est pas installé. "
                "Installe: pip install sentence-transformers"
            ) from e
        name = os.getenv("AUTOVIS_EMBED_MODEL",
                         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        device = os.getenv("AUTOVIS_EMBED_DEVICE", "cpu")
        _EMBED_MODEL = SentenceTransformer(name, device=device)
    return _EMBED_MODEL

def _embed_series_text(s: pd.Series) -> np.ndarray:
    """Encode une série texte -> matrice (n, d). Normalise les embeddings."""
    s = s.fillna("").astype(str)
    model = _get_embedder()
    vecs = model.encode(s.tolist(), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs)

def _pca_np(X: np.ndarray, k: int) -> np.ndarray:
    """
    PCA maison via SVD. Retourne (n, k).
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD compacte
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T  # projette sur les k 1ers axes


def clean_column_name(raw) -> str:
    """Unwrap headers like "{'name':'col'}" -> "col" """
    s = str(raw)
    m = re.match(r"""\{\s*['"]name['"]\s*:\s*['"]([^'"]+)['"]\s*\}""", s)
    if m:
        return m.group(1)
    return s

def _norm(s: str) -> str:
    """Normalize for tolerant matching (case/accents/spaces/punct insensitive)."""
    s = str(s)
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def resolve_column_name(df: pd.DataFrame, col_like: Optional[str]) -> Optional[str]:
    """Return best matching column name for col_like (exact, normalized, fuzzy)."""
    if not col_like:
        return None
    if col_like in df.columns:
        return col_like
    norm_target = _norm(col_like)
    norm_map = {c: _norm(c) for c in df.columns}
    exact_norm = [c for c, n in norm_map.items() if n == norm_target]
    if exact_norm:
        return exact_norm[0]
    close = difflib.get_close_matches(norm_target, list(norm_map.values()), n=1, cutoff=0.72)
    if close:
        for c, n in norm_map.items():
            if n == close[0]:
                return c
    return None

def _pick_one_if_duplicate(df: pd.DataFrame, name: str) -> pd.Series:
    """
    If df[name] returns a DataFrame (duplicate headers), pick the most complete column by POSITION.
    """
    obj = df[name]
    if isinstance(obj, pd.DataFrame):
        nn = obj.notna().sum()
        pos = int(np.argmax(nn.to_numpy()))
        return obj.iloc[:, pos]
    return obj

def resolve_series(df: pd.DataFrame, col_like: str) -> pd.Series:
    """Resolve 'col_like' to a single Series (handles duplicates)."""
    real = resolve_column_name(df, col_like)
    if real is None:
        raise KeyError(f"Column not found: {col_like}")
    return _pick_one_if_duplicate(df, real)

def resolve_cols_names(df: pd.DataFrame, names: List[str]) -> List[str]:
    out = []
    for n in names:
        c = resolve_column_name(df, n)
        if c is not None and c not in out:
            out.append(c)
    return out

# --- Robust filtering: query()+fallbacks (contains/in/num/date + fuzzy columns) ---
# --- Robust filtering: query()+fallbacks, avec matching catégoriel approximatif ---

_ID = r"[A-Za-z_]\w*"

def _protect_strings(expr: str):
    toks = []
    def sub(m):
        toks.append(m.group(0))
        return f"__STR{len(toks)-1}__"
    tmp = re.sub(r"""(['"])(?:\\.|(?!\1).)*\1""", sub, expr)
    return tmp, toks

def _restore_strings(expr: str, toks: list[str]):
    return re.sub(r"__STR(\d+)__", lambda m: toks[int(m.group(1))], expr)

def _map_columns_in_expr(expr: str, df: pd.DataFrame) -> str:
    tmp, toks = _protect_strings(expr)
    def repl_id(m):
        w = m.group(0)
        if w in {"and","or","not","in","True","False","None","between"}:
            return w
        c = resolve_column_name(df, w)
        return f'`{c}`' if c else w
    tmp = re.sub(rf"\b{_ID}\b", repl_id, tmp)
    return _restore_strings(tmp, toks)

_SPLIT_BOOL = re.compile(r'\b(and|or)\b(?=(?:[^"\']|"[^"]*"|\'[^\']*\')*$)', re.IGNORECASE)

# Protège le 'and' à l'intérieur d'un 'between ... and ...'
_BETWEEN_PROTECT_RE = re.compile(
    rf'\b({_ID})\s+between\s+((?:"[^"]*"|\'[^\']*\'))\s+and\s+((?:"[^"]*"|\'[^\']*\'))',
    re.IGNORECASE
)

def _protect_between(expr: str) -> str:
    # Remplace le 'and' interne par un placeholder pour éviter le split
    return _BETWEEN_PROTECT_RE.sub(
        lambda m: f"{m.group(1)} between {m.group(2)} __AND__ {m.group(3)}",
        expr
    )

def _is_categorical_like(s: pd.Series) -> bool:
    try:
        nun = s.nunique(dropna=True)
    except Exception:
        nun = 999999
    return s.dtype == object or pd.api.types.is_categorical_dtype(s) or nun <= 200

def _norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s)

def _approx_match_values(candidates: List[str], query: str) -> List[str]:
    """Renvoie les valeurs candidates qui matchent approx. la requête (égalité normalisée, sous-chaîne, fuzzy)."""
    qn = _norm_text(query)
    if not qn:
        return []
    # exact normalisé
    exact = [v for v in candidates if _norm_text(v) == qn]
    if exact:
        return exact
    # sous-chaîne normalisée
    sub = [v for v in candidates if qn in _norm_text(v)]
    if sub:
        return sub
    # fuzzy
    norms = [_norm_text(v) for v in candidates]
    close = difflib.get_close_matches(qn, norms, n=3, cutoff=0.72)
    if close:
        out = []
        for c in close:
            for v in candidates:
                if _norm_text(v) == c:
                    out.append(v)
        return out
    return []

def _make_condition(df: pd.DataFrame, part: str) -> pd.Series:
    part = part.strip()

    # .str.contains("...") (+case facultatif)
    m = re.fullmatch(rf'\s*(not\s+)?({_ID})\.str\.contains\((["\'])(.+?)\3(?:,\s*case\s*=\s*(True|False))?\)\s*', part, re.I)
    if m:
        neg, col, _, pat, case = m.groups()
        colname = resolve_column_name(df, col) or col
        ser = df[colname].astype(str)
        mask = ser.str.contains(pat, na=False, case=(None if case is None else (case == "True")))
        return ~mask if neg else mask

    # startswith/endswith
    m = re.fullmatch(rf'\s*(not\s+)?({_ID})\.str\.(starts|ends)with\((["\'])(.+?)\4\)\s*', part, re.I)
    if m:
        neg, col, which, _, val = m.groups()
        colname = resolve_column_name(df, col) or col
        ser = df[colname].astype(str)
        mask = ser.str.startswith(val) if which.lower()=="starts" else ser.str.endswith(val)
        mask = mask.fillna(False)
        return ~mask if neg else mask

    # col in ["a","b"]  ou col in (1,2,3)
    m = re.fullmatch(rf'\s*({_ID})\s+in\s+(\[.*\]|\(.*\))\s*', part, re.I)
    if m:
        col, seq = m.groups()
        colname = resolve_column_name(df, col) or col
        s = df[colname]
        # parse liste
        try:
            vals = json.loads(seq.replace("(", "[").replace(")", "]"))
        except Exception:
            vals = [v.strip().strip('"\'' ) for v in _SPLIT_SEMI.split(seq.strip()[1:-1]) if v.strip()]

        if _is_categorical_like(s):
            uniques = [str(v) for v in s.dropna().unique().tolist()]
            targets = set()
            for v in vals:
                matches = _approx_match_values(uniques, str(v))
                targets.update(_norm_text(mv) for mv in matches) if matches else targets.add(_norm_text(v))
            ser_norm = s.astype(str).map(_norm_text)
            return ser_norm.isin(targets)
        else:
            return s.astype(str).isin([str(v) for v in vals])

    # Comparaisons simples: == (et alias '='), !=, >=, <=, >, <
    m = re.fullmatch(rf'\s*({_ID})\s*(==|=|!=|>=|<=|>|<)\s*(.+?)\s*', part)
    if m:
        col, op, rhs = m.groups()
        if op == "=":  # alias SQL-like
            op = "=="
        colname = resolve_column_name(df, col) or col
        s = df[colname]

        # RHS string ?
        if (rhs.startswith('"') and rhs.endswith('"')) or (rhs.startswith("'") and rhs.endswith("'")):
            lit = rhs[1:-1]
            if _is_categorical_like(s):
                uniques = [str(v) for v in s.dropna().unique().tolist()]
                matches = _approx_match_values(uniques, lit)
                if matches:
                    targets = set(_norm_text(mv) for mv in matches)
                    ser_norm = s.astype(str).map(_norm_text)
                    mask = ser_norm.isin(targets)
                else:
                    mask = s.astype(str).map(_norm_text) == _norm_text(lit)
                return ~mask if op == "!=" else mask
            # sinon compare string brute
            mask = (s.astype(str) != lit) if op == "!=" else (s.astype(str) == lit)
            return mask

        # RHS numérique ?
        try:
            num = float(rhs)
            return eval(f'pd.to_numeric(s, errors="coerce") {op} num')
        except Exception:
            pass

        # RHS date ?
        try:
            dt_col = coerce_to_datetime(s)
            dt_rhs = pd.to_datetime(rhs, errors="coerce", dayfirst=True)
            return eval(f'dt_col {op} dt_rhs')
        except Exception:
            # dernier recours: string
            return eval(f's.astype(str) {op} "{rhs}"')

    # BETWEEN dates:  date between '2024-01-01' and '2024-03-01'
    m = re.fullmatch(rf'\s*({_ID})\s+between\s+(["\'])(.+?)\2\s+and\s+(["\'])(.+?)\4\s*', part, re.I)
    if m:
        col, _, a, _, b = m.groups()
        colname = resolve_column_name(df, col) or col
        s = coerce_to_datetime(df[colname])
        a = pd.to_datetime(a, errors="coerce", dayfirst=True)
        b = pd.to_datetime(b, errors="coerce", dayfirst=True)
        return (s >= a) & (s <= b)

    # rien reconnu → True (ne filtre pas ce bout)
    m = re.fullmatch(r'\s*(not\s+)?(?:"([^"]+)"|\'([^\']+)\')\s*', part)
    term = None
    neg = False
    if m:
        neg = bool(m.group(1))
        term = m.group(2) or m.group(3)
    else:
        # motNonQuoté : un token sans espaces (évite de voler les autres syntaxes)
        m2 = re.fullmatch(r'\s*(not\s+)?([^\s"\'=<>:()]+)\s*', part)
        if m2 and not re.search(r'[.](str|dt)\b', part) and " in " not in part and not re.search(r'\b(between|==|!=|>=|<=|>|<)\b', part):
            neg = bool(m2.group(1))
            term = m2.group(2)

    if term:
        # Colonnes candidates = “catégorielles-like” (texte / peu de modalités)
        cols = [c for c in df.columns if _is_categorical_like(df[c])]
        if not cols:
            # fallback: toutes les colonnes
            cols = list(df.columns)
        mask = pd.Series(False, index=df.index)
        # recherche insensible à la casse
        for c in cols:
            try:
                s = df[c].astype(str)
                mask = mask | s.str.contains(term, case=False, na=False)
            except Exception:
                # au cas où une colonne plante en astype/contains : on ignore
                pass
        return ~mask if neg else mask
    return pd.Series(True, index=df.index)

def apply_filter(df: pd.DataFrame, expr: Optional[str]) -> pd.DataFrame:
    if not expr or not str(expr).strip():
        return df

    # 1) tentative df.query() avec mapping des noms de colonnes
    try:
        mapped = _map_columns_in_expr(str(expr), df)
        return df.query(mapped, engine="python")
    except Exception:
        pass

    # 2) parseur booléen simple: split sur and/or hors guillemets
    expr2 = _protect_between(str(expr))
    parts = _SPLIT_BOOL.split(expr2)
    mask = pd.Series(True, index=df.index)
    pending = None
    for token in parts:
        t = token.replace('__AND__', 'and').strip().lower()
        if t in {"and", "or"}:
            pending = t; continue
        cond = _make_condition(df, token.replace('__AND__', 'and'))
        if pending == "and":
            mask = mask & cond
        elif pending == "or":
            mask = mask | cond
        else:
            mask = cond
        pending = None

    try:
        return df[mask]
    except Exception:
        print(f"[WARN] Filter ignored: could not parse '{expr}'")
        return df



def coerce_to_datetime(s: pd.Series) -> pd.Series:
    """Parse many date formats, including Excel serials. Accepts a DataFrame by mistake."""
    if isinstance(s, pd.DataFrame):
        nn = s.notna().sum()
        pos = int(np.argmax(nn.to_numpy()))
        s = s.iloc[:, pos]
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    # Excel serials (days since 1899-12-30)
    if pd.api.types.is_numeric_dtype(s):
        ser = pd.to_numeric(s, errors="coerce")
        dt = pd.to_datetime(ser, unit="d", origin="1899-12-30", errors="coerce")
        if dt.notna().sum():
            return dt
    # Common parse (EU style dayfirst)
    if s.dtype == object:
        st = s.astype(str)
        iso = st.str.extract(r'(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?)')[0]
        if iso.notna().sum():
            s = iso.fillna("")
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if dt.notna().sum():
        return dt
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            dt = pd.to_datetime(s, format=fmt, errors="coerce")
            if dt.notna().sum():
                return dt
        except Exception:
            pass
    return dt  # maybe all NaT

def coerce_to_numeric_like(s: pd.Series) -> pd.Series:
    """
    Convertit '28 min', '1 h 30', '1h30', '12,5', '45s' en numérique.
    - Si ça ressemble à une durée, retourne des **minutes**.
    - Sinon, extrait le premier nombre trouvé (gère la virgule décimale).
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    st = s.astype(str).str.strip().str.lower()
    st2 = st.str.replace(',', '.', regex=False)

    # '1 h 30' / '1h30' / '1 h'
    hm = st2.str.extract(r'(?P<h>\d+(?:\.\d+)?)\s*h(?:\s*(?P<m>\d+(?:\.\d+)?))?')
    hm_mask = hm['h'].notna()
    minutes_from_hm = pd.Series(np.nan, index=st2.index, dtype=float)
    if hm_mask.any():
        h = pd.to_numeric(hm.loc[hm_mask, 'h'], errors='coerce').fillna(0)
        m = pd.to_numeric(hm.loc[hm_mask, 'm'], errors='coerce').fillna(0)
        minutes_from_hm.loc[hm_mask] = h * 60 + m

    # '30 min' ou '30m'
    m_only = st2.str.extract(r'(?P<m>\d+(?:\.\d+)?)\s*m(?:in)?\b')
    m_mask = m_only['m'].notna()
    minutes_from_m = pd.Series(np.nan, index=st2.index, dtype=float)
    if m_mask.any():
        minutes_from_m.loc[m_mask] = pd.to_numeric(m_only.loc[m_mask, 'm'], errors='coerce')

    # '45s' → minutes
    s_only = st2.str.extract(r'(?P<s>\d+(?:\.\d+)?)\s*s\b')
    s_mask = s_only['s'].notna()
    minutes_from_s = pd.Series(np.nan, index=st2.index, dtype=float)
    if s_mask.any():
        minutes_from_s.loc[s_mask] = pd.to_numeric(s_only.loc[s_mask, 's'], errors='coerce') / 60.0

    time_like = minutes_from_hm.combine_first(minutes_from_m).combine_first(minutes_from_s)

    # fallback: premier nombre flottant
    generic = st2.str.extract(r'([-+]?\d+(?:\.\d+)?)')[0]
    generic = pd.to_numeric(generic, errors='coerce')

    return time_like.combine_first(generic)

def tokenize_series_if_listlike(ser: pd.Series) -> pd.Series:
    """If object-like with separators, split into tokens and explode; else return as-is."""
    if ser.dtype == object:
        sample = ser.dropna().astype(str).head(50)
        if sample.str.contains(r"[;,]").mean() > 0.1:
            out = ser.dropna().astype(str).str.split(r"[;,]")
            out = out.explode().str.strip()
            out = out[out != ""]
            return out
    return ser

# ===================== Instruction dataclass & parsers =====================

@dataclass
class VisCommand:
    kind: str
    params: Dict[str, Any]

import re

# Split on separators that are OUTSIDE quotes
_SPLIT_SEMI = re.compile(r';(?=(?:[^"\']|"[^"]*"|\'[^\']*\')*$)')
_SPLIT_COMMA = re.compile(r',(?=(?:[^"\']|"[^"]*"|\'[^\']*\')*$)')

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

# Split on separators or just whitespace between key=value pairs (quotes-safe)
_KV_PAIR = re.compile(
    r'([A-Za-z_]\w*)\s*=\s*'                      # key=
    r'(?:'                                        # value is:
    r'"(?:[^"\\]|\\.)*"'                          #   "double-quoted"
    r'|'                                          #   or
    r"'(?:[^'\\]|\\.)*'"                          #   'single-quoted'
    r'|'                                          #   or
    r'[^;\n]+'                                    #   anything up to ; or end-of-line (commas allowed)
    r')'
)

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1]
    return s

def _parse_pairs(rest: str) -> dict:
    """
    Parse 'k=v; k2=v2' OU 'k=v k2=v2' (séparé par espaces),
    en respectant les guillemets et sans casser les virgules dans les valeurs.
    Règle d'arrêt de valeur :
      - ';' (point-virgule), OU
      - espace(s) suivis d'un début d'identifiant + '=' (nouvelle paire), OU
      - fin de chaîne.
    """
    params = {}
    s = rest.strip()
    i = 0
    n = len(s)
    while i < n:
        # skip espaces/; init
        while i < n and (s[i].isspace() or s[i] == ';'):
            i += 1
        if i >= n:
            break

        # key
        m = re.match(r'[A-Za-z_]\w*', s[i:])
        if not m:
            # rien de parsable -> on s'arrête
            break
        key = m.group(0).lower()
        i += m.end()

        # '='
        while i < n and s[i].isspace():
            i += 1
        if i >= n or s[i] != '=':
            # paire invalide -> on saute au prochain séparateur
            while i < n and s[i] not in ';':
                i += 1
            if i < n and s[i] == ';':
                i += 1
            continue
        i += 1  # skip '='
        while i < n and s[i].isspace():
            i += 1

        # value (état-machine avec guillemets + profondeur de parenthèses)
        val_chars = []
        in_quote = None
        depth = 0  # ← NEW: profondeur de (), [], {}
        while i < n:
            ch = s[i]

            # ouverture/fermeture guillemets
            if in_quote:
                val_chars.append(ch)
                if ch == in_quote:
                    in_quote = None
                i += 1
                continue
            else:
                if ch in ("'", '"'):
                    in_quote = ch
                    val_chars.append(ch)
                    i += 1
                    continue

            # suivi profondeur de parenthèses (hors guillemets)
            if ch in "([{":
                depth += 1
                val_chars.append(ch)
                i += 1
                continue
            if ch in ")]}":
                depth = max(0, depth - 1)
                val_chars.append(ch)
                i += 1
                continue

            # fin de valeur si ';' HORS parenthèses
            if ch == ';' and depth == 0:
                i += 1
                break

            # fin de valeur si espace + nouvelle 'key=' HORS parenthèses
            if ch.isspace() and depth == 0:
                j = i
                while j < n and s[j].isspace():
                    j += 1
                m2 = re.match(r'[A-Za-z_]\w*\s*=', s[j:])  # prochaine paire ?
                if m2:
                    i = j
                    break
                # sinon, l'espace fait partie de la valeur
                val_chars.append(ch)
                i += 1
                continue

            # char normal
            val_chars.append(ch)
            i += 1

        v = ''.join(val_chars).strip()
        v = _strip_quotes(v)

        # post-traitement selon la clé
        if key == "columns":
            parts = [ _strip_quotes(x.strip()) for x in _SPLIT_COMMA.split(v) if x.strip() ]
            params[key] = parts
        elif key in {"bins", "limit"}:
            try:
                params[key] = int(v)
            except Exception:
                params[key] = v
        else:
            params[key] = v

    return params

_KIND_ALIASES = {
    "histogram":"histogram","hist":"histogram","histo":"histogram",
    "bar":"bar","barplot":"bar",
    "line":"line",
    "scatter":"scatter","scatterplot":"scatter",
    "timeline":"timeline","frise":"timeline",
    "table":"table","tab":"table","preview":"table",
}

def parse_plan_minilang(plan_text: str) -> List[VisCommand]:
    cmds: List[VisCommand] = []
    for raw_line in plan_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid line (missing ':'): {line}")
        kind, rest = line.split(":", 1)
        kind = _KIND_ALIASES.get(kind.strip().lower())
        if not kind:
            raise ValueError(f"Unknown visualization kind in line: {raw_line}")
        params = _parse_pairs(rest)
        cmds.append(VisCommand(kind=kind, params=params))
    return cmds


def parse_plan_json(plan_json: str) -> List[VisCommand]:
    spec = json.loads(plan_json)
    items = []
    if isinstance(spec, dict) and "visualizations" in spec:
        seq = spec["visualizations"]
    elif isinstance(spec, list):
        seq = spec
    else:
        seq = [spec]
    for it in seq:
        kind = _KIND_ALIASES.get(str(it.get("type", "")).lower(), None)
        if not kind:
            raise ValueError(f"Missing/unknown 'type' in JSON item: {it}")
        params = {k: v for k, v in it.items() if k != "type"}
        items.append(VisCommand(kind=kind, params=params))
    return items


# Fallback si les helpers n'existent pas déjà
try:
    _norm_text
except NameError:
    import unicodedata, re, difflib
    def _norm_text(s: str) -> str:
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[^a-z0-9]+", "", s)
try:
    _approx_match_values
except NameError:
    import difflib
    def _approx_match_values(candidates: list[str], query: str) -> list[str]:
        qn = _norm_text(query)
        if not qn: return []
        exact = [v for v in candidates if _norm_text(v) == qn]
        if exact: return exact
        sub = [v for v in candidates if qn in _norm_text(v)]
        if sub: return sub
        norms = [_norm_text(v) for v in candidates]
        close = difflib.get_close_matches(qn, norms, n=3, cutoff=0.72)
        out = []
        for c in close:
            for v in candidates:
                if _norm_text(v) == c:
                    out.append(v)
        return out

_CATEGORY_NAME_HINTS = [
    "category","categorie","cat","type","type_doc","type_doc_clinique","doc_type",
    "classe","class","groupe","group","domaine","domain","service","organ","theme","tag","status","etat"
]

_LABEL_NAME_HINTS = [
    "motif","raison","reason","subject","sujet","title","titre","label",
    "type_doc_clinique","type_doc","examen","acte","procedure","description","commentaire","comments","remarque"
]

_DATE_NAME_HINTS = [
    "date","dt","jour","day","time","timestamp","dat","doc_date","date_doc","date_visite","date_consultation"
]

def detect_category_columns(df: pd.DataFrame, max_unique: int = 200) -> list[str]:
    out = []
    for c in df.columns:
        try:
            nun = df[c].nunique(dropna=True)
        except Exception:
            nun = 999999
        name_hit = any(h in _norm_text(c) for h in _CATEGORY_NAME_HINTS)
        if name_hit or (df[c].dtype == object) or pd.api.types.is_categorical_dtype(df[c]) or nun <= max_unique:
            out.append(c)
    # priorité aux noms ressemblant à "category"
    out.sort(key=lambda c: (0 if any(h in _norm_text(c) for h in _CATEGORY_NAME_HINTS) else 1, c))
    return out

def top_values(df: pd.DataFrame, col: str, k: int = 50) -> list[str]:
    try:
        ser = df[col].dropna().astype(str)
        return ser.value_counts().head(k).index.tolist()
    except Exception:
        return []

def guess_date_columns(df: pd.DataFrame) -> list[str]:
    hits = []
    for c in df.columns:
        name_hit = any(h in _norm_text(c) for h in _DATE_NAME_HINTS)
        looks_date = False
        try:
            dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            looks_date = dt.notna().mean() >= 0.3
        except Exception:
            pass
        if name_hit or looks_date:
            hits.append(c)
    # priorité par signal sur le nom
    hits = list(dict.fromkeys(hits))
    hits.sort(key=lambda c: (0 if any(h in _norm_text(c) for h in _DATE_NAME_HINTS) else 1, c))
    return hits

def guess_label_columns(df: pd.DataFrame) -> list[str]:
    cand = []
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c]):
            if any(h in _norm_text(c) for h in _LABEL_NAME_HINTS):
                cand.append(c)
    # fallback: quelques colonnes texte les plus “descriptives”
    if not cand:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        cand = text_cols[:5]
    return cand

def build_tools_schema_text() -> str:
    return (
    "Tools and arguments (mini-language):\n"
    "- table: columns=a,b,c | limit=50 | sort=col|-col | filter=expr | title=... | bg=color | grid=on|off\n"
    "- histogram: x=col | bins=30 | filter=expr | title=... | color=color | bg=color | grid=on|off\n"
    "- bar: x=col [y=col agg=sum|mean|count] | filter=expr | title=... | color=color | bg=color | grid=on|off\n"
    "- line: x=col y=col [group=col] | filter=expr | title=... | color=color | bg=color | grid=on|off\n"
    "- scatter: x=col y=col [group=col] | filter=expr | title=... | color=color | bg=color | grid=on|off\n"
    "- timeline: date=col label=col [color_by=col] | filter=expr | title=... | color=color | bg=color | grid=on|off\n"
    )

def build_filters_cheatsheet_text() -> str:
    return (
        "Filters (mini-language):\n"
        "- Free-text across text columns: use filter=\"mot\" or filter=not \"mot\".\n"
        "- Quoted expressions with spaces are allowed: filter=\"deux mots\".\n"
        "- Combinations work: filter=\"aspirine\" and Date between \"2024-01-01\" and \"2024-12-31\".\n"
        "- Column-specific string ops still work: Ville.str.startswith(\"Par\"); Motif.str.contains(\"douleur\", case=False).\n"
    )

# ---- helpers numeric parsing -------------------------------------------------
import re
import numpy as np
import pandas as pd

def _to_numeric_smart(s: pd.Series, *, range_policy: str = "mean") -> pd.Series:
    """
    Convertit une série hétérogène (texte/nombres) en float.
    Gère notamment :
      - '50 min', '1h30', '01:30[:15]'  -> minutes (float)
      - '75-year-old'                   -> 75
      - 'between 3.4 and 5.8 cm'       -> moyenne/min/max des deux (range_policy)
      - '3,5-5,2' ou '3–5'             -> idem
      - '12 %'                          -> 0.12
      - décimales à virgule             -> 3,14 -> 3.14

    range_policy: 'mean' | 'min' | 'max'
    """
    s = s.copy()

    # 1) tentative directe
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= max(3, int(0.2 * len(s))):
        return num

    # 2) parseur unitaire
    def parse_one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).strip().lower()
        t = t.replace("\u202f", " ").replace("\xa0", " ")  # espaces insécables

        # -- time formats -> minutes --
        # a) hh:mm[:ss]
        m = re.match(r'^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$', t)
        if m:
            h, mnt, sec = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            return h*60 + mnt + sec/60.0

        # b) "XhY", "X h Y min/mn", "Xh", "X h"
        m = re.match(r'^\s*(\d+(?:[.,]\d+)?)\s*h(?:\s*(\d+(?:[.,]\d+)?)\s*(?:min|mn))?\s*$', t)
        if m:
            h = float(m.group(1).replace(',', '.'))
            mnt = float((m.group(2) or '0').replace(',', '.'))
            # support "1h30" sans 'min'
            if mnt == 0 and re.search(r'\d+h\d+$', t):
                mnt = float(re.findall(r'\d+h(\d+)$', t)[0])
            return h*60 + mnt

        # c) "X min" / "X mn"
        m = re.match(r'^\s*(\d+(?:[.,]\d+)?)\s*(?:min|mn)\s*$', t)
        if m:
            return float(m.group(1).replace(',', '.'))

        # d) "Xs" / "X sec/secondes" -> minutes
        m = re.match(r'^\s*(\d+(?:[.,]\d+)?)\s*(?:s|sec|secs|secondes?)\s*$', t)
        if m:
            return float(m.group(1).replace(',', '.')) / 60.0

        # -- pourcentage --
        m = re.match(r'^\s*(\d+(?:[.,]\d+)?)\s*%\s*$', t)
        if m:
            return float(m.group(1).replace(',', '.')) / 100.0

        # -- nombres isolés / plages --
        # normalise séparateurs décimaux
        # On capte toutes les occurrences numériques (garde virgule décimale)
        nums = re.findall(r'\d+(?:[.,]\d+)?', t)
        if not nums:
            return np.nan

        vals = [float(n.replace(',', '.')) for n in nums]

        # plage de deux nombres ( "3-5", "3 à 5", "3 and 5", "3–5" )
        if len(vals) >= 2 and re.search(r'\b(to|and|à)\b|[-–]', t):
            a, b = vals[0], vals[1]
            if range_policy == "min":
                return min(a, b)
            if range_policy == "max":
                return max(a, b)
            return (a + b) / 2.0  # mean par défaut

        # sinon: premier nombre trouvé
        return vals[0]

    out = s.apply(parse_one)

    # 3) si rien n'a été extrait, renvoyer quand même la tentative num initiale
    if out.notna().sum() == 0:
        return num
    return out.astype(float)

# ================================ Tools ====================================

def tool_table(df: pd.DataFrame, *, columns: Optional[List[str]] = None, limit: int = 50,
               sort: Optional[str] = None, title: Optional[str] = None, **vis) -> None:
    data = df.copy()
    if columns:
        cols = resolve_cols_names(df, columns)
        if not cols:
            raise ValueError(f"No valid columns resolved from: {columns}")
        data = data[cols]
    if sort:
        ascending = True
        key = sort
        if sort.startswith("-"):
            ascending = False
            key = sort[1:]
        keyname = resolve_column_name(data, key) or key
        if keyname in data.columns:
            data = data.sort_values(keyname, ascending=ascending)
    data = data.head(limit)

    rows = len(data)
    cols = len(data.columns)
    # largeur ~ 1.2 par colonne, hauteur ~ 0.55 par ligne (bornées), puis * FIG_SCALE
    w = max(8, min(22, 1.2 * cols + 2))
    h = max(4, min(24, 0.55 * rows + 2))
    fig, ax = plt.subplots(figsize=(w * FIG_SCALE, h * FIG_SCALE))
    _apply_visuals(fig, ax, _extract_visuals(vis))
    ax.axis("off")

    tbl = ax.table(cellText=data.values, colLabels=list(data.columns), loc="center",
    cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(int(9 * FONT_SCALE))
    # scale(x, y) agrandit les cellules → gagne en lisibilité
    tbl.scale(1.05 * FIG_SCALE, 1.15 * FIG_SCALE)
    # Autosize des colonnes en fonction des longueurs de texte (nécessite le renderer prêt)
    try:
        fig.canvas.draw()
        tbl.auto_set_column_width(col=list(range(len(data.columns))))
    except Exception:
        pass

    if title:
        ax.set_title(title, fontsize=int(12 * FONT_SCALE), pad=10)
    fig.tight_layout()
    plt.show()

def tool_histogram(df: pd.DataFrame, *, x: str, bins: int = 30, title: Optional[str] = None, **vis) -> None:
    s = resolve_series(df, x)
    try:
        num = _to_numeric_smart(s).dropna()
    except NameError:
        num = pd.to_numeric(s, errors="coerce").dropna()

    # Cas numérique -> vrai histo
    if len(num) > 0:
        fig, ax = new_fig(7, 5)
        _vis = _extract_visuals(vis)
        _apply_visuals(fig, ax, _vis)
        color = _normalize_color_token(_vis.get("color")) if _vis.get("color") else None
        n, b, _ = ax.hist(num, bins=(bins or "auto"), edgecolor="white", linewidth=0.6,
                        color=color)
        n, b, _ = ax.hist(num, bins=(bins or "auto"), edgecolor="white", linewidth=0.6)
        ax.set_xlabel(resolve_column_name(df, x) or x)
        ax.set_ylabel("count")
        beautify(ax, title)
        if len(n) <= 20:
            for count, left, right in zip(n, b[:-1], b[1:]):
                if count > 0:
                    ax.text((left + right) / 2, count, int(count),
                            ha="center", va="bottom", fontsize=9 * FONT_SCALE)
        plt.show()
        return

    # Fallback texte -> barh des tokens
    tokens = tokenize_series_if_listlike(s.astype(str))
    counts = tokens.value_counts().head(50)
    if counts.empty:
        fig, ax = new_fig(6, 3.5)
        ax.axis("off")
        ax.text(0.5, 0.5, f"No numeric or tokenizable data in '{x}'", ha="center", va="center")
        beautify(ax, title)
        plt.show()
        return

    fig, ax = new_fig(10, max(3, len(counts) * 0.22))
    
    bars = ax.barh(counts.index.astype(str)[::-1], counts.values[::-1])
    ax.set_xlabel("count")
    ax.set_ylabel(f"{resolve_column_name(df, x) or x} (top tokens)")
    beautify(ax, (title + " (auto: tokens)") if title else "Token frequencies (auto)")
    try:
        ax.bar_label(bars, labels=[str(v) for v in counts.values[::-1]], padding=3)
    except Exception:
        pass
    plt.show()


def tool_bar(df: pd.DataFrame, *, 
             x: str, 
             y: Optional[str] = None, 
             agg: str = "sum",
             title: Optional[str] = None, 
             **vis) -> None:
    """
    - Si y est fourni: agrège par x avec mean/sum/count robustes.
    - Si y est texte/non-numérique: conversion intelligente; sinon fallback count.
    - Si y est absent: bar des fréquences de x (tokenized si nécessaire).
    """
    xs = resolve_series(df, x)

    # Helper de conversion numérique (utilise _to_numeric_smart si dispo)
    def to_num(s: pd.Series) -> pd.Series:
        try:
            return _to_numeric_smart(s)   # ton helper si tu l'as ajouté
        except NameError:
            return pd.to_numeric(s, errors="coerce")

    if y:
        ys_raw = resolve_series(df, y)
        ys_num = to_num(ys_raw)

        data = pd.DataFrame({x: xs, y: ys_num})
        g = data.groupby(x, dropna=False)[y]
        agg_l = (agg or "sum").lower()

        # SeriesGroupBy n’accepte pas numeric_only => on enlève cet argument
        if agg_l in {"mean", "avg"}:
            vals = g.mean()
        elif agg_l in {"sum", "total"}:
            # min_count=1 évite 0 “magique” si toutes valeurs NaN
            try:
                vals = g.sum(min_count=1)
            except TypeError:
                vals = g.sum()
        elif agg_l in {"count", "size"}:
            vals = data.groupby(x, dropna=False)[y].count()
        else:
            # défaut: si on a des num valides → sum ; sinon → count
            vals = g.sum(min_count=1) if ys_num.notna().any() else data.groupby(x, dropna=False).size()

        # Fallback ultime: si tout est NaN / vide → count par x
        if isinstance(vals, pd.Series) and (vals.fillna(0) == 0).all():
            vals = data.groupby(x, dropna=False).size()

        vals = vals.sort_values(ascending=False).head(50)

        fig, ax = new_fig(10, max(3, len(vals) * 0.22))
        _vis = _extract_visuals(vis)
        _apply_visuals(fig, ax, _vis)
        color = _normalize_color_token(_vis.get("color")) if _vis.get("color") else None
        bars = ax.barh(vals.index.astype(str)[::-1], vals.values[::-1], color=color)
        ax.set_xlabel(agg_l if agg_l in {"mean","avg","sum","total","count","size"} else "value")
        ax.set_ylabel(resolve_column_name(df, x) or x)
        beautify(ax, title)
        try:
            ax.bar_label(bars, labels=[str(v) for v in vals.values[::-1]], padding=3)
        except Exception:
            pass
        plt.show()
        return

    # ---- Cas sans y: fréquences de x (tokenization si colonnes texte list-like) ----
    cat = tokenize_series_if_listlike(xs.astype(str))
    vals = cat.value_counts(dropna=False).head(50)
    fig, ax = new_fig(10, max(3, len(vals) * 0.22))
    _vis = _extract_visuals(vis)
    _apply_visuals(fig, ax, _vis)
    color = _normalize_color_token(_vis.get("color")) if _vis.get("color") else None
    bars = ax.barh(vals.index.astype(str)[::-1], vals.values[::-1], color=color)
    ax.set_xlabel("count")
    ax.set_ylabel(resolve_column_name(df, x) or x)
    beautify(ax, title)
    try:
        ax.bar_label(bars, labels=[str(v) for v in vals.values[::-1]], padding=3)
    except Exception:
        pass
    plt.show()


def tool_line(df: pd.DataFrame, *, x: str, y: str, group: Optional[str] = None,
              title: Optional[str] = None, **vis) -> None:
    xs = resolve_series(df, x); ys = resolve_series(df, y)
    xdt = coerce_to_datetime(xs); x_is_date = xdt.notna().sum() > len(xs) * 0.3
    if x_is_date:
        X = xdt
    else:
        try:
            X = _to_numeric_smart(xs)
        except NameError:
            X = pd.to_numeric(xs, errors="coerce")
    try:
        Y = _to_numeric_smart(ys)
    except NameError:
        Y = pd.to_numeric(ys, errors="coerce")
    data = pd.DataFrame({"__x__": X, "__y__": Y})
    if group: data["__g__"] = resolve_series(df, group).astype(str)
    data = data.dropna(subset=["__x__", "__y__"])
    if data.empty: raise ValueError("No valid data to plot line.")

    fig, ax = new_fig(8, 5)
    _vis = _extract_visuals(vis)
    _apply_visuals(fig, ax, _vis)
    color = _normalize_color_token(_vis.get("color")) if _vis.get("color") else None
    if "__g__" in data.columns:
        for key, chunk in data.groupby("__g__"):
            chunk = chunk.sort_values("__x__")
            ax.plot(chunk["__x__"], chunk["__y__"], label=str(key))
        ax.legend(title=group)
    else:
        data = data.sort_values("__x__")
        ax.plot(data["__x__"], data["__y__"], color=color)

    ax.set_xlabel(resolve_column_name(df, x) or x)
    ax.set_ylabel(resolve_column_name(df, y) or y)
    beautify(ax, title)
    plt.show()

def tool_scatter(df: pd.DataFrame, *,
                 x: str,
                 y: str,
                 group: Optional[str] = None,
                 color_by: Optional[str] = None,   # alias accepté comme pour timeline
                 title: Optional[str] = None,
                 embed_fallback: bool = True,      # ← NEW : active le fallback texte->embedding
                 **vis) -> None:
    """
    Scatter 2D robuste :
      - Si X/Y numériques -> comportement classique.
      - Si un seul axe est texte -> embedding + PCA 1D pour cet axe.
      - Si les deux axes sont texte -> embedding du texte combiné (x || y) + PCA 2D.
    """
    # 0) mapping group
    gcol = group or color_by

    # 1) récup brutes
    sx_raw = resolve_series(df, x)
    sy_raw = resolve_series(df, y)

    # 2) try numeric direct (avec ton helper si tu l'as, sinon to_numeric)
    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    xs_num = to_num(sx_raw)
    ys_num = to_num(sy_raw)

    x_ok = xs_num.notna().sum() >= 3
    y_ok = ys_num.notna().sum() >= 3

    # 3) fallback embeddings si besoin
    xs_final, ys_final = xs_num.copy(), ys_num.copy()
    x_lab = resolve_column_name(df, x) or x
    y_lab = resolve_column_name(df, y) or y

    try:
        if embed_fallback and not (x_ok and y_ok):
            if x_ok and not y_ok:
                # Y texte -> PCA 1D sur embeddings(y)
                Ey = _embed_series_text(sy_raw)
                pc1 = _pca_np(Ey, 1)[:, 0]
                ys_final = pd.Series(pc1, index=sy_raw.index)
                y_lab = f"{y_lab} (PC1 embedding)"
                y_ok = True
            elif (not x_ok) and y_ok:
                # X texte -> PCA 1D sur embeddings(x)
                Ex = _embed_series_text(sx_raw)
                pc1 = _pca_np(Ex, 1)[:, 0]
                xs_final = pd.Series(pc1, index=sx_raw.index)
                x_lab = f"{x_lab} (PC1 embedding)"
                x_ok = True
            elif (not x_ok) and (not y_ok):
                # Les deux axes texte -> embedding du texte combiné + PCA 2D
                combo = sx_raw.fillna("").astype(str) + " || " + sy_raw.fillna("").astype(str)
                E = _embed_series_text(combo)
                pcs = _pca_np(E, 2)
                xs_final = pd.Series(pcs[:, 0], index=sx_raw.index)
                ys_final = pd.Series(pcs[:, 1], index=sy_raw.index)
                x_lab = f"PC1({x}+{y} embeddings)"
                y_lab = f"PC2({x}+{y} embeddings)"
                x_ok = y_ok = True
    except Exception as e:
        # si embedding indispo (lib pas installée), on continue sans fallback
        print(f"[WARN] Embedding fallback disabled or failed: {e}")

    # 4) assemble les données
    data = pd.DataFrame({"__x__": xs_final, "__y__": ys_final})
    if gcol:
        data["__g__"] = resolve_series(df, gcol).astype(str)

    data = data.dropna(subset=["__x__", "__y__"])
    if data.empty or not (x_ok and y_ok):
        raise ValueError("No valid numeric data for scatter (and embedding fallback unavailable).")

    # 5) rendu
    fig, ax = new_fig(7, 5)
    _vis = _extract_visuals(vis)
    _apply_visuals(fig, ax, _vis)
    color = _normalize_color_token(_vis.get("color")) if _vis.get("color") else None

    if "__g__" in data.columns:
        for key, chunk in data.groupby("__g__"):
            ax.scatter(chunk["__x__"], chunk["__y__"], label=str(key))
        ax.legend(title=(resolve_column_name(df, gcol) or gcol))
    else:
        ax.scatter(data["__x__"], data["__y__"], color=color)

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    beautify(ax, title)
    plt.show()



def tool_timeline(df: pd.DataFrame, *, date: str, label: Optional[str] = None,
                  color_by: Optional[str] = None, title: Optional[str] = None, **vis) -> None:
    dser = resolve_series(df, date); dser = coerce_to_datetime(dser)
    lser = resolve_series(df, label).astype(str) if label else pd.Series([""] * len(df), index=df.index)
    data = pd.DataFrame({"__date__": dser, "__label__": lser})
    if color_by: data["__color__"] = resolve_series(df, color_by).astype(str)
    data = data.dropna(subset=["__date__"]).sort_values("__date__")
    if data.empty: raise ValueError(f"No valid dates in '{date}'.")

    fig, ax = new_fig(12, 4)
    _vis = _extract_visuals(vis)
    _apply_visuals(fig, ax, _vis)
    point_color = _normalize_color_token(_vis.get("color")) if _vis.get("color") else None
    if "__color__" in data.columns:
        for key, chunk in data.groupby("__color__"):
            ax.scatter(chunk["__date__"], np.zeros(len(chunk)), label=str(key))
        ax.legend(title=color_by, loc="upper left", bbox_to_anchor=(1,1))
    else:
        ax.scatter(data["__date__"], np.zeros(len(data)), color=point_color)

    for i, (dt, lab) in enumerate(zip(data["__date__"], data["__label__"])):
        dy = 0.1 if (i % 2) else -0.1
        ax.text(dt, dy, str(lab or ""), rotation=45, ha="right", va="center")
    ax.set_yticks([])
    ax.set_xlabel(resolve_column_name(df, date) or date)
    beautify(ax, title)
    plt.show()


# ================================ Registry =================================

TOOL_REGISTRY: Dict[str, Callable[..., None]] = {
    "table":     tool_table,
    "histogram": tool_histogram,
    "bar":       tool_bar,
    "line":      tool_line,
    "scatter":   tool_scatter,
    "timeline":  tool_timeline,
}

# ============================ Compiler & Runner ============================

class PlanExecutionError(Exception):
    def __init__(self, failed_command: 'VisCommand', error: str, tb: str, logs: List[Dict[str, Any]]):
        super().__init__(error)
        self.failed_command = failed_command
        self.error = error
        self.traceback = tb
        self.logs = logs

def compile_and_run(df: pd.DataFrame, commands: List[VisCommand], stop_on_error: bool = True) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    for cmd in commands:
        params = dict(cmd.params)

        # >>> hardening columns (évite une pollution résiduelle)
        if "columns" in params and isinstance(params["columns"], list):
            params["columns"] = [c.strip() for c in params["columns"] if c and "=" not in c]

        if "filter" in params:
            flt = params.pop("filter"); wdf = apply_filter(df, flt)
        else:
            wdf = df

        tool = TOOL_REGISTRY.get(cmd.kind)
        if not tool:
            raise ValueError(f"No tool registered for: {cmd.kind}")
        if "hue" in params and "group" not in params:
            params["group"] = params.pop("hue")

        print(f"[RUN] {cmd.kind} {params}")
        try:
            tool(wdf, **params)
            logs.append({"status":"ok","cmd":cmd.kind,"params":params})
        except Exception as e:
            tb = traceback.format_exc()
            logs.append({"status":"error","cmd":cmd.kind,"params":params,"error":str(e)})
            if stop_on_error:
                raise PlanExecutionError(VisCommand(cmd.kind, params), str(e), tb, logs)
    return logs


# =============================== Planner(s) ================================

def naive_planner_from_prompt(prompt: str, df: pd.DataFrame) -> List[VisCommand]:
    p = prompt.lower()
    cmds: List[VisCommand] = []
    if any(k in p for k in ["timeline", "frise"]):
        date_guess = next((c for c in df.columns if "date" in str(c).lower()), None) or df.columns[0]
        label_guess = next((c for c in df.columns if str(c).lower() in {"title","name","filename","file","document"}), None) or df.columns[0]
        cmds.append(VisCommand("timeline", {"date": date_guess, "label": label_guess}))
    if "scatter" in p or "dispersion" in p:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(cols) >= 2:
            cmds.append(VisCommand("scatter", {"x": cols[0], "y": cols[1]}))
    if "hist" in p or "histogram" in p:
        col = next((c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])), None)
        if col:
            cmds.append(VisCommand("histogram", {"x": col, "bins": 30}))
    if "bar" in p:
        cat = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), None) or df.columns[0]
        cmds.append(VisCommand("bar", {"x": cat}))
    if "table" in p or "preview" in p:
        cmds.append(VisCommand("table", {"limit": 20}))
    if not cmds:
        cmds.append(VisCommand("table", {"limit": 20}))
    return cmds

# =============================== IO helpers ================================

def load_dataframe(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        try:
            df = pd.read_excel(path, sheet_name=(sheet if sheet is not None else 0))
        except Exception:
            df = pd.read_excel(path, sheet_name=sheet)
            if isinstance(df, dict):
                df = next(iter(df.values()))
    else:
        df = pd.read_csv(path)
    df.columns = [clean_column_name(c) for c in df.columns]
    # deduplicate names: add suffixes _2, _3 ...
    if pd.Index(df.columns).duplicated().any():
        counts = {}
        newcols = []
        for c in df.columns:
            if c not in counts:
                counts[c] = 1; newcols.append(c)
            else:
                counts[c] += 1; newcols.append(f"{c}_{counts[c]}")
        df.columns = newcols
    return df

def parse_commands_from_args(plan: Optional[str], plan_json: Optional[str], plan_file: Optional[str]) -> List[VisCommand]:
    if plan_json: return parse_plan_json(plan_json)
    if plan_file:
        txt = Path(plan_file).read_text(encoding="utf-8")
        t = txt.lstrip()
        return parse_plan_json(txt) if t.startswith("{") or t.startswith("[") else parse_plan_minilang(txt)
    if plan:
        t = plan.lstrip()
        return parse_plan_json(plan) if t.startswith("{") or t.startswith("[") else parse_plan_minilang(plan)
    return []

# ========================= LLM Planner (Mistral) ===========================

def _build_system_prompt() -> str:
    return (
        "You translate a user's natural-language request into a strict mini-language for data visualization.\n"
        "Output ONLY the mini-language lines (no prose, no markdown, no backticks).\n\n"
        "Rules:\n"
        "0) Use ONLY columns from the provided lists. Correct approximate names to exact column names.\n"
        "1) Always keep filters in 'filter=...'. NEVER concatenate filters into other keys (e.g., x, columns, etc.).\n"
        "2) If the user mentions a domain/category (e.g., médical/medical), and a category-like column is available,\n"
        "   then add an appropriate filter that selects the matching category values (case/accent-insensitive;\n"
        "   choose from the provided category values list).\n"
        "3) For 'timeline', pick the best date column (from 'date candidates') and a descriptive label column\n"
        "   (prefer columns like motif/raison/title/type_doc_clinique when available).\n"
        "4) histogram/line/scatter must use numeric x/y. If the requested data is textual, prefer bar with value counts\n"
        "   or instruct tokens frequency (bar) when appropriate.\n"
        "5) Keep plans simple and valid. One command per line: 'kind: key=value; key2=value2'.\n"
        "6) Free-text filter: you may write filter=\"word\" (or filter=not \"word\") to match that word across all text/categorical columns; quoted expressions with spaces are allowed (e.g., filter=\"deux mots\").\n"
        "7) Visual modifiers:\n"
        "   - Use 'bg=<color>' for figure/axes background (e.g., bg=blue, bg=beige, bg=transparent).\n"
        "   - Use 'color=<color>' for the primary mark (bars, line, points).\n"
        "   - Use 'grid=on|off' to toggle grid.\n"
        "   - Map natural language to these keys: 'fond bleu' → bg=blue; 'barres rouges' → color=red.\n"
        "   - Colors accept French/English names or hex (#RRGGBB). Keep pairs top-level (never inside parentheses).\n"

    )


def _build_user_prompt(user_prompt: str, df_columns: List[str], df: Optional[pd.DataFrame] = None) -> str:
    cols = "\n".join(f"- {c}" for c in df_columns)

    extras = []
    tools = build_tools_schema_text()
    extras.append(tools)
    extras.append(build_filters_cheatsheet_text())

    if df is not None:
        # candidats
        date_cands  = guess_date_columns(df)
        label_cands = guess_label_columns(df)
        if date_cands:
            extras.append("Date candidates:\n" + "\n".join(f"- {c}" for c in date_cands))
        if label_cands:
            extras.append("Label candidates:\n" + "\n".join(f"- {c}" for c in label_cands))

        # colonnes catégorie + valeurs
        cat_cols = detect_category_columns(df)
        if cat_cols:
            seen = set()
            lines = []
            for cc in cat_cols[:3]:  # on ne liste que les 3 plus pertinentes
                if cc in seen: continue
                seen.add(cc)
                vals = top_values(df, cc, k=40)
                if vals:
                    lines.append(f"- {cc}: " + ", ".join(vals))
            if lines:
                extras.append("Category columns and their values (top):\n" + "\n".join(lines))

    examples = (
        "Examples:\n"
        "Request: Timeline des rendez-vous médicaux avec le motif de consultation\n"
        "Output:\n"
        "timeline: date=DateDoc; label=Motif; filter=category in [\"RDV médical\",\"Consultation\",\"Clinique\"]; title=Timeline des rendez-vous médicaux par motif\n\n"
        "Request: Histogramme du nombre d'ingrédients (catégorie Recettes)\n"
        "Output:\n"
        "histogram: x=IngredientsCount; bins=20; filter=category.str.contains(\"recett\", case=False); title=Distribution des ingrédients\n"
    )

    extra_text = "\n\n".join(extras)
    return (
        f"Dataset columns:\n{cols}\n\n"
        f"{extra_text}\n\n"
        f"User request:\n{user_prompt}\n\n"
        f"{examples}"
        "Now produce ONLY the mini-language lines for this dataset (French titles if user wrote French)."
    )


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import requests  # type: ignore
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        import json as _json
        import urllib.request
        req = urllib.request.Request(url, _json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

def strip_code_fences(text: str) -> str:
    m = re.search(r"```(?:[a-zA-Z]+)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def mistral_minilang_from_prompt(
    prompt: str,
    df: pd.DataFrame,
    *,
    model: str = "mistral-medium-2508",
    endpoint: str = "https://api.mistral.ai/v1/chat/completions",
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    openai_compatible: bool = False,
    oa_endpoint: Optional[str] = None,
    oa_api_key: Optional[str] = None,
) -> str:
    sys_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(prompt, list(df.columns), df)
    if openai_compatible:
        url = oa_endpoint or "http://localhost:11434/v1/chat/completions"
        key = oa_api_key or ""
        headers = {"Content-Type": "application/json"}
        if key: headers["Authorization"] = f"Bearer {key}"
        payload = {"model": model, "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}], "temperature": temperature}
        data = _http_post_json(url, headers, payload)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return strip_code_fences(content)
    key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
    if not key: raise ValueError("Missing Mistral API key. Set --mistral-api-key or env MISTRAL_API_KEY.")
    url = endpoint
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}], "temperature": temperature}
    data = _http_post_json(url, headers, payload)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return strip_code_fences(content)

# ----------------------- Replan on failure (LLM) -----------------------

def df_signature(df: pd.DataFrame, max_examples: int = 3) -> str:
    """Build a compact signature of columns, dtypes, and a few examples per column."""
    lines = []
    for c in df.columns:
        ser = df[c]
        dtype = str(ser.dtype)
        ex = ser.dropna().astype(str).unique()[:max_examples]
        ex_s = ", ".join(map(lambda s: s[:50], ex))
        lines.append(f"- {c} :: {dtype} :: {ex_s}")
    return "\n".join(lines)

def commands_to_minilang(commands: List[VisCommand]) -> str:
    parts = []
    for cmd in commands:
        kv = "; ".join(f"{k}={v if not isinstance(v,list) else ','.join(v)}" for k,v in cmd.params.items())
        parts.append(f"{cmd.kind}: {kv}")
    return "\n".join(parts)

def _build_replan_prompt(user_prompt: str, dataset_sig: str, prev_plan: str, failure_logs: List[Dict[str, Any]]) -> str:
    fail_lines = []
    for log in failure_logs[-3:]:
        if log.get("status") == "error":
            fail_lines.append(f"- Failed: {log.get('cmd')} {log.get('params')} -> {log.get('error')}")
    fails = "\n".join(fail_lines) if fail_lines else "- (no prior errors?)"
    return (
        "Dataset columns, types and examples:\n"
        f"{dataset_sig}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Previous plan (mini-language):\n"
        f"{prev_plan}\n\n"
        "Execution errors:\n"
        f"{fails}\n\n"
        "Task: Produce a CORRECTED mini-language plan that satisfies the user with these rules:\n"
        "- Use ONLY existing columns; fix column names if they were approximate.\n"
        "- If the requested chart mismatches the column type, choose a better tool (e.g. tokens bar for text).\n"
        "- For dates with duplicates/dirty formats, pick the most valid-looking column.\n"
        "- Keep it simple and valid. Output ONLY the mini-language lines."
    )

def mistral_replan_from_failure(
    user_prompt: str,
    df: pd.DataFrame,
    previous_commands: List[VisCommand],
    failure_logs: List[Dict[str, Any]],
    *,
    model: str,
    endpoint: str,
    api_key: Optional[str],
    temperature: float,
    openai_compatible: bool,
    oa_endpoint: Optional[str],
    oa_api_key: Optional[str],
) -> str:
    sys_prompt = _build_system_prompt()
    dataset_sig = df_signature(df)
    prev_plan = commands_to_minilang(previous_commands)
    user = _build_replan_prompt(user_prompt, dataset_sig, prev_plan, failure_logs)

    if openai_compatible:
        url = oa_endpoint or "http://localhost:11434/v1/chat/completions"
        key = oa_api_key or ""
        headers = {"Content-Type": "application/json"}
        if key: headers["Authorization"] = f"Bearer {key}"
        payload = {"model": model, "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":user}], "temperature": temperature}
        data = _http_post_json(url, headers, payload)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return strip_code_fences(content)

    key = api_key or os.environ.get("MISTRAL_API_KEY") or ""
    if not key: raise ValueError("Missing Mistral API key. Set --mistral-api-key or env MISTRAL_API_KEY.")
    url = endpoint
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":user}], "temperature": temperature}
    data = _http_post_json(url, headers, payload)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return strip_code_fences(content)

# =================================== CLI ===================================

def main():
    ap = argparse.ArgumentParser(description="Auto data-vis pipeline (planner→compiler→tools) + LLM retries")
    # LLM planner options
    ap.add_argument("--planner", choices=["naive", "mistral"], default="naive", help="Planner for initial plan.")
    ap.add_argument("--temperature", type=float, default=0.1, help="LLM temperature.")
    ap.add_argument("--mistral-model", default="mistral-medium-2508", help="Mistral model name.")
    ap.add_argument("--mistral-endpoint", default="https://api.mistral.ai/v1/chat/completions", help="Mistral API endpoint.")
    ap.add_argument("--mistral-api-key", default=None, help="Mistral API key (or env MISTRAL_API_KEY).")
    # OpenAI-compatible mode
    ap.add_argument("--openai-compatible", action="store_true", help="Use OpenAI-compatible /v1/chat/completions.")
    ap.add_argument("--oa-endpoint", default="http://localhost:11434/v1/chat/completions", help="OpenAI-compatible endpoint.")
    ap.add_argument("--oa-api-key", default=None, help="Bearer token for OpenAI-compatible endpoint.")
    # Data & plan
    ap.add_argument("data", help="Path to CSV or Excel file")
    ap.add_argument("--sheet", help="Excel sheet name (if Excel)")
    ap.add_argument("--plan", help="Mini-language plan string")
    ap.add_argument("--plan-json", help="JSON plan string")
    ap.add_argument("--plan-file", help="Path to a .txt or .json plan file")
    ap.add_argument("--prompt", help="Free-text hint (used by planner if no plan provided)")
    # Retries
    ap.add_argument("--max-retries", type=int, default=3, help="Max LLM replans on failure (default 3).")
    args = ap.parse_args()

    df = load_dataframe(args.data, sheet=args.sheet)
    if isinstance(df, dict):
        if args.sheet and args.sheet in df: df = df[args.sheet]
        else:
            print("[INFO] Multiple sheets detected; using the first by default. Pass --sheet to choose.")
            df = next(iter(df.values()))

    commands = parse_commands_from_args(args.plan, args.plan_json, args.plan_file)
    if not commands:
        if not args.prompt:
            print("[INFO] No explicit plan given. Using a planner on the column types.")
            args.prompt = "table, histogram, bar, scatter, timeline"
        if args.planner == "mistral":
            mini = mistral_minilang_from_prompt(
                args.prompt, df,
                model=args.mistral_model, endpoint=args.mistral_endpoint, api_key=args.mistral_api_key,
                temperature=args.temperature, openai_compatible=args.openai_compatible,
                oa_endpoint=args.oa_endpoint, oa_api_key=args.oa_api_key,
            )
            commands = parse_plan_minilang(mini)
        else:
            commands = naive_planner_from_prompt(args.prompt, df)

    # Run with up to N replans on failure
    attempts = 0
    aggregate_logs: List[Dict[str, Any]] = []
    while True:
        try:
            logs = compile_and_run(df, commands, stop_on_error=True)
            aggregate_logs.extend(logs)
            break  # success
        except PlanExecutionError as e:
            aggregate_logs.extend(e.logs)
            attempts += 1
            if args.planner != "mistral" or attempts > args.max_retries:
                print(f"[ERROR] Execution failed after {attempts-1} replans. Last error:\n{e.traceback}")
                raise
            print(f"[INFO] Attempt {attempts} failed. Asking LLM to REPLAN…")
            # Build a corrected plan using failure logs + dataset signature
            mini = mistral_replan_from_failure(
                args.prompt or "(no user prompt provided)",
                df,
                commands,
                aggregate_logs,
                model=args.mistral_model, endpoint=args.mistral_endpoint, api_key=args.mistral_api_key,
                temperature=args.temperature, openai_compatible=args.openai_compatible,
                oa_endpoint=args.oa_endpoint, oa_api_key=args.oa_api_key,
            )
            print("[INFO] New plan from LLM:\n" + mini)
            commands = parse_plan_minilang(mini)

if __name__ == "__main__":
    main()
