# -*- coding: utf-8 -*-
"""08 — Bond Stress Studio (masaüstü uygulama altlığı)

Maksimum aderans dayanımı (tau_max) tahmini için uçtan uca masaüstü aracı:
  1) Veri    : Excel seç/yükle, aykırı filtre (Grubbs / IQR, kolon seçimli)
  2) Eğitim  : RF / XGBoost / SVR + Bayesian hiperparametre araması, senaryo filtresi
  3) Grafik  : parite, BO yakınsaması, özellik önemi, artıklar, serbest saçılım;
               eksenler canlı değiştirilebilir (değişken, log, min/maks) + zoom/pan
  4) Tahmin  : eğitilen modelle tek numune tahmini (eksik özellik bırakılabilir)

Çalıştırma:  python 08_bond_stress_app.py
EXE paketleme (pyinstaller kurulu ortamda):
  pyinstaller --onefile --windowed --collect-all xgboost --collect-all sklearn ^
              --collect-all skopt 08_bond_stress_app.py

Not: 02 defterindeki ANN (PyTorch) bilerek dahil edilmedi — torch, exe boyutunu
GB mertebesine taşır. RF/XGB/SVR aynı Bayesian arama düzeniyle mevcuttur.
"""

import os
import threading
import queue
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except ImportError:      # skopt yoksa rastgele aramaya düşülür (uyarı verilir)
    from sklearn.model_selection import RandomizedSearchCV
    HAS_SKOPT = False

# ══════════════════════════════════════════════════════════════════════
#  Proje sabitleri (pinn_common ile tutarlı; uygulama tek dosya kalsın diye kopya)
# ══════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "bar_diameter_mm",
    "nominal_yield_strength_mpa",
    "concrete_strength_mpa",
    "embedment_length_db",
    "concrete_cover_db",
    "transverse_reinforcement",
    "transverse_reinforcement_mm",
    "transverse_reinforcement_bar_diameter_mm",
]
TARGET_COL = "maximum_bond_stress_mpa"
TR_FLAG = "transverse_reinforcement"
TR_GEO_COLS = ["transverse_reinforcement_mm",
               "transverse_reinforcement_bar_diameter_mm"]

DEFAULT_FILTER_COLS = FEATURE_COLS + [TARGET_COL, "maximum_slip_mm"]

SCENARIOS = [
    {"scenario_name": "tum_veri", "filters": {}},
    {"scenario_name": "plain_all", "filters": {"bar_type": "plain"}},
    {"scenario_name": "deformed_all", "filters": {"bar_type": "deformed"}},
    {"scenario_name": "plain_fc_0_20", "filters": {"bar_type": "plain", "concrete_strength_mpa": (None, 20)}},
    {"scenario_name": "plain_fc_gte20", "filters": {"bar_type": "plain", "concrete_strength_mpa": (20, None)}},
    {"scenario_name": "deformed_fc_0_55", "filters": {"bar_type": "deformed", "concrete_strength_mpa": (None, 55)}},
    {"scenario_name": "deformed_fc_gte55", "filters": {"bar_type": "deformed", "concrete_strength_mpa": (55, None)}},
    {"scenario_name": "deformed_tr_1", "filters": {"bar_type": "deformed", "transverse_reinforcement": 1}},
    {"scenario_name": "deformed_tr_0", "filters": {"bar_type": "deformed", "transverse_reinforcement": 0}},
    {"scenario_name": "deformed_fc_0_55_tr_0", "filters": {"bar_type": "deformed", "concrete_strength_mpa": (None, 55), "transverse_reinforcement": 0}},
    {"scenario_name": "deformed_fc_0_55_tr_1", "filters": {"bar_type": "deformed", "concrete_strength_mpa": (None, 55), "transverse_reinforcement": 1}},
    {"scenario_name": "deformed_fc_gte55_tr_0", "filters": {"bar_type": "deformed", "concrete_strength_mpa": (55, None), "transverse_reinforcement": 0}},
    {"scenario_name": "deformed_fc_gte55_tr_1", "filters": {"bar_type": "deformed", "concrete_strength_mpa": (55, None), "transverse_reinforcement": 1}},
]
SCENARIO_MAP = {s["scenario_name"]: s["filters"] for s in SCENARIOS}


def filter_scenario(df, filters):
    data = df.copy()
    for col, rule in filters.items():
        if rule is None:
            continue
        if col == "transverse_reinforcement":
            vals = pd.to_numeric(data[col], errors="coerce")
            if rule in (1, "1"):
                data = data[vals == 1]
            elif rule in (0, "0"):
                data = data[(vals == 0) | (vals.isna())]
            continue
        if isinstance(rule, tuple):
            lo, hi = rule
            vals = pd.to_numeric(data[col], errors="coerce")
            mask = pd.Series(True, index=data.index)
            if lo is not None:
                mask &= vals > lo
            if hi is not None:
                mask &= vals <= hi
            data = data[mask]
        else:
            data = data[data[col] == rule]
    return data.reset_index(drop=True)


class MissingAwareScaler(BaseEstimator, TransformerMixin):
    """Gözlenen değerler standardize edilir; eksikler doldurulmaz, ayrı
    'mevcut mu' kanalıyla modele bildirilir (02 ile birebir aynı yaklaşım)."""

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.mean_ = np.zeros(self.n_features_in_)
        self.scale_ = np.ones(self.n_features_in_)
        for j in range(self.n_features_in_):
            obs = arr[np.isfinite(arr[:, j]), j]
            if len(obs):
                self.mean_[j] = obs.mean()
                std = obs.std()
                self.scale_[j] = std if std > 1e-12 else 1.0
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        avail = np.isfinite(arr).astype(float)
        z = (arr - self.mean_) / self.scale_
        z[~np.isfinite(arr)] = 0.0
        return np.concatenate([z, avail], axis=1)


def apply_tr_logic(feats):
    """TR=0/boş -> geometri kolonları eksik sayılır; TR=1 -> gözlenenler girer."""
    feats = feats.copy()
    feats[TR_FLAG] = pd.to_numeric(feats[TR_FLAG], errors="coerce").fillna(0)
    active = feats[TR_FLAG].eq(1)
    feats.loc[~active, TR_GEO_COLS] = np.nan
    return feats


# ══════════════════════════════════════════════════════════════════════
#  Aykırı değer filtreleri (kolon bazlı; aykırı içeren SATIR elenir)
# ══════════════════════════════════════════════════════════════════════

def grubbs_outlier_rows(df, cols, alpha=0.05):
    """Tek geçişli iki yönlü Grubbs: her kolonda G > G_crit olan uç değer aykırıdır."""
    bad = set()
    detail = []
    for col in cols:
        v = pd.to_numeric(df[col], errors="coerce")
        obs = v.dropna()
        n = len(obs)
        if n < 3:
            continue
        t2 = stats.t.ppf(1 - alpha / (2 * n), n - 2) ** 2
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t2 / (n - 2 + t2))
        mu, sd = obs.mean(), obs.std(ddof=1)
        if sd <= 1e-12:
            continue
        g = (obs - mu).abs() / sd
        idx = obs.index[g > g_crit]
        bad.update(idx.tolist())
        detail.append((col, len(idx), g_crit))
    return bad, detail


def iqr_outlier_rows(df, cols, k=1.5):
    """Klasik IQR kuralı: [Q1 - k*IQR, Q3 + k*IQR] dışındaki değerler aykırıdır."""
    bad = set()
    detail = []
    for col in cols:
        v = pd.to_numeric(df[col], errors="coerce")
        obs = v.dropna()
        if len(obs) < 4:
            continue
        q1, q3 = obs.quantile(0.25), obs.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 1e-12:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        idx = obs.index[(obs < lo) | (obs > hi)]
        bad.update(idx.tolist())
        detail.append((col, len(idx), (lo, hi)))
    return bad, detail


# ══════════════════════════════════════════════════════════════════════
#  Model tanımları ve Bayesian arama uzayları
# ══════════════════════════════════════════════════════════════════════

def model_space(name, random_state):
    if name == "Random Forest":
        est = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        if HAS_SKOPT:
            space = {"model__n_estimators": Integer(100, 800),
                     "model__max_depth": Integer(3, 20),
                     "model__min_samples_leaf": Integer(1, 10),
                     "model__max_features": Real(0.3, 1.0)}
        else:
            space = {"model__n_estimators": [100, 200, 400, 800],
                     "model__max_depth": [3, 6, 10, 20],
                     "model__min_samples_leaf": [1, 2, 5, 10],
                     "model__max_features": [0.3, 0.6, 1.0]}
    elif name == "XGBoost":
        est = XGBRegressor(objective="reg:squarederror",
                           random_state=random_state, n_jobs=-1)
        if HAS_SKOPT:
            space = {"model__n_estimators": Integer(100, 800),
                     "model__max_depth": Integer(2, 8),
                     "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                     "model__subsample": Real(0.6, 1.0),
                     "model__colsample_bytree": Real(0.6, 1.0),
                     "model__reg_lambda": Real(0.1, 10.0, prior="log-uniform")}
        else:
            space = {"model__n_estimators": [100, 200, 400, 800],
                     "model__max_depth": [2, 4, 6, 8],
                     "model__learning_rate": [0.01, 0.05, 0.1, 0.3],
                     "model__subsample": [0.6, 0.8, 1.0],
                     "model__reg_lambda": [0.1, 1.0, 10.0]}
    elif name == "SVR":
        est = SVR()
        if HAS_SKOPT:
            space = {"model__C": Real(0.1, 100.0, prior="log-uniform"),
                     "model__gamma": Real(1e-3, 1.0, prior="log-uniform"),
                     "model__epsilon": Real(0.01, 1.0, prior="log-uniform")}
        else:
            space = {"model__C": [0.1, 1, 10, 100],
                     "model__gamma": [1e-3, 1e-2, 0.1, 1.0],
                     "model__epsilon": [0.01, 0.1, 0.5, 1.0]}
    else:
        raise ValueError(name)
    return Pipeline([("scaler", MissingAwareScaler()), ("model", est)]), space


# ══════════════════════════════════════════════════════════════════════
#  Uygulama
# ══════════════════════════════════════════════════════════════════════

class BondStressApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bond Stress Studio — 08 (altlık sürüm)")
        self.geometry("1180x760")

        self.df_raw = None          # yüklenen ham veri
        self.df = None              # filtre uygulanmış veri
        self.results = {}           # model adı -> sonuç sözlüğü
        self.log_queue = queue.Queue()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab_data = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb)
        self.tab_plot = ttk.Frame(nb)
        self.tab_pred = ttk.Frame(nb)
        nb.add(self.tab_data, text=" 1) Veri & Filtre ")
        nb.add(self.tab_train, text=" 2) Eğitim ")
        nb.add(self.tab_plot, text=" 3) Grafikler ")
        nb.add(self.tab_pred, text=" 4) Tahmin ")

        self._build_data_tab()
        self._build_train_tab()
        self._build_plot_tab()
        self._build_pred_tab()
        self.after(200, self._poll_log)

    # ────────────────────────── SEKME 1: VERİ ──────────────────────────
    def _build_data_tab(self):
        f = self.tab_data
        top = ttk.Frame(f); top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Excel Seç...", command=self.pick_file).pack(side="left")
        self.file_var = tk.StringVar(value="(dosya seçilmedi)")
        ttk.Label(top, textvariable=self.file_var).pack(side="left", padx=8)
        ttk.Label(top, text="Sayfa:").pack(side="left", padx=(16, 2))
        self.sheet_cb = ttk.Combobox(top, width=18, state="readonly")
        self.sheet_cb.pack(side="left")
        ttk.Button(top, text="Yükle", command=self.load_data).pack(side="left", padx=8)

        mid = ttk.Frame(f); mid.pack(fill="both", expand=True, padx=8, pady=4)

        # Sol: filtre yöntemleri + hiperparametreleri
        left = ttk.LabelFrame(mid, text="Aykırı Değer Filtreleri")
        left.pack(side="left", fill="y", padx=(0, 8))
        self.use_grubbs = tk.BooleanVar(value=True)
        self.use_iqr = tk.BooleanVar(value=False)
        row = ttk.Frame(left); row.pack(anchor="w", padx=6, pady=4)
        ttk.Checkbutton(row, text="Grubbs testi   alpha =", variable=self.use_grubbs).pack(side="left")
        self.grubbs_alpha = tk.StringVar(value="0.05")
        ttk.Entry(row, width=6, textvariable=self.grubbs_alpha).pack(side="left")
        row = ttk.Frame(left); row.pack(anchor="w", padx=6, pady=4)
        ttk.Checkbutton(row, text="IQR kuralı      k =", variable=self.use_iqr).pack(side="left")
        self.iqr_k = tk.StringVar(value="1.5")
        ttk.Entry(row, width=6, textvariable=self.iqr_k).pack(side="left")
        ttk.Button(left, text="Filtreyi Uygula", command=self.apply_filters).pack(padx=6, pady=8)
        ttk.Button(left, text="Filtreyi Sıfırla (ham veri)", command=self.reset_filter).pack(padx=6, pady=2)

        # Orta: filtrenin uygulanacağı kolonlar (özellik seçer gibi)
        colf = ttk.LabelFrame(mid, text="Filtre Uygulanacak Kolonlar")
        colf.pack(side="left", fill="both", expand=False, padx=(0, 8))
        self.col_list = tk.Listbox(colf, selectmode="multiple", width=38, height=18,
                                   exportselection=False)
        self.col_list.pack(fill="both", expand=True, padx=4, pady=4)

        # Sağ: özet/log
        right = ttk.LabelFrame(mid, text="Veri Özeti / Filtre Raporu")
        right.pack(side="left", fill="both", expand=True)
        self.data_log = tk.Text(right, height=20, wrap="word")
        self.data_log.pack(fill="both", expand=True, padx=4, pady=4)

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="Veri dosyası seç",
            filetypes=[("Excel", "*.xlsx *.xls"), ("Tümü", "*.*")])
        if not path:
            return
        self.file_var.set(path)
        try:
            sheets = pd.ExcelFile(path).sheet_names
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya okunamadı:\n{e}")
            return
        self.sheet_cb["values"] = sheets
        self.sheet_cb.set("Classification" if "Classification" in sheets else sheets[0])

    def load_data(self):
        path = self.file_var.get()
        if not os.path.exists(path):
            messagebox.showwarning("Uyarı", "Önce bir dosya seçin.")
            return
        try:
            self.df_raw = pd.read_excel(path, sheet_name=self.sheet_cb.get())
        except Exception as e:
            messagebox.showerror("Hata", f"Yükleme hatası:\n{e}")
            return
        self.df = self.df_raw.copy()
        self.col_list.delete(0, "end")
        numeric_like = [c for c in self.df_raw.columns
                        if pd.to_numeric(self.df_raw[c], errors="coerce").notna().sum() > 0]
        for c in numeric_like:
            self.col_list.insert("end", c)
            if c in DEFAULT_FILTER_COLS:
                self.col_list.selection_set("end")
        self._refresh_free_plot_columns()
        self._data_report(f"Yüklendi: {len(self.df_raw)} satır, "
                          f"{len(self.df_raw.columns)} kolon.\n"
                          f"Hedef kolon '{TARGET_COL}' "
                          f"{'MEVCUT' if TARGET_COL in self.df_raw.columns else 'YOK!'} "
                          f"(dolu: {pd.to_numeric(self.df_raw.get(TARGET_COL), errors='coerce').notna().sum()})\n"
                          "Filtre kolonlarını seçip 'Filtreyi Uygula'ya basın "
                          "(varsayılan proje kolonları önceden seçili).")

    def apply_filters(self):
        if self.df_raw is None:
            messagebox.showwarning("Uyarı", "Önce veri yükleyin.")
            return
        cols = [self.col_list.get(i) for i in self.col_list.curselection()]
        if not cols:
            messagebox.showwarning("Uyarı", "En az bir kolon seçin.")
            return
        df = self.df_raw.copy()
        report = [f"Başlangıç: {len(df)} satır"]
        if self.use_grubbs.get():
            alpha = float(self.grubbs_alpha.get())
            bad, det = grubbs_outlier_rows(df, cols, alpha=alpha)
            df = df.drop(index=bad)
            report.append(f"Grubbs (alpha={alpha}): {len(bad)} satır elendi")
            for col, n, gc in det:
                if n:
                    report.append(f"   {col}: {n} aykırı (G_crit={gc:.3f})")
        if self.use_iqr.get():
            k = float(self.iqr_k.get())
            bad, det = iqr_outlier_rows(df, cols, k=k)
            df = df.drop(index=bad)
            report.append(f"IQR (k={k}): {len(bad)} satır elendi")
            for col, n, (lo, hi) in det:
                if n:
                    report.append(f"   {col}: {n} aykırı (sınır {lo:.2f}..{hi:.2f})")
        self.df = df.reset_index(drop=True)
        report.append(f"Sonuç: {len(self.df)} satır (eğitimde bu veri kullanılacak)")
        self._data_report("\n".join(report))

    def reset_filter(self):
        if self.df_raw is None:
            return
        self.df = self.df_raw.copy()
        self._data_report(f"Filtre sıfırlandı; ham veri ({len(self.df)} satır) aktif.")

    def _data_report(self, text):
        self.data_log.delete("1.0", "end")
        self.data_log.insert("1.0", text)

    # ────────────────────────── SEKME 2: EĞİTİM ──────────────────────────
    def _build_train_tab(self):
        f = self.tab_train
        cfg = ttk.LabelFrame(f, text="Eğitim Ayarları")
        cfg.pack(fill="x", padx=8, pady=6)

        row = ttk.Frame(cfg); row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text="Senaryo:").pack(side="left")
        self.scen_cb = ttk.Combobox(row, width=26, state="readonly",
                                    values=[s["scenario_name"] for s in SCENARIOS])
        self.scen_cb.set("tum_veri"); self.scen_cb.pack(side="left", padx=4)
        ttk.Label(row, text="fc üssü (özellik):").pack(side="left", padx=(16, 2))
        self.fc_power_cb = ttk.Combobox(row, width=6, state="readonly",
                                        values=["1.0", "0.5", "0.25"])
        self.fc_power_cb.set("1.0"); self.fc_power_cb.pack(side="left")

        row = ttk.Frame(cfg); row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text="Modeller:").pack(side="left")
        self.use_rf = tk.BooleanVar(value=True)
        self.use_xgb = tk.BooleanVar(value=True)
        self.use_svr = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Random Forest", variable=self.use_rf).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="XGBoost", variable=self.use_xgb).pack(side="left", padx=4)
        ttk.Checkbutton(row, text="SVR", variable=self.use_svr).pack(side="left", padx=4)

        row = ttk.Frame(cfg); row.pack(fill="x", padx=6, pady=3)
        srch = "Bayesian" if HAS_SKOPT else "Rastgele (skopt kurulu değil!)"
        ttk.Label(row, text=f"Hiperparametre araması ({srch}) — deneme:").pack(side="left")
        self.n_iter = tk.StringVar(value="25")
        ttk.Entry(row, width=5, textvariable=self.n_iter).pack(side="left", padx=2)
        ttk.Label(row, text="CV katı:").pack(side="left", padx=(12, 2))
        self.cv_folds = tk.StringVar(value="5")
        ttk.Entry(row, width=4, textvariable=self.cv_folds).pack(side="left")
        ttk.Label(row, text="Test oranı:").pack(side="left", padx=(12, 2))
        self.test_size = tk.StringVar(value="0.20")
        ttk.Entry(row, width=5, textvariable=self.test_size).pack(side="left")
        ttk.Label(row, text="Rastgele tohum:").pack(side="left", padx=(12, 2))
        self.rstate = tk.StringVar(value="42")
        ttk.Entry(row, width=5, textvariable=self.rstate).pack(side="left")

        self.train_btn = ttk.Button(cfg, text="EĞİT", command=self.start_training)
        self.train_btn.pack(pady=6)

        # Sonuç tablosu
        cols = ("model", "cv_r2", "test_r2", "test_rmse", "n_train", "n_test")
        heads = ("Model", "CV R² (en iyi)", "Test R²", "Test RMSE (MPa)", "n eğitim", "n test")
        self.res_tree = ttk.Treeview(f, columns=cols, show="headings", height=5)
        for c, h in zip(cols, heads):
            self.res_tree.heading(c, text=h)
            self.res_tree.column(c, width=130, anchor="center")
        self.res_tree.pack(fill="x", padx=8, pady=4)

        logf = ttk.LabelFrame(f, text="Eğitim Günlüğü")
        logf.pack(fill="both", expand=True, padx=8, pady=4)
        self.train_log = tk.Text(logf, height=12, wrap="word")
        self.train_log.pack(fill="both", expand=True, padx=4, pady=4)

    def start_training(self):
        if self.df is None:
            messagebox.showwarning("Uyarı", "Önce 1. sekmede veri yükleyin.")
            return
        if TARGET_COL not in self.df.columns:
            messagebox.showerror("Hata", f"Veride '{TARGET_COL}' kolonu yok.")
            return
        models = [n for n, v in (("Random Forest", self.use_rf),
                                 ("XGBoost", self.use_xgb),
                                 ("SVR", self.use_svr)) if v.get()]
        if not models:
            messagebox.showwarning("Uyarı", "En az bir model seçin.")
            return
        self.train_btn.config(state="disabled")
        self.train_log.delete("1.0", "end")
        for i in self.res_tree.get_children():
            self.res_tree.delete(i)
        cfg = dict(models=models,
                   scenario=self.scen_cb.get(),
                   fc_power=float(self.fc_power_cb.get()),
                   n_iter=int(self.n_iter.get()),
                   cv=int(self.cv_folds.get()),
                   test_size=float(self.test_size.get()),
                   rs=int(self.rstate.get()))
        threading.Thread(target=self._train_worker, args=(cfg,), daemon=True).start()

    def _train_worker(self, cfg):
        try:
            log = self.log_queue.put
            d = filter_scenario(self.df, SCENARIO_MAP[cfg["scenario"]])
            y = pd.to_numeric(d[TARGET_COL], errors="coerce")
            keep = y.notna() & (y >= 0)
            X = d.loc[keep, FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
            X = apply_tr_logic(X)
            X["concrete_strength_mpa"] = np.power(
                X["concrete_strength_mpa"].clip(lower=0), cfg["fc_power"])
            y = y[keep].astype(float)
            log(f"Senaryo '{cfg['scenario']}': {len(y)} numune "
                f"(fc üssü={cfg['fc_power']:g}).")
            if len(y) < 30:
                log("HATA: eğitim için veri çok az (<30). Senaryoyu genişletin.")
                return
            Xtr, Xte, ytr, yte = train_test_split(
                X.values, y.values, test_size=cfg["test_size"],
                random_state=cfg["rs"])
            for name in cfg["models"]:
                log(f"\n── {name}: {cfg['n_iter']} deneme × {cfg['cv']}-kat CV "
                    f"({'Bayesian' if HAS_SKOPT else 'rastgele'}) ...")
                pipe, space = model_space(name, cfg["rs"])
                if HAS_SKOPT:
                    search = BayesSearchCV(pipe, space, n_iter=cfg["n_iter"],
                                           cv=cfg["cv"], scoring="r2",
                                           random_state=cfg["rs"], n_jobs=1,
                                           refit=True)
                else:
                    search = RandomizedSearchCV(pipe, space, n_iter=cfg["n_iter"],
                                                cv=cfg["cv"], scoring="r2",
                                                random_state=cfg["rs"], n_jobs=1,
                                                refit=True)
                search.fit(Xtr, ytr)
                yp = search.predict(Xte)
                r2 = r2_score(yte, yp)
                rmse = float(np.sqrt(mean_squared_error(yte, yp)))
                hist = pd.DataFrame({
                    "iter": np.arange(1, len(search.cv_results_["mean_test_score"]) + 1),
                    "R2": search.cv_results_["mean_test_score"]})
                self.results[name] = dict(
                    estimator=search.best_estimator_, bo_history=hist,
                    cv_r2=float(search.best_score_), test_r2=float(r2),
                    test_rmse=rmse, X_test=Xte, y_test=yte, y_pred=yp,
                    X_train=Xtr, y_train=ytr,
                    best_params={k.replace("model__", ""): v
                                 for k, v in search.best_params_.items()},
                    fc_power=cfg["fc_power"], scenario=cfg["scenario"])
                log(f"   en iyi CV R²={search.best_score_:.3f} | "
                    f"Test R²={r2:.3f}  RMSE={rmse:.2f} MPa")
                log(f"   en iyi parametreler: {self.results[name]['best_params']}")
                self.log_queue.put(("ROW", name, search.best_score_, r2, rmse,
                                    len(ytr), len(yte)))
            log("\n✓ Eğitim bitti — 3. sekmede grafikleri, 4. sekmede tahmini kullanın.")
        except Exception:
            self.log_queue.put("HATA:\n" + traceback.format_exc())
        finally:
            self.log_queue.put(("DONE",))

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "ROW":
                    _, name, cv, r2, rmse, ntr, nte = msg
                    self.res_tree.insert("", "end", values=(
                        name, f"{cv:.3f}", f"{r2:.3f}", f"{rmse:.2f}", ntr, nte))
                    self._refresh_model_boxes()
                elif isinstance(msg, tuple) and msg[0] == "DONE":
                    self.train_btn.config(state="normal")
                else:
                    self.train_log.insert("end", str(msg) + "\n")
                    self.train_log.see("end")
        except queue.Empty:
            pass
        self.after(200, self._poll_log)

    # ────────────────────────── SEKME 3: GRAFİKLER ──────────────────────────
    def _build_plot_tab(self):
        f = self.tab_plot
        ctrl = ttk.LabelFrame(f, text="Grafik Kontrolleri (eksenler canlı değiştirilebilir)")
        ctrl.pack(fill="x", padx=8, pady=4)

        row = ttk.Frame(ctrl); row.pack(fill="x", padx=6, pady=3)
        ttk.Label(row, text="Grafik:").pack(side="left")
        self.plot_kind = ttk.Combobox(row, width=26, state="readonly", values=[
            "Deneysel vs Tahmin (parite)", "BO yakınsaması",
            "Özellik önemi (permütasyon)", "Artıklar (residual)",
            "Serbest saçılım (veri)"])
        self.plot_kind.set("Deneysel vs Tahmin (parite)")
        self.plot_kind.pack(side="left", padx=4)
        self.plot_kind.bind("<<ComboboxSelected>>", lambda e: self._toggle_free_axes())
        ttk.Label(row, text="Model:").pack(side="left", padx=(14, 2))
        self.plot_model = ttk.Combobox(row, width=16, state="readonly")
        self.plot_model.pack(side="left")
        ttk.Button(row, text="ÇİZ", command=self.draw_plot).pack(side="left", padx=12)

        # Serbest saçılım eksen seçimleri
        row2 = ttk.Frame(ctrl); row2.pack(fill="x", padx=6, pady=3)
        ttk.Label(row2, text="Serbest saçılım —  X:").pack(side="left")
        self.free_x = ttk.Combobox(row2, width=26, state="readonly")
        self.free_x.pack(side="left", padx=2)
        ttk.Label(row2, text="Y:").pack(side="left", padx=(8, 2))
        self.free_y = ttk.Combobox(row2, width=26, state="readonly")
        self.free_y.pack(side="left", padx=2)
        ttk.Label(row2, text="Renk:").pack(side="left", padx=(8, 2))
        self.free_c = ttk.Combobox(row2, width=22, state="readonly")
        self.free_c.pack(side="left", padx=2)

        # Eksen limit/ölçek kontrolleri (tüm grafik türlerine uygulanır)
        row3 = ttk.Frame(ctrl); row3.pack(fill="x", padx=6, pady=3)
        self.logx = tk.BooleanVar(); self.logy = tk.BooleanVar()
        ttk.Checkbutton(row3, text="log X", variable=self.logx).pack(side="left")
        ttk.Checkbutton(row3, text="log Y", variable=self.logy).pack(side="left", padx=(6, 14))
        self.xmin = tk.StringVar(); self.xmax = tk.StringVar()
        self.ymin = tk.StringVar(); self.ymax = tk.StringVar()
        for lab, var in (("X min", self.xmin), ("X max", self.xmax),
                         ("Y min", self.ymin), ("Y max", self.ymax)):
            ttk.Label(row3, text=lab + ":").pack(side="left", padx=(6, 1))
            ttk.Entry(row3, width=7, textvariable=var).pack(side="left")
        ttk.Label(row3, text="(boş = otomatik; araç çubuğundan zoom/pan da yapılabilir)"
                  ).pack(side="left", padx=10)

        body = ttk.Frame(f); body.pack(fill="both", expand=True, padx=8, pady=4)
        self.fig = Figure(figsize=(8.6, 5.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, body)

    def _refresh_model_boxes(self):
        names = list(self.results)
        self.plot_model["values"] = names
        if names and not self.plot_model.get():
            self.plot_model.set(names[0])
        self.pred_model["values"] = names
        if names and not self.pred_model.get():
            self.pred_model.set(names[0])

    def _refresh_free_plot_columns(self):
        if self.df_raw is None:
            return
        num = [c for c in self.df_raw.columns
               if pd.to_numeric(self.df_raw[c], errors="coerce").notna().sum() > 0]
        self.free_x["values"] = num
        self.free_y["values"] = num
        self.free_c["values"] = ["(yok)"] + num
        if "concrete_strength_mpa" in num:
            self.free_x.set("concrete_strength_mpa")
        if TARGET_COL in num:
            self.free_y.set(TARGET_COL)
        self.free_c.set("(yok)")

    def _toggle_free_axes(self):
        pass  # yer tutucu: serbest saçılım kontrolleri her zaman görünür (sade tutuldu)

    def _apply_axes(self, ax):
        if self.logx.get():
            ax.set_xscale("log")
        if self.logy.get():
            ax.set_yscale("log")
        def _f(v):
            try:
                return float(v)
            except ValueError:
                return None
        x0, x1 = _f(self.xmin.get()), _f(self.xmax.get())
        y0, y1 = _f(self.ymin.get()), _f(self.ymax.get())
        if x0 is not None or x1 is not None:
            ax.set_xlim(left=x0, right=x1)
        if y0 is not None or y1 is not None:
            ax.set_ylim(bottom=y0, top=y1)

    def draw_plot(self):
        kind = self.plot_kind.get()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        try:
            if kind == "Serbest saçılım (veri)":
                if self.df is None:
                    raise RuntimeError("Önce veri yükleyin.")
                xc, yc = self.free_x.get(), self.free_y.get()
                if not xc or not yc:
                    raise RuntimeError("X ve Y kolonlarını seçin.")
                x = pd.to_numeric(self.df[xc], errors="coerce")
                yv = pd.to_numeric(self.df[yc], errors="coerce")
                cc = self.free_c.get()
                if cc and cc != "(yok)":
                    cv = pd.to_numeric(self.df[cc], errors="coerce")
                    sc = ax.scatter(x, yv, c=cv, s=16, alpha=0.6, cmap="viridis")
                    self.fig.colorbar(sc, ax=ax, label=cc)
                else:
                    ax.scatter(x, yv, s=16, alpha=0.6)
                ax.set_xlabel(xc); ax.set_ylabel(yc)
                ax.set_title(f"Veri saçılımı (n={int((x.notna() & yv.notna()).sum())})")
            else:
                name = self.plot_model.get()
                if name not in self.results:
                    raise RuntimeError("Önce 2. sekmede model eğitin.")
                r = self.results[name]
                if kind == "Deneysel vs Tahmin (parite)":
                    ax.scatter(r["y_test"], r["y_pred"], s=16, alpha=0.55,
                               label=f"test verisi (n={len(r['y_test'])})")
                    lim = [0, max(r["y_test"].max(), r["y_pred"].max()) * 1.05]
                    ax.plot(lim, lim, "r--", lw=1.4, label="1:1")
                    ax.set_xlabel("Deneysel max bond stress (MPa)")
                    ax.set_ylabel("Tahmin (MPa)")
                    ax.set_title(f"{name} — {r['scenario']} | Test R²={r['test_r2']:.3f} "
                                 f"RMSE={r['test_rmse']:.2f} MPa")
                    ax.legend(fontsize=9)
                elif kind == "BO yakınsaması":
                    h = r["bo_history"]
                    ax.plot(h["iter"], h["R2"], "o", ms=4, alpha=0.5, label="deneme CV R²")
                    ax.plot(h["iter"], h["R2"].cummax(), "-", lw=2, color="crimson",
                            label="kümülatif en iyi")
                    ax.set_xlabel("Deneme sırası"); ax.set_ylabel("CV R²")
                    ax.set_title(f"{name} — hiperparametre araması yakınsaması")
                    ax.legend(fontsize=9)
                elif kind == "Özellik önemi (permütasyon)":
                    imp = permutation_importance(
                        r["estimator"], r["X_test"], r["y_test"],
                        n_repeats=10, random_state=42, scoring="r2")
                    order = np.argsort(imp.importances_mean)
                    ax.barh(np.array(FEATURE_COLS)[order],
                            imp.importances_mean[order],
                            xerr=imp.importances_std[order], color="steelblue")
                    ax.set_xlabel("Permütasyon önemi (R² düşüşü)")
                    ax.set_title(f"{name} — özellik önemi (test setinde)")
                elif kind == "Artıklar (residual)":
                    res = r["y_test"] - r["y_pred"]
                    ax.scatter(r["y_pred"], res, s=16, alpha=0.55)
                    ax.axhline(0, color="r", lw=1.2, ls="--")
                    ax.set_xlabel("Tahmin (MPa)"); ax.set_ylabel("Artık = deneysel − tahmin (MPa)")
                    ax.set_title(f"{name} — artık dağılımı "
                                 f"(ort={res.mean():.2f}, std={res.std():.2f} MPa)")
            self._apply_axes(ax)
            ax.grid(alpha=0.3)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Grafik hatası", str(e))

    # ────────────────────────── SEKME 4: TAHMİN ──────────────────────────
    def _build_pred_tab(self):
        f = self.tab_pred
        top = ttk.Frame(f); top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Model:").pack(side="left")
        self.pred_model = ttk.Combobox(top, width=18, state="readonly")
        self.pred_model.pack(side="left", padx=6)
        ttk.Button(top, text="TAHMİN ET", command=self.predict_one).pack(side="left", padx=10)
        self.pred_out = tk.StringVar(value="—")
        ttk.Label(top, textvariable=self.pred_out,
                  font=("Segoe UI", 12, "bold")).pack(side="left", padx=16)

        grid = ttk.LabelFrame(f, text="Numune Özellikleri (boş bırakılan = ölçülmemiş; "
                                      "model eksikliği kendisi işler)")
        grid.pack(fill="x", padx=8, pady=6)
        self.pred_entries = {}
        labels = {
            "bar_diameter_mm": "Donatı çapı (mm)",
            "nominal_yield_strength_mpa": "Akma dayanımı (MPa)",
            "concrete_strength_mpa": "Beton dayanımı fc (MPa)",
            "embedment_length_db": "Gömme boyu (db katı)",
            "concrete_cover_db": "Paspayı (db katı)",
            "transverse_reinforcement": "Enine donatı var mı? (1/0)",
            "transverse_reinforcement_mm": "Etriye aralığı (mm)",
            "transverse_reinforcement_bar_diameter_mm": "Etriye çapı (mm)",
        }
        for i, col in enumerate(FEATURE_COLS):
            r, c = divmod(i, 2)
            ttk.Label(grid, text=labels[col] + ":").grid(row=r, column=2 * c,
                                                         sticky="e", padx=6, pady=4)
            var = tk.StringVar()
            ttk.Entry(grid, width=12, textvariable=var).grid(row=r, column=2 * c + 1,
                                                             sticky="w", padx=4)
            self.pred_entries[col] = var

        info = ttk.LabelFrame(f, text="Not")
        info.pack(fill="both", expand=True, padx=8, pady=6)
        txt = tk.Text(info, height=8, wrap="word")
        txt.pack(fill="both", expand=True, padx=4, pady=4)
        txt.insert("1.0",
                   "• Tahmin, 2. sekmede EN SON eğitilen modelle yapılır; eğitimde seçilen "
                   "senaryo ve fc üssü otomatik uygulanır.\n"
                   "• Enine donatı 0/boş girilirse etriye aralığı/çapı otomatik yok sayılır "
                   "(proje TR mantığı).\n"
                   "• Eksik bıraktığınız her özellik 'ölçülmemiş' olarak modele bildirilir; "
                   "medyanla doldurulmaz.\n"
                   "• Sonuç, maksimum aderans dayanımı tau_max (MPa) tahminidir.")
        txt.config(state="disabled")

    def predict_one(self):
        name = self.pred_model.get()
        if name not in self.results:
            messagebox.showwarning("Uyarı", "Önce 2. sekmede model eğitin.")
            return
        r = self.results[name]
        row = {}
        for col, var in self.pred_entries.items():
            v = var.get().strip().replace(",", ".")
            row[col] = float(v) if v else np.nan
        X = pd.DataFrame([row], columns=FEATURE_COLS)
        X = apply_tr_logic(X)
        X["concrete_strength_mpa"] = np.power(
            X["concrete_strength_mpa"].clip(lower=0), r["fc_power"])
        try:
            pred = float(r["estimator"].predict(X.values)[0])
        except Exception as e:
            messagebox.showerror("Hata", f"Tahmin hatası:\n{e}")
            return
        self.pred_out.set(f"tau_max ≈ {pred:.2f} MPa   "
                          f"({name}, senaryo: {r['scenario']}, Test R²={r['test_r2']:.2f})")


if __name__ == "__main__":
    app = BondStressApp()
    app.mainloop()
