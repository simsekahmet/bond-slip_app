# Bond Stress Prediction Tool — τ_max (Web)

Maksimum aderans dayanımı (τ_max) tahmini için tarayıcıda çalışan uçtan uca araç.
Tüm hesaplama **tarayıcıda** yapılır (Pyodide / WebAssembly Python) — sunucu yoktur,
veriniz hiçbir yere gönderilmez.

**Canlı uygulama:** https://simsekahmet.github.io/bond-slip_app/

## Özellikler

1. **Veri** — kendi CSV/Excel dosyanı yükle (sürükle-bırak desteklenir), sayfa seçimi
2. **Aykırı değer filtresi** — Grubbs testi / IQR kuralı, kolon seçimli
3. **Eğitim** — Random Forest / XGBoost / SVR + Bayesian hiperparametre araması
   (scikit-optimize), senaryo filtresi, fc üssü seçeneği
4. **Grafikler** — parite, BO yakınsaması, 3B BO gezinme yolu, permütasyon özellik
   önemi, SHAP özet/önem, artıklar, serbest saçılım, 3B parametre yüzeyi;
   min/maks, zoom/pan (Plotly). Her grafiğin çizildiği eksen verileri **Excel/CSV**
   olarak, görseli **PDF** olarak indirilebilir.
5. **Tahmin** — eğitilen modelle tek numune tahmini; boş bırakılan özellik
   "ölçülmemiş" olarak işlenir (MissingAwareScaler), TR mantığı otomatik uygulanır

Eğitim/veri mantığı, tez deposundaki `02_max_bond_stress_pred.ipynb` defteri ve
eski masaüstü uygulamasıyla (bkz. `legacy/`) birebir aynıdır.

## Yerelde çalıştırma

Web worker `file://` altında çalışmaz; basit bir HTTP sunucu gerekir:

```bash
python -m http.server 8000
# http://localhost:8000
```

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `index.html` | Uygulamanın tamamı (arayüz + Pyodide çekirdeği, tek dosya) |
| `legacy/` | Eski tkinter masaüstü sürümü (referans) |

Not: Uygulama veri içermez — kullanıcı kendi CSV/Excel dosyasını yükler ve tüm
işlem tarayıcıda kalır.

## Not

İlk açılışta Pyodide ve bilimsel paketler (~30 MB) CDN'den indirilir; 15–40 sn
sürebilir. Sonraki açılışlar tarayıcı önbelleği sayesinde hızlıdır.

---
Ahmet Şimşek — MSc Tez, aderans gerilmesi–sıyrılma davranışının tahmini
