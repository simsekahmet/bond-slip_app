# Bond Stress Studio — τ_max Tahmin Aracı (Web)

Maksimum aderans dayanımı (τ_max) tahmini için tarayıcıda çalışan uçtan uca araç.
Tüm hesaplama **tarayıcıda** yapılır (Pyodide / WebAssembly Python) — sunucu yoktur,
veriniz hiçbir yere gönderilmez.

**Canlı uygulama:** https://simsekahmet.github.io/bond-slip_app/

## Özellikler

1. **Veri** — Excel yükle (varsayılan: `filtered_data.xlsx`), sayfa seçimi
2. **Aykırı değer filtresi** — Grubbs testi / IQR kuralı, kolon seçimli
3. **Eğitim** — Random Forest / XGBoost / SVR + Bayesian hiperparametre araması
   (scikit-optimize), senaryo filtresi, fc üssü seçeneği
4. **Grafikler** — parite, BO yakınsaması, permütasyon özellik önemi, artıklar,
   serbest saçılım; log eksen, min/maks, zoom/pan (Plotly)
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
| `filtered_data.xlsx` | Varsayılan veri seti (Grubbs-filtreli, 1120 satır) |
| `legacy/` | Eski tkinter masaüstü sürümü (referans) |

## Not

İlk açılışta Pyodide ve bilimsel paketler (~30 MB) CDN'den indirilir; 15–40 sn
sürebilir. Sonraki açılışlar tarayıcı önbelleği sayesinde hızlıdır.

---
Ahmet Şimşek — MSc Tez, aderans gerilmesi–sıyrılma davranışının tahmini
