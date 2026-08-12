import streamlit as st
from datetime import datetime, timedelta
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

try:
    import holidays as holidays_lib
    HOLIDAYS_MEVCUT = True
except ImportError:
    HOLIDAYS_MEVCUT = False

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_MEVCUT = True
except ImportError:
    SKLEARN_MEVCUT = False

# =========================================================================
# CONFIG & BAŞLIK
# =========================================================================
st.set_page_config(page_title="Turkish Technic Gelişmiş Optimizasyon", layout="wide")
st.title("✈️ Gelişmiş Uçak Bakım Slot Optimizasyon Sistemi")
st.write("Turkish Technic operasyonel kurallarına göre dinamik planlama paneli.")

BAKIM_TIPLERI = ["A-Check (Hafif)", "C-Check (Ağır)", "D-Check (En Ağır)"]
BAKIM_KISA = {"A-Check (Hafif)": "A-Check", "C-Check (Ağır)": "C-Check", "D-Check (En Ağır)": "D-Check"}
BAKIM_VARSAYILAN = {"A-Check": (2, 15), "C-Check": (5, 30), "D-Check": (8, 45)}

# =========================================================================
# YARDIMCI FONKSİYON: BASİT DOĞRUSAL REGRESYON (sklearn yoksa)
# =========================================================================
def manuel_dogrusal_regresyon(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    X_b = np.column_stack([np.ones(len(X)), X])
    katsayilar, *_ = np.linalg.lstsq(X_b, y, rcond=None)
    return katsayilar  # [sabit, egim...]


def sure_tahmin_modeli_egit():
    """
    NOT: Gerçek işletme verisi (geçmiş bakım kayıtları) bağlandığında bu sentetik
    veri seti gerçek CSV/veritabanı sorgusu ile değiştirilmelidir. Burada, uçak yaşı
    arttıkça bakım süresinin ortalama olarak nasıl uzadığını gösteren temsili bir
    ilişki öğretmek için sentetik bir veri seti üretiyoruz.
    """
    rng = np.random.default_rng(42)
    kayitlar = []
    for bakim, (taban_sure, _) in BAKIM_VARSAYILAN.items():
        for _ in range(60):
            yas = rng.uniform(0, 25)
            gurultu = rng.normal(0, 0.4)
            sure = taban_sure + 0.06 * yas + gurultu
            kayitlar.append((bakim, yas, max(taban_sure * 0.6, sure)))
    df_egitim = pd.DataFrame(kayitlar, columns=["bakim_tipi", "yas", "sure"])

    modeller = {}
    for bakim in BAKIM_VARSAYILAN:
        alt = df_egitim[df_egitim["bakim_tipi"] == bakim]
        if SKLEARN_MEVCUT:
            model = LinearRegression().fit(alt[["yas"]], alt["sure"])
            modeller[bakim] = ("sklearn", model)
        else:
            katsayilar = manuel_dogrusal_regresyon(alt[["yas"]].values, alt["sure"].values)
            modeller[bakim] = ("manuel", katsayilar)
    return modeller


def tahmini_sure_hesapla(modeller, bakim_kisa, yas):
    yontem, model = modeller[bakim_kisa]
    if yontem == "sklearn":
        tahmin = model.predict(pd.DataFrame({"yas": [yas]}))[0]
    else:
        sabit, egim = model
        tahmin = sabit + egim * yas
    return max(1, round(tahmin))


MODELLER = sure_tahmin_modeli_egit()

# =========================================================================
# YARDIMCI FONKSİYON: TATİL GÜNLERİ
# =========================================================================
def tatil_indekslerini_getir(baslangic_tarihi, ufuk):
    indeksler = []
    if HOLIDAYS_MEVCUT:
        try:
            tr_tatiller = holidays_lib.Turkey(years=[baslangic_tarihi.year, baslangic_tarihi.year + 1])
            for d in range(ufuk + 1):
                gun = (baslangic_tarihi + timedelta(days=d)).date()
                if gun in tr_tatiller:
                    indeksler.append(d)
        except Exception:
            pass
    return indeksler


# =========================================================================
# SOL PANEL: GENEL PARAMETRELER
# =========================================================================
st.sidebar.header("⚙️ 1. Genel Planlama Ayarları")
TOPLAM_SLOT = st.sidebar.number_input("Toplam Hangar Slot Sayısı", min_value=1, max_value=10, value=2)
GUNLUK_MAKS_ADAM_SAAT = st.sidebar.number_input("Günlük Maksimum Adam/Saat", min_value=10, max_value=3000, value=70)
PLANLAMA_UFUKU = st.sidebar.number_input("Zaman Ufku (Gün)", min_value=5, max_value=400, value=20)
HAFTA_SONU_YASAGI = st.sidebar.checkbox("Bakım hafta sonu başlayamaz / bitemez", value=True)
RESMI_TATIL_YASAGI = st.sidebar.checkbox(
    "Resmi tatillerde başlayamaz / bitemez",
    value=True,
    help="'holidays' paketi kurulu değilse bu seçenek etkisizdir (requirements.txt içine 'holidays' ekleyin)."
)
if RESMI_TATIL_YASAGI and not HOLIDAYS_MEVCUT:
    st.sidebar.warning("'holidays' paketi bulunamadı. `pip install holidays` ile kurup yeniden başlatın.")

ML_SURE_TAHMINI = st.sidebar.checkbox(
    "Uçak yaşına göre bakım süresini otomatik tahmin et (ML)",
    value=True,
    help="Sentetik/temsili veriyle eğitilmiş basit bir regresyon modeliyle süreleri uçak yaşına göre ayarlar."
)

st.sidebar.markdown("---")
st.sidebar.header("👷 2. Ekip Yönetimi")
if "ekip_listesi" not in st.session_state:
    st.session_state.ekip_listesi = [
        {"ad": "Ekip-1", "yetkinlikler": ["A-Check", "C-Check"]},
        {"ad": "Ekip-2", "yetkinlikler": ["C-Check", "D-Check"]},
        {"ad": "Ekip-3", "yetkinlikler": ["A-Check", "D-Check"]},
    ]
with st.sidebar.expander("➕ Yeni Ekip Tanımla", expanded=False):
    e_ad = st.text_input("Ekip Adı", "Ekip-4", key="yeni_ekip_ad")
    e_yetkinlik = st.multiselect("Yetkinlikler (Bakım Türleri)", list(BAKIM_VARSAYILAN.keys()), default=["A-Check"])
    if st.button("Ekip Ekle"):
        st.session_state.ekip_listesi.append({"ad": e_ad, "yetkinlikler": e_yetkinlik})
        st.success(f"{e_ad} eklendi.")
for idx, ek in enumerate(st.session_state.ekip_listesi):
    st.sidebar.caption(f"• {ek['ad']}: {', '.join(ek['yetkinlikler']) if ek['yetkinlikler'] else 'yetkinlik yok'}")

st.sidebar.markdown("---")
st.sidebar.header("🧰 3. Ekipman Yönetimi")
if "ekipman_listesi" not in st.session_state:
    st.session_state.ekipman_listesi = {"Ağır Bakım Ekipmanı": 1, "NDT Test Cihazı": 1}
with st.sidebar.expander("➕ Yeni Ekipman Tanımla", expanded=False):
    ekp_ad = st.text_input("Ekipman Adı", "Yeni Ekipman", key="yeni_ekipman_ad")
    ekp_kap = st.number_input("Eş Zamanlı Kapasite", min_value=1, value=1, key="yeni_ekipman_kap")
    if st.button("Ekipman Ekle"):
        st.session_state.ekipman_listesi[ekp_ad] = ekp_kap
        st.success(f"{ekp_ad} eklendi.")
for ad, kap in st.session_state.ekipman_listesi.items():
    st.sidebar.caption(f"• {ad}: {kap} eş zamanlı")

st.sidebar.markdown("---")
st.sidebar.header("🛩️ 4. Filo Yönetimi (Uçak Ekleme)")

if "ucak_listesi" not in st.session_state:
    st.session_state.ucak_listesi = [
        {"ad": "TC-JPE", "model_tipi": "Dar Gövde (A320/B737)", "bakim_tipi": "C-Check", "teslim_hedefi": 5,
         "ceza": 1000, "parca_gun": 0, "oncelik": "3 - Normal", "imalat_yili": 2015,
         "gerekli_ekipman": []},
        {"ad": "TC-JJJ", "model_tipi": "Geniş Gövde (A330/B787)", "bakim_tipi": "D-Check", "teslim_hedefi": 8,
         "ceza": 2000, "parca_gun": 2, "oncelik": "5 - AOG (Kritik)", "imalat_yili": 2008,
         "gerekli_ekipman": ["Ağır Bakım Ekipmanı"]},
    ]

with st.sidebar.expander("➕ Yeni Uçak Tanımla", expanded=False):
    y_ad = st.text_input("Kuyruk Tescili / Adı", "TC-XYZ")
    y_model = st.selectbox("Gövde Tipi", ["Dar Gövde (A320/B737)", "Geniş Gövde (A330/B787)"])
    y_bakim = st.selectbox("Bakım Türü", BAKIM_TIPLERI)
    y_hedef = st.number_input("Hedef Teslim (Gün)", min_value=1, value=6)
    y_ceza = st.number_input("Gecikme Cezası ($/Gün)", min_value=0, value=1200)
    y_parca = st.number_input("Parça Bekleme (Gün)", min_value=0, value=0)
    y_yil = st.number_input("İmalat Yılı", min_value=1970, max_value=datetime.now().year, value=2015)
    y_oncelik = st.selectbox("Öncelik Derecesi", ["1 - Düşük", "2 - Düşük-Orta", "3 - Normal", "4 - Yüksek", "5 - AOG (Kritik)"])
    y_ekipman = st.multiselect("Gerekli Özel Ekipman", list(st.session_state.ekipman_listesi.keys()))

    if st.button("Filoya Ekle"):
        if not y_ad.strip():
            st.error("Uçak adı boş olamaz.")
        elif y_ad in [u["ad"] for u in st.session_state.ucak_listesi]:
            st.error(f"'{y_ad}' tescili zaten filoda mevcut. Farklı bir tescil kullanın.")
        else:
            st.session_state.ucak_listesi.append({
                "ad": y_ad.strip(), "model_tipi": y_model, "bakim_tipi": y_bakim,
                "teslim_hedefi": y_hedef, "ceza": y_ceza, "parca_gun": y_parca,
                "oncelik": y_oncelik, "imalat_yili": y_yil, "gerekli_ekipman": y_ekipman
            })
            st.success(f"{y_ad} başarıyla filoya eklendi!")

st.sidebar.write(f"**Filodaki Güncel Uçak Sayısı:** {len(st.session_state.ucak_listesi)}")
guncel_ucaklar = []
dogrulama_hatalari = []

for idx, uc in enumerate(st.session_state.ucak_listesi):
    with st.sidebar.expander(f"⚙️ {uc['ad']} ({uc['bakim_tipi']})"):
        u_ad = st.text_input("Ad", uc["ad"], key=f"ad_{idx}")
        u_model = st.selectbox("Gövde", ["Dar Gövde (A320/B737)", "Geniş Gövde (A330/B787)"],
                                index=0 if "Dar" in uc["model_tipi"] else 1, key=f"mod_{idx}")
        u_bakim = st.selectbox("Tür", BAKIM_TIPLERI, index=BAKIM_TIPLERI.index(uc["bakim_tipi"]) if uc["bakim_tipi"] in BAKIM_TIPLERI else 0, key=f"bak_{idx}")
        u_hedef = st.number_input("Hedef (Gün)", min_value=1, value=uc["teslim_hedefi"], key=f"hdf_{idx}")
        u_ceza = st.number_input("Ceza ($)", min_value=0, value=uc["ceza"], key=f"cz_{idx}")
        u_parca = st.number_input("Parça Günü", min_value=0, value=uc["parca_gun"], key=f"prc_{idx}")
        u_yil = st.number_input("İmalat Yılı", min_value=1970, max_value=datetime.now().year,
                                 value=uc.get("imalat_yili", 2015), key=f"yil_{idx}")
        u_oncelik = st.selectbox("Öncelik", ["1 - Düşük", "2 - Düşük-Orta", "3 - Normal", "4 - Yüksek", "5 - AOG (Kritik)"],
                                  index=int(uc["oncelik"][0]) - 1, key=f"onc_{idx}")
        u_ekipman = st.multiselect("Gerekli Özel Ekipman", list(st.session_state.ekipman_listesi.keys()),
                                    default=uc.get("gerekli_ekipman", []), key=f"ekp_{idx}")

        if st.button("🗑️ Bu Uçağı Filodan Çıkar", key=f"sil_{idx}"):
            st.session_state.ucak_listesi.pop(idx)
            st.rerun()

        bakim_kisa = BAKIM_KISA[u_bakim]
        taban_sure, adam = BAKIM_VARSAYILAN[bakim_kisa]
        yas = max(0, datetime.now().year - u_yil)
        if ML_SURE_TAHMINI:
            sure = tahmini_sure_hesapla(MODELLER, bakim_kisa, yas)
        else:
            sure = taban_sure

        if not u_ad.strip():
            dogrulama_hatalari.append(f"{idx + 1}. sıradaki uçağın adı boş olamaz.")
        if u_hedef < sure:
            st.info(f"ℹ️ Hedef teslim ({u_hedef} gün) tahmini bakım süresinden ({sure} gün) kısa; gecikme kaçınılmaz olabilir.")

        guncel_ucaklar.append({
            "ad": u_ad.strip(), "model_tipi": u_model, "bakim_tipi": bakim_kisa, "sure": sure, "adam_saat": adam,
            "teslim_hedefi": u_hedef, "ceza": u_ceza, "parca_gun": u_parca, "oncelik": int(u_oncelik[0]),
            "imalat_yili": u_yil, "yas": yas, "gerekli_ekipman": u_ekipman
        })

isimler = [u["ad"] for u in guncel_ucaklar]
if len(isimler) != len(set(isimler)):
    dogrulama_hatalari.append("Filoda birbirinin aynısı iki tescil adı var. Lütfen tescilleri benzersiz yapın.")

if st.sidebar.button("🗑️ Tüm Filoyu Sıfırla"):
    st.session_state.ucak_listesi = []
    st.rerun()

if dogrulama_hatalari:
    for hata in dogrulama_hatalari:
        st.sidebar.error(hata)

baslat_butonu = st.sidebar.button("🚀 Planlamayı Başlat", type="primary", disabled=bool(dogrulama_hatalari) or len(guncel_ucaklar) == 0)

st.sidebar.markdown("---")
senaryo_karsilastir = st.sidebar.checkbox("🔀 Senaryo Karşılaştırma Modu (Slot +1)", value=False,
                                           help="Mevcut ayarlarla birlikte, hangar slot sayısı bir fazla olan alternatif bir senaryoyu da çözüp karşılaştırır.")

# =========================================================================
# OPERASYONEL OPTİMİZASYON MOTORU
# =========================================================================
def gelismis_optimizasyon(ucaklar, ekipler, ekipmanlar, toplam_slot, gunluk_maks_adam_saat,
                           planlama_ufuku, hafta_sonu_yasagi, tatil_indeksleri):
    model = cp_model.CpModel()
    n = len(ucaklar)
    baslangic, bitis, aralik, gecikmeler, slot_atama, ekip_atama = {}, {}, {}, {}, {}, {}
    bugun = datetime.now()

    kapali_gunler = set()
    for d in range(planlama_ufuku + 1):
        gelecek_tarih = bugun + timedelta(days=d)
        if hafta_sonu_yasagi and gelecek_tarih.weekday() in (5, 6):
            kapali_gunler.add(d)
    kapali_gunler |= set(tatil_indeksleri)

    for i, ucak in enumerate(ucaklar):
        sure = ucak["sure"]
        baslangic[i] = model.NewIntVar(0, max(0, planlama_ufuku - sure), f"bas_{i}")
        bitis[i] = model.NewIntVar(0, planlama_ufuku, f"bit_{i}")
        aralik[i] = model.NewIntervalVar(baslangic[i], sure, bitis[i], f"aralik_{i}")

        if "Geniş Gövde" in ucak["model_tipi"] and toplam_slot >= 2:
            slot_atama[i] = model.NewIntVar(2, toplam_slot, f"slot_{i}")
        else:
            slot_atama[i] = model.NewIntVar(1, toplam_slot, f"slot_{i}")

        yetkin_ekip_idx = [k for k, ek in enumerate(ekipler) if ucak["bakim_tipi"] in ek["yetkinlikler"]]
        if not yetkin_ekip_idx:
            yetkin_ekip_idx = list(range(len(ekipler)))
        ekip_atama[i] = model.NewIntVarFromDomain(cp_model.Domain.FromValues(yetkin_ekip_idx), f"ekip_{i}")

        gecikmeler[i] = model.NewIntVar(0, planlama_ufuku, f"gecikme_{i}")
        model.AddMaxEquality(gecikmeler[i], [0, bitis[i] - ucak["teslim_hedefi"]])

        for h in kapali_gunler:
            model.Add(baslangic[i] != h)
            model.Add(bitis[i] != h)

        if ucak["parca_gun"] > 0:
            model.Add(baslangic[i] >= ucak["parca_gun"])

    model.AddCumulative([aralik[i] for i in range(n)], [1] * n, toplam_slot)
    model.AddCumulative([aralik[i] for i in range(n)], [u["adam_saat"] for u in ucaklar], gunluk_maks_adam_saat)

    for ekipman_adi, kapasite in ekipmanlar.items():
        gerekli_araliklar, talepler = [], []
        for i, ucak in enumerate(ucaklar):
            if ekipman_adi in ucak.get("gerekli_ekipman", []):
                gerekli_araliklar.append(aralik[i])
                talepler.append(1)
        if gerekli_araliklar:
            model.AddCumulative(gerekli_araliklar, talepler, kapasite)

    for i in range(n):
        for j in range(i + 1, n):
            ayni_slot = model.NewBoolVar(f"ayni_slot_{i}_{j}")
            model.Add(slot_atama[i] == slot_atama[j]).OnlyEnforceIf(ayni_slot)
            model.Add(slot_atama[i] != slot_atama[j]).OnlyEnforceIf(ayni_slot.Not())
            i_once = model.NewBoolVar(f"i_once_{i}_{j}")
            j_once = model.NewBoolVar(f"j_once_{i}_{j}")
            model.Add(bitis[i] <= baslangic[j]).OnlyEnforceIf([ayni_slot, i_once])
            model.Add(bitis[j] <= baslangic[i]).OnlyEnforceIf([ayni_slot, j_once])
            model.AddBoolOr([i_once, j_once]).OnlyEnforceIf(ayni_slot)

    for i in range(n):
        for j in range(i + 1, n):
            ayni_ekip = model.NewBoolVar(f"ayni_ekip_{i}_{j}")
            model.Add(ekip_atama[i] == ekip_atama[j]).OnlyEnforceIf(ayni_ekip)
            model.Add(ekip_atama[i] != ekip_atama[j]).OnlyEnforceIf(ayni_ekip.Not())
            i_once2 = model.NewBoolVar(f"i_once2_{i}_{j}")
            j_once2 = model.NewBoolVar(f"j_once2_{i}_{j}")
            model.Add(bitis[i] <= baslangic[j]).OnlyEnforceIf([ayni_ekip, i_once2])
            model.Add(bitis[j] <= baslangic[i]).OnlyEnforceIf([ayni_ekip, j_once2])
            model.AddBoolOr([i_once2, j_once2]).OnlyEnforceIf(ayni_ekip)

    toplam_ceza = sum(gecikmeler[i] * ucaklar[i]["ceza"] * ucaklar[i]["oncelik"] for i in range(n))
    toplam_erken_baslama = sum(baslangic[i] for i in range(n))
    model.Minimize(toplam_ceza * 100 + toplam_erken_baslama)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = min(30.0, 5.0 + 2.0 * n)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    return solver, status, baslangic, bitis, gecikmeler, slot_atama, ekip_atama


def esneklik_ile_coz(ucaklar, ekipler, ekipmanlar, toplam_slot, gunluk_maks_adam_saat,
                      planlama_ufuku, hafta_sonu_yasagi, tatil_indeksleri):
    """
    İlk çözüm bulunamazsa, sırasıyla slot sayısını, adam/saat kapasitesini ve zaman
    ufkunu esneterek çözülebilirlik arayan bir esneklik analizi yapar. Hangi
    gevşetmenin işe yaradığını kullanıcıya raporlar.
    """
    denemeler = [
        ("Mevcut Parametreler", toplam_slot, gunluk_maks_adam_saat, planlama_ufuku),
        ("Hangar Slot +1", toplam_slot + 1, gunluk_maks_adam_saat, planlama_ufuku),
        ("Adam/Saat Kapasitesi +%25", toplam_slot, int(gunluk_maks_adam_saat * 1.25), planlama_ufuku),
        ("Zaman Ufku +10 Gün", toplam_slot, gunluk_maks_adam_saat, planlama_ufuku + 10),
        ("Slot +1 ve Adam/Saat +%25", toplam_slot + 1, int(gunluk_maks_adam_saat * 1.25), planlama_ufuku),
    ]
    for etiket, ts, gm, pu in denemeler:
        sonuc = gelismis_optimizasyon(ucaklar, ekipler, ekipmanlar, ts, gm, pu, hafta_sonu_yasagi, tatil_indeksleri)
        status = sonuc[1]
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return sonuc, etiket, (ts, gm, pu)
    return sonuc, None, (toplam_slot, gunluk_maks_adam_saat, planlama_ufuku)


def sonuclari_dataframe_yap(guncel_ucaklar, ekipler, solver, baslangic, bitis, gecikmeler, slot_atama, ekip_atama):
    bugun = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    veri = []
    for i, uc in enumerate(guncel_ucaklar):
        b_gun = solver.Value(baslangic[i])
        bt_gun = solver.Value(bitis[i])
        g_gun = solver.Value(gecikmeler[i])
        veri.append({
            "Uçak Tescil": uc["ad"],
            "Gövde Tipi": uc["model_tipi"],
            "Bakım Türü": uc["bakim_tipi"],
            "Yaş (Yıl)": uc["yas"],
            "Atanan Slot": f"Hangar Slot {solver.Value(slot_atama[i])}",
            "Atanan Ekip": ekipler[solver.Value(ekip_atama[i])]["ad"],
            "Planlanan Başlangıç": (bugun + timedelta(days=b_gun)).strftime("%Y-%m-%d %H:%M"),
            "Planlanan Bitiş": (bugun + timedelta(days=bt_gun)).strftime("%Y-%m-%d %H:%M"),
            "Süre (Gün)": uc["sure"],
            "Adam/Saat": uc["adam_saat"],
            "Gecikme (Gün)": g_gun,
            "Öncelik Skoru": uc["oncelik"],
            "Durum": "🔴 Gecikmeli" if g_gun > 0 else "🟢 Zamanında"
        })
    return pd.DataFrame(veri)


def excel_indirme_arabellegi(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Bakım_Plani', index=False)
    buffer.seek(0)
    return buffer


# =========================================================================
# DASHBOARD / ÇIKTI SİSTEMİ
# =========================================================================
if baslat_butonu and len(guncel_ucaklar) > 0:
    tatil_indeksleri = tatil_indekslerini_getir(datetime.now(), PLANLAMA_UFUKU) if RESMI_TATIL_YASAGI else []

    (solver, status, baslangic, bitis, gecikmeler, slot_atama, ekip_atama), esneklik_etiketi, kullanilan_parametreler = \
        esneklik_ile_coz(guncel_ucaklar, st.session_state.ekip_listesi, st.session_state.ekipman_listesi,
                          TOPLAM_SLOT, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU, HAFTA_SONU_YASAGI, tatil_indeksleri)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        st.error(
            "❌ Belirtilen operasyonel kurallar ve kapasite sınırları dahilinde ÇÖZÜM BULUNAMADI! "
            "Hangar slotunu, günlük adam/saat sınırını veya zaman ufkunu esnetmeyi deneyin; "
            "otomatik esneklik denemeleri de sonuç veremedi."
        )
    else:
        if esneklik_etiketi and esneklik_etiketi != "Mevcut Parametreler":
            st.warning(
                f"⚠️ Girilen parametrelerle doğrudan çözüm bulunamadı. Sistem otomatik olarak "
                f"**'{esneklik_etiketi}'** senaryosunu deneyip çözüm buldu "
                f"(Slot={kullanilan_parametreler[0]}, Adam/Saat={kullanilan_parametreler[1]}, Ufuk={kullanilan_parametreler[2]} gün). "
                f"Aşağıdaki sonuçlar bu gevşetilmiş parametrelerle üretilmiştir."
            )

        df = sonuclari_dataframe_yap(guncel_ucaklar, st.session_state.ekip_listesi, solver, baslangic, bitis,
                                      gecikmeler, slot_atama, ekip_atama)

        # ---- KPI Kartları ----
        ortalama_gecikme = df["Gecikme (Gün)"].mean()
        zamaninda_oran = (df["Durum"] == "🟢 Zamanında").mean() * 100
        toplam_adam_saat_talep = sum(u["adam_saat"] * u["sure"] for u in guncel_ucaklar)
        maks_adam_saat_kapasite = kullanilan_parametreler[1] * PLANLAMA_UFUKU
        kapasite_kullanim_orani = min(100, (toplam_adam_saat_talep / maks_adam_saat_kapasite) * 100) if maks_adam_saat_kapasite else 0
        slot_gun_kapasitesi = kullanilan_parametreler[0] * PLANLAMA_UFUKU
        slot_gun_talep = sum(u["sure"] for u in guncel_ucaklar)
        slot_doluluk_orani = min(100, (slot_gun_talep / slot_gun_kapasitesi) * 100) if slot_gun_kapasitesi else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Ağırlıklı Toplam Ceza", f"{solver.Value(sum(gecikmeler[i] * guncel_ucaklar[i]['ceza'] * guncel_ucaklar[i]['oncelik'] for i in range(len(guncel_ucaklar)))):,.0f} $")
        with c2:
            st.metric("Planlanan Uçak", f"{len(guncel_ucaklar)}")
        with c3:
            st.metric("Ortalama Gecikme", f"{ortalama_gecikme:.1f} gün")
        with c4:
            st.metric("Zamanında Teslim Oranı", f"{zamaninda_oran:.0f}%")
        with c5:
            st.metric("Adam/Saat Kapasite Kullanımı", f"{kapasite_kullanim_orani:.0f}%")

        st.caption(f"Hangar slot doluluk oranı (slot-gün bazında): **{slot_doluluk_orani:.0f}%** • "
                   f"Yazılım Durumu: **{'Optimal Çözüm' if status == cp_model.OPTIMAL else 'Uygun (Feasible) Çözüm'}**")

        # ---- Gantt Grafik ----
        st.subheader("📊 Dijital Bakım Planlama Gantt Çizelgesi")
        fig = px.timeline(
            df,
            x_start="Planlanan Başlangıç",
            x_end="Planlanan Bitiş",
            y="Atanan Slot",
            color="Uçak Tescil",
            text="Uçak Tescil",
            hover_data=["Bakım Türü", "Gövde Tipi", "Atanan Ekip", "Gecikme (Gün)", "Durum"]
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Operasyon Zaman Akışı", height=400, legend_title="Uçaklar")
        st.plotly_chart(fig, use_container_width=True)

        # ---- Detaylı Tablo + Manuel Müdahale ----
        st.subheader("📋 Detaylı Çizelge Raporu (Manuel Düzenlemeye Açık)")
        st.caption("Gerekirse ekip veya slot atamasını manuel olarak değiştirebilirsiniz; sistem çakışma olup olmadığını "
                    "aşağıda uyarı olarak gösterir (otomatik yeniden optimizasyon yapılmaz).")
        duzenlenmis_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            key="duzenleyici",
            disabled=[c for c in df.columns if c not in ("Atanan Slot", "Atanan Ekip", "Planlanan Başlangıç", "Planlanan Bitiş")]
        )

        # Basit manuel çakışma kontrolü (slot bazında)
        try:
            kontrol_df = duzenlenmis_df.copy()
            kontrol_df["_bas"] = pd.to_datetime(kontrol_df["Planlanan Başlangıç"])
            kontrol_df["_bit"] = pd.to_datetime(kontrol_df["Planlanan Bitiş"])
            uyarilar = []
            for slot, grup in kontrol_df.groupby("Atanan Slot"):
                grup_sirali = grup.sort_values("_bas")
                for k in range(len(grup_sirali) - 1):
                    su = grup_sirali.iloc[k]
                    sonraki = grup_sirali.iloc[k + 1]
                    if su["_bit"] > sonraki["_bas"]:
                        uyarilar.append(f"{slot}: {su['Uçak Tescil']} ile {sonraki['Uçak Tescil']} tarihleri çakışıyor.")
            if uyarilar:
                st.error("⚠️ Manuel düzenleme sonrası çakışma tespit edildi:\n" + "\n".join(f"- {u}" for u in uyarilar))
        except Exception:
            pass

        # ---- Excel İndirme ----
        excel_buffer = excel_indirme_arabellegi(duzenlenmis_df)
        st.download_button(
            label="⬇️ Planı Excel Olarak İndir",
            data=excel_buffer,
            file_name=f"bakim_plani_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ---- Senaryo Karşılaştırma ----
        if senaryo_karsilastir:
            st.subheader("🔀 Senaryo Karşılaştırma: Mevcut vs. Slot +1")
            alt_sonuc, alt_etiket, alt_parametreler = esneklik_ile_coz(
                guncel_ucaklar, st.session_state.ekip_listesi, st.session_state.ekipman_listesi,
                TOPLAM_SLOT + 1, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU, HAFTA_SONU_YASAGI, tatil_indeksleri
            )
            alt_solver, alt_status = alt_sonuc[0], alt_sonuc[1]
            if alt_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                alt_gecikmeler = alt_sonuc[4]
                alt_ceza = alt_solver.Value(sum(
                    alt_gecikmeler[i] * guncel_ucaklar[i]['ceza'] * guncel_ucaklar[i]['oncelik']
                    for i in range(len(guncel_ucaklar))
                ))
                mevcut_ceza = solver.Value(sum(
                    gecikmeler[i] * guncel_ucaklar[i]['ceza'] * guncel_ucaklar[i]['oncelik']
                    for i in range(len(guncel_ucaklar))
                ))
                karsilastirma_df = pd.DataFrame({
                    "Senaryo": [f"Mevcut ({kullanilan_parametreler[0]} Slot)", f"Slot +1 ({TOPLAM_SLOT + 1} Slot)"],
                    "Ağırlıklı Ceza ($)": [mevcut_ceza, alt_ceza]
                })
                fig2 = px.bar(karsilastirma_df, x="Senaryo", y="Ağırlıklı Ceza ($)", color="Senaryo",
                              text="Ağırlıklı Ceza ($)")
                st.plotly_chart(fig2, use_container_width=True)
                fark = mevcut_ceza - alt_ceza
                if fark > 0:
                    st.info(f"💡 Bir ek hangar slotu, toplam ağırlıklı cezayı **{fark:,.0f} $** azaltabilir.")
                else:
                    st.info("💡 Ek hangar slotu bu senaryoda ceza maliyetini azaltmıyor; darboğaz başka bir kaynakta (adam/saat, ekip veya ekipman) olabilir.")
            else:
                st.warning("Alternatif senaryo (Slot +1) için de çözüm bulunamadı.")

elif baslat_butonu and len(guncel_ucaklar) == 0:
    st.info("Planlamayı başlatmak için önce filoya en az bir uçak ekleyin.")
else:
    st.info("Sol panelden filo, ekip ve ekipman bilgilerini girip **🚀 Planlamayı Başlat** butonuna basın.")
    if not SKLEARN_MEVCUT:
        st.caption("Not: scikit-learn kurulu değil, süre tahmini manuel doğrusal regresyon ile yapılacak (fonksiyonel olarak eşdeğer).")
