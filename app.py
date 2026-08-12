import streamlit as st
from datetime import datetime, timedelta
from ortools.sat.python import cp_model
import pandas as pd
import plotly.express as px
import io

# =========================================================================
# CONFIG & BAŞLIK
# =========================================================================
st.set_page_config(page_title="Turkish Technic Gelişmiş Optimizasyon", layout="wide")
st.title("✈️ Gelişmiş Uçak Bakım Slot Optimizasyon Sistemi")
st.write("Turkish Technic operasyonel kurallarına göre dinamik planlama paneli.")

# =========================================================================
# SOL PANEL: GENEL PARAMETRELER
# =========================================================================
st.sidebar.header("⚙️ 1. Genel Planlama Ayarları")
TOPLAM_SLOT = st.sidebar.number_input("Toplam Hangar Slot Sayısı", min_value=1, max_value=5, value=2)
GUNLUK_MAKS_ADAM_SAAT = st.sidebar.number_input("Günlük Maksimum Adam/Saat", min_value=10, max_value=300, value=70)
PLANLAMA_UFUKU = st.sidebar.number_input("Zaman Ufku (Gün)", min_value=5, max_value=40, value=20)
HAFTA_SONU_YASAGI = st.sidebar.checkbox("Bakım hafta sonu başlayamaz / bitemez", value=True)

st.sidebar.markdown("---")
st.sidebar.header("🛩️ 2. Filo Yönetimi (Uçak Ekleme)")

# Session State ile dinamik uçak listesi tutma
if "ucak_listesi" not in st.session_state:
    st.session_state.ucak_listesi = [
        {"ad": "TC-JPE", "model_tipi": "Dar Gövde (A320/B737)", "bakim_tipi": "C-Check", "teslim_hedefi": 5, "ceza": 1000, "parca_gun": 0, "oncelik": "3 - Normal"},
        {"ad": "TC-JJJ", "model_tipi": "Geniş Gövde (A330/B787)", "bakim_tipi": "D-Check", "teslim_hedefi": 8, "ceza": 2000, "parca_gun": 2, "oncelik": "5 - AOG (Kritik)"},
    ]

# Yeni uçak ekleme arayüzü
with st.sidebar.expander("➕ Yeni Uçak Tanımla", expanded=False):
    y_ad = st.text_input("Kuyruk Tescili / Adı", "TC-XYZ")
    y_model = st.selectbox("Gövde Tipi", ["Dar Gövde (A320/B737)", "Geniş Gövde (A330/B787)"])
    y_bakim = st.selectbox("Bakım Türü", ["A-Check (Hafif)", "C-Check (Ağır)", "D-Check (En Ağır)"])
    y_hedef = st.number_input("Hedef Teslim (Gün)", min_value=1, value=6)
    y_ceza = st.number_input("Gecikme Cezası ($/Gün)", min_value=0, value=1200)
    y_parca = st.number_input("Parça Bekleme (Gün)", min_value=0, value=0)
    y_oncelik = st.selectbox("Öncelik Derecesi", ["1 - Düşük", "2 - Düşük-Orta", "3 - Normal", "4 - Yüksek", "5 - AOG (Kritik)"])
    
    if st.button("Filoya Ekle"):
        st.session_state.ucak_listesi.append({
            "ad": y_ad, "model_tipi": y_model, "bakim_tipi": y_bakim,
            "teslim_hedefi": y_hedef, "ceza": y_ceza, "parca_gun": y_parca, "oncelik": y_oncelik
        })
        st.success(f"{y_ad} başarıyla filoya eklendi!")

# Mevcut Uçakları Listeleme ve Düzenleme
st.sidebar.write(f"**Filodaki Güncel Uçak Sayısı:** {len(st.session_state.ucak_listesi)}")
guncel_ucaklar = []

for idx, uc in enumerate(st.session_state.ucak_listesi):
    with st.sidebar.expander(f"⚙️ {uc['ad']} ({uc['bakim_tipi']})"):
        u_ad = st.text_input("Ad", uc["ad"], key=f"ad_{idx}")
        u_model = st.selectbox("Gövde", ["Dar Gövde (A320/B737)", "Geniş Gövde (A330/B787)"], index=0 if "Dar" in uc["model_tipi"] else 1, key=f"mod_{idx}")
        u_bakim = st.selectbox("Tür", ["A-Check (Hafif)", "C-Check (Ağır)", "D-Check (En Ağır)"], index=0 if "A-Check" in uc["bakim_tipi"] else (1 if "C-Check" in uc["bakim_tipi"] else 2), key=f"bak_{idx}")
        u_hedef = st.number_input("Hedef (Gün)", min_value=1, value=uc["teslim_hedefi"], key=f"hdf_{idx}")
        u_ceza = st.number_input("Ceza ($)", min_value=0, value=uc["ceza"], key=f"cz_{idx}")
        u_parca = st.number_input("Parça Günü", min_value=0, value=uc["parca_gun"], key=f"prc_{idx}")
        u_oncelik = st.selectbox("Öncelik", ["1 - Düşük", "2 - Düşük-Orta", "3 - Normal", "4 - Yüksek", "5 - AOG (Kritik)"], index=int(uc["oncelik"][0])-1, key=f"onc_{idx}")
        
        # Bakım tipine göre otomatik teknik kısıt atamaları
        if "A-Check" in u_bakim:
            sure, adam = 2, 15
        elif "C-Check" in u_bakim:
            sure, adam = 5, 30
        else: # D-Check
            sure, adam = 8, 45
            
        guncel_ucaklar.append({
            "ad": u_ad, "model_tipi": u_model, "bakim_tipi": u_bakim, "sure": sure, "adam_saat": adam,
            "teslim_hedefi": u_hedef, "ceza": u_ceza, "parca_gun": u_parca, "oncelik": int(u_oncelik[0])
        })

if st.sidebar.button("🗑️ Tüm Filoyu Sıfırla"):
    st.session_state.ucak_listesi = []
    st.rerun()

baslat_butonu = st.sidebar.button("🚀 Planlamayı Başlat", type="primary")

# =========================================================================
# OPERASYONEL OPTİMİZASYON MOTORU
# =========================================================================
def gelismis_optimizasyon(ucaklar, TOPLAM_SLOT, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU, HAFTA_SONU_YASAGI):
    model = cp_model.CpModel()
    baslangic, bitis, aralik, gecikmeler, slot_atama = {}, {}, {}, {}, {}
    bugun = datetime.now()

    # Hafta sonu günlerinin indeks tespiti
    haftasonu_indeksleri = []
    for d in range(PLANLAMA_UFUKU + 1):
        gelecek_tarih = bugun + timedelta(days=d)
        if gelecek_tarih.weekday() in (5, 6): # 5: Cumartesi, 6: Pazar
            haftasonu_indeksleri.append(d)

    for i, ucak in enumerate(ucaklar):
        baslangic[i] = model.NewIntVar(0, PLANLAMA_UFUKU - ucak["sure"], f"bas_{i}")
        bitis[i] = model.NewIntVar(0, PLANLAMA_UFUKU, f"bit_{i}")
        aralik[i] = model.NewIntervalVar(baslangic[i], ucak["sure"], bitis[i], f"aralik_{i}")
        
        # Hangar Slot Kısıtı: Geniş gövde sadece Slot 2 ve sonrasına girebilsin
        if "Geniş Gövde" in ucak["model_tipi"] and TOPLAM_SLOT >= 2:
            slot_atama[i] = model.NewIntVar(2, TOPLAM_SLOT, f"slot_{i}")
        else:
            slot_atama[i] = model.NewIntVar(1, TOPLAM_SLOT, f"slot_{i}")
            
        gecikmeler[i] = model.NewIntVar(0, PLANLAMA_UFUKU, f"gecikme_{i}")
        model.AddMaxEquality(gecikmeler[i], [0, bitis[i] - ucak["teslim_hedefi"]])

        # Hafta sonu yasağı kısıtı
        if HAFTA_SONU_YASAGI:
            for h_gun in haftasonu_indeksleri:
                model.Add(baslangic[i] != h_gun)
                model.Add(bitis[i] != h_gun)

        # Parça Tedarik Kısıtı
        if ucak["parca_gun"] > 0:
            model.Add(baslangic[i] >= ucak["parca_gun"])

    # Kapasite Kısıtları
    model.AddCumulative(intervals=[aralik[i] for i in range(len(ucaklar))], demands=[1 for _ in ucaklar], capacity=TOPLAM_SLOT)
    model.AddCumulative(intervals=[aralik[i] for i in range(len(ucaklar))], demands=[ucak["adam_saat"] for ucak in ucaklar], capacity=GUNLUK_MAKS_ADAM_SAAT)

    # Çakışma ve Slot Atama Mantığı
    for i in range(len(ucaklar)):
        for j in range(i + 1, len(ucaklar)):
            ayni_slot = model.NewBoolVar(f"ayni_slot_{i}_{j}")
            model.Add(slot_atama[i] == slot_atama[j]).OnlyEnforceIf(ayni_slot)
            model.Add(slot_atama[i] != slot_atama[j]).OnlyEnforceIf(ayni_slot.Not())
            i_once = model.NewBoolVar(f"i_once_{i}_{j}")
            j_once = model.NewBoolVar(f"j_once_{i}_{j}")
            model.Add(bitis[i] <= baslangic[j]).OnlyEnforceIf([ayni_slot, i_once])
            model.Add(bitis[j] <= baslangic[i]).OnlyEnforceIf([ayni_slot, j_once])
            model.AddBoolOr([i_once, j_once]).OnlyEnforceIf(ayni_slot)

    # Öncelikli Amaç Fonksiyonu
    toplam_ceza = sum(gecikmeler[i] * ucaklar[i]["ceza"] * ucaklar[i]["oncelik"] for i in range(len(ucaklar)))
    model.Minimize(toplam_ceza)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    return solver, status, baslangic, bitis, gecikmeler, slot_atama

# =========================================================================
# DASHBOARD / ÇIKTI SİSTEMİ
# =========================================================================
if baslat_butonu and len(guncel_ucaklar) > 0:
    solver, status, baslangic, bitis, gecikmeler, slot_atama = gelismis_optimizasyon(
        guncel_ucaklar, TOPLAM_SLOT, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU, HAFTA_SONU_YASAGI
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        st.error("❌ Belirtilen operasyonel kurallar ve kapasite sınırları dahilinde ÇÖZÜM BULUNAMADI! Lütfen hangar slotunu veya günlük adam/saat sınırını esnetin.")
    else:
        # Metrik Skor Kartları
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Ağırlıklı Toplam Ceza Maliyeti", f"{solver.ObjectiveValue():,.0f} $")
        with c2: st.metric("Planlanan Toplam Uçak", f"{len(guncel_ucaklar)} Yakın")
        with c3: st.metric("Yazılım Durumu", "Optimal Çözüm Başarılı" if status == cp_model.OPTIMAL else "Uygun Plan Bulundu")

        # Veri Dönüştürme
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
                "Atanan Slot": f"Hangar Slot {solver.Value(slot_atama[i])}",
                "Planlanan Başlangıç": (bugun + timedelta(days=b_gun)).strftime("%Y-%m-%d %H:%M"),
                "Planlanan Bitiş": (bugun + timedelta(days=bt_gun)).strftime("%Y-%m-%d %H:%M"),
                "Süre (Gün)": uc["sure"],
                "Gecikme (Gün)": g_gun,
                "Öncelik Skoru": uc["oncelik"],
                "Durum": "🔴 Gecikmeli" if g_gun > 0 else "🟢 Zamanında"
            })
        df = pd.DataFrame(veri)

        # Gantt Grafik Gösterimi
        st.subheader("📊 Dijital Bakım Planlama Gantt Çizelgesi")
fig = px.timeline(df, x_start="Planlanan Başlangıç", x_end="Planlanan Bitiş", y="Atanan Slot",color="Uçak Tescil", text="Uçak Tescil",hover_data=["Bakım Türü", "Gövde Tipi", "Gecikme (Gün)", "Durum"])fig.update_yaxes(autorange="reversed")fig.update_layout(xaxis_title="Operasyon Zaman Akışı", height=400, legend_title="Uçaklar")st.plotly_chart(fig, use_container_width=True) 
