import streamlit as st
from datetime import datetime, timedelta, date
from ortools.sat.python import cp_model
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# =========================================================================
# CONFIG & BAŞLIK
# =========================================================================
st.set_page_config(page_title="Turkish Technic Enterprise Optimizasyon", layout="wide")
st.title("✈️ Turkish Technic - Akıllı Bakım Slot & Kaynak Optimizasyonu")
st.write("Yapay Zeka Destekli Operasyonel Çizelgeleme ve Kaynak Planlama Kontrol Paneli.")

# =========================================================================
# SOL PANEL: GENEL PARAMETRELER
# =========================================================================
st.sidebar.header("⚙️ 1. Planlama Kontrol Parametreleri")
PLAN_BASLANGIC = st.sidebar.date_input("Planlama Başlangıç Tarihi", date.today())
TOPLAM_SLOT = st.sidebar.number_input("Toplam Hangar Slot Sayısı", min_value=1, max_value=5, value=2)
PLANLAMA_UFUKU = st.sidebar.number_input("Zaman Ufku (Gün)", min_value=5, max_value=40, value=20)
HAFTA_SONU_YASAGI = st.sidebar.checkbox("Bakım hafta sonu başlayamaz / bitemez", value=True)

st.sidebar.markdown("---")
st.sidebar.header("👷 2. Teknik Personel & Sertifika Kısıtları")
GUNLUK_MAKS_ADAM_SAAT = st.sidebar.number_input("Günlük Maksimum Adam/Saat", min_value=10, max_value=400, value=80)
DAR_GOVDE_TEKNISYEN = st.sidebar.number_input("Maks. Dar Gövde Teknisyeni / Gün", min_value=1, max_value=50, value=15)
GENIS_GOVDE_TEKNISYEN = st.sidebar.number_input("Maks. Geniş Gövde Teknisyeni / Gün", min_value=1, max_value=50, value=15)

st.sidebar.markdown("---")
st.sidebar.header("🛩️ 3. Filo Yönetimi (Uçak Ekleme)")

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
        
        # Bakım tipine göre dinamik insan gücü ve teknisyen gereksinimleri
        if "A-Check" in u_bakim:
            sure, adam, teknisyen_ihtiyac = 2, 15, 3
        elif "C-Check" in u_bakim:
            sure, adam, teknisyen_ihtiyac = 5, 30, 6
        else: # D-Check
            sure, adam, teknisyen_ihtiyac = 8, 45, 10
            
        guncel_ucaklar.append({
            "ad": u_ad, "model_tipi": u_model, "bakim_tipi": u_bakim, "sure": sure, "adam_saat": adam,
            "teknisyen_ihtiyac": teknisyen_ihtiyac, "teslim_hedefi": u_hedef, "ceza": u_ceza, 
            "parca_gun": u_parca, "oncelik": int(u_oncelik[0])
        })

if st.sidebar.button("🗑️ Tüm Filoyu Sıfırla"):
    st.session_state.ucak_listesi = []
    st.rerun()

baslat_butonu = st.sidebar.button("🚀 Planlamayı Başlat", type="primary")

# =========================================================================
# OPERASYONEL OPTİMİZASYON MOTORU
# =========================================================================
def gelismis_optimizasyon(ucaklar, TOPLAM_SLOT, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU, HAFTA_SONU_YASAGI, PLAN_BASLANGIC, DAR_GOVDE_TEKNISYEN, GENIS_GOVDE_TEKNISYEN):
    model = cp_model.CpModel()
    baslangic, bitis, aralik, gecikmeler, slot_atama = {}, {}, {}, {}, {}

    # Hafta sonu günlerinin indeks tespiti
    haftasonu_indeksleri = []
    for d in range(PLANLAMA_UFUKU + 1):
        gelecek_tarih = datetime.combine(PLAN_BASLANGIC, datetime.min.time()) + timedelta(days=d)
        if gelecek_tarih.weekday() in (5, 6):
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

        if HAFTA_SONU_YASAGI:
            for h_gun in haftasonu_indeksleri:
                model.Add(baslangic[i] != h_gun)
                model.Add(bitis[i] != h_gun)

        if ucak["parca_gun"] > 0:
            model.Add(baslangic[i] >= ucak["parca_gun"])

    # Global Kapasite Kısıtları
    model.AddCumulative(intervals=[aralik[i] for i in range(len(ucaklar))], demands=[1 for _ in ucaklar], capacity=TOPLAM_SLOT)
    model.AddCumulative(intervals=[aralik[i] for i in range(len(ucaklar))], demands=[ucak["adam_saat"] for ucak in ucaklar], capacity=GUNLUK_MAKS_ADAM_SAAT)

    # Spesifik Sertifikalı Teknisyen Havuz Kısıtları
    model.AddCumulative(
        intervals=[aralik[i] for i in range(len(ucaklar))],
        demands=[ucak["teknisyen_ihtiyac"] if "Dar Gövde" in ucak["model_tipi"] else 0 for ucak in ucaklar],
        capacity=DAR_GOVDE_TEKNISYEN
    )
    model.AddCumulative(
        intervals=[aralik[i] for i in range(len(ucaklar))],
        demands=[ucak["teknisyen_ihtiyac"] if "Geniş Gövde" in ucak["model_tipi"] else 0 for ucak in ucaklar],
        capacity=GENIS_GOVDE_TEKNISYEN
    )

    # Çakışma Mantığı
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

    # Öncelikli Maliyet Minimizasyonu
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
        guncel_ucaklar, TOPLAM_SLOT, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU, HAFTA_SONU_YASAGI, PLAN_BASLANGIC, DAR_GOVDE_TEKNISYEN, GENIS_GOVDE_TEKNISYEN
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        st.error("❌ Belirtilen teknik sertifika, personel veya slot sınırları dahilinde ÇÖZÜM BULUNAMADI! Lütfen kısıtları gevşetin.")
    else:
        # Veri Dönüştürme (Öncelikli Başlangıç)
        bugun = datetime.combine(PLAN_BASLANGIC, datetime.min.time()).replace(hour=8, minute=0)
        veri, maliyet_veri = [], []
        toplam_is_gunu = 0

        for i, uc in enumerate(guncel_ucaklar):
            b_gun = solver.Value(baslangic[i])
        for i, uc in enumerate(guncel_ucaklar):
            b_gun = solver.Value(baslangic[i])
            bt_gun = solver.Value(bitis[i])
            g_gun = solver.Value(gecikmeler[i])
            gercek_maliyet = g_gun * uc["ceza"]
            toplam_is_gunu += uc["sure"]

            veri.append({
                "Uçak Tescil": uc["ad"],
                "Gövde Tipi": uc["model_tipi"],
                "Bakım Türü": uc["bakim_tipi"],
                "Atanan Slot": f"Hangar Slot {solver.Value(slot_atama[i])}",
                "Planlanan Başlangıç": (bugun + timedelta(days=b_gun)).strftime("%Y-%m-%d %H:%M"),
                "Planlanan Bitiş": (bugun + timedelta(days=bt_gun)).strftime("%Y-%m-%d %H:%M"),
                "Süre (Gün)": uc["sure"],
                "Gecikme (Gün)": g_gun,
                "Ceza Maliyeti ($)": gercek_maliyet,
                "Durum": "🔴 Gecikmeli" if g_gun > 0 else "🟢 Zamanında"
            })
            
            if gercek_maliyet > 0:
                maliyet_veri.append({"Uçak": uc["ad"], "Maliyet": gercek_maliyet})

        df = pd.DataFrame(veri)

                st.subheader("📊 Operasyonel Performans Özet Paneli")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.metric("Toplam Gerçekleşen Ceza", f"{df['Ceza Maliyeti ($)'].sum():,.0f} $")
            st.metric("Zamanında Teslim Oranı", f"{(df['Gecikme (Gün)'] == 0).mean() * 100:.0f}%")
            
        with m2:
            st.metric("Planlanan Uçak", f"{len(guncel_ucaklar)} Adet")
            st.metric("Toplam Bakım İş Gücü", f"{toplam_is_gunu} Gün")
            
        with m3:
            max_kapasite_gun = TOPLAM_SLOT * PLANLAMA_UFUKU
            doluluk_orani = (toplam_is_gunu / max_kapasite_gun) * 100 if max_kapasite_gun > 0 else 0
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=doluluk_orani,
                title={'text': "Hangar Slot Doluluk Oranı (%)"},
                gauge={
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range':, 'color': 'lightgray'}, 
                        {'range':, 'color': 'gray'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")


        g1, g2 = st.columns(2)
        
        with g1:
            st.subheader("📅 Dijital Bakım Planlama Gantt Çizelgesi")
            fig = px.timeline(
                df, 
                x_start="Planlanan Başlangıç", 
                x_end="Planlanan Bitiş", 
                y="Atanan Slot", 
                color="Uçak Tescil", 
                text="Uçak Tescil",
                hover_data=["Bakım Türü", "Gövde Tipi", "Gecikme (Gün)", "Durum"]
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(xaxis_title="Operasyon Zaman Akışı", height=350, legend_title="Uçaklar")
            st.plotly_chart(fig, use_container_width=True)
            
        with g2:
            st.subheader("💰 Ceza Maliyet Dağılımı")
            if len(maliyet_veri) > 0:
                df_mal = pd.DataFrame(maliyet_veri)
                fig_pie = px.pie(df_mal, values='Maliyet', names='Uçak', hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)
                fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.success("🎉 Harika! Hiçbir uçakta gecikme cezası oluşmadı.")

        st.subheader("📋 Detaylı Çizelge Raporu")
        st.dataframe(df, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Bakım_Plani', index=False)
        
        st.download_button(
            label="📥 Planlama Raporunu Excel Olarak İndir",
            data=buffer.getvalue(),
            file_name=f"Turkish_Technic_Gelişmiş_Plan_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )
elif len(guncel_ucaklar) == 0:
    st.warning("⚠️ Lütfen sol menüden filoya en az bir uçak ekleyin.")
else:
    st.info("💡 Tüm kurumsal ve teknik kısıtlar entegre edildi. Başlatmak için sol paneldeki **'Planlamayı Başlat'** butonuna basın.")
