#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:51:41 2026

@author: ardamercan
"""

import streamlit as st
from datetime import datetime, timedelta
from ortools.sat.python import cp_model
import pandas as pd
import plotly.express as px

# =========================================================================
# ARAYÜZ BAŞLIĞI VE AYARLARI
# =========================================================================
st.set_page_config(page_title="Turkish Technic Optimizasyon", layout="wide")
st.title("✈️ Turkish Technic - Bakım Slot Optimizasyonu")
st.write("Uçak bilgilerini sol menüden değiştirip 'Planlamayı Başlat' butonuna basın.")

# =========================================================================
# SOL PANEL: KULLANICI GİRDİLERİ (INTERFACE)
# =========================================================================
st.sidebar.header("🛠️ Genel Planlama Parametreleri")
TOPLAM_SLOT = st.sidebar.number_input("Toplam Hangar Slot Sayısı", min_value=1, max_value=10, value=2)
GUNLUK_MAKS_ADAM_SAAT = st.sidebar.number_input("Günlük Maksimum Adam/Saat", min_value=10, max_value=2000, value=60)
PLANLAMA_UFUKU = st.sidebar.number_input("Planlama Zaman Ufku (Gün)", min_value=5, max_value=30, value=15)

st.sidebar.header("🛩️ Uçak Özellikleri")

# Uçak verilerini kullanıcıdan dinamik almak için form alanları
ucaklar = []
ucak_harfleri = ["A", "B", "C", "D"]
varsayilanlar = [
    {"sure": 4, "hedef": 5, "ceza": 1000, "adam": 25, "parca": 0, "model": "A320"},
    {"sure": 3, "hedef": 4, "ceza": 1500, "adam": 30, "parca": 2, "model": "B737"},
    {"sure": 6, "hedef": 8, "ceza": 2000, "adam": 40, "parca": 0, "model": "A330"},
    {"sure": 5, "hedef": 6, "ceza": 1200, "adam": 35, "parca": 1, "model": "B787"},
]

for i, harf in enumerate(ucak_harfleri):
    with st.sidebar.expander(f"Uçak_{harf} ({varsayilanlar[i]['model']}) Ayarları"):
        sure = st.number_input(f"Bakım Süresi (Gün) - Uçak_{harf}", min_value=1, value=varsayilanlar[i]["sure"])
        hedef = st.number_input(f"Teslim Hedefi (Gün) - Uçak_{harf}", min_value=1, value=varsayilanlar[i]["hedef"])
        ceza = st.number_input(f"Gecikme Cezası ($) - Uçak_{harf}", min_value=0, value=varsayilanlar[i]["ceza"])
        adam = st.number_input(f"Günlük Adam/Saat - Uçak_{harf}", min_value=0, value=varsayilanlar[i]["adam"])
        parca = st.number_input(f"Parça Bekleme Süresi (Gün) - Uçak_{harf}", min_value=0, value=varsayilanlar[i]["parca"])
        
        ucaklar.append({
            "ad": f"Uçak_{harf} ({varsayilanlar[i]['model']})",
            "sure": sure,
            "teslim_hedefi": hedef,
            "ceza": ceza,
            "adam_saat": adam,
            "parca_gun": parca
        })

# Hesaplama Butonu
baslat_butonu = st.sidebar.button("🚀 Planlamayı Başlat", type="primary")

# =========================================================================
# OPTİMİZASYON MOTORU
# =========================================================================
def optimizasyon_calistir(ucaklar, TOPLAM_SLOT, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU):
    model = cp_model.CpModel()
    baslangic, bitis, aralik, gecikmeler, slot_atama = {}, {}, {}, {}, {}

    for i, ucak in enumerate(ucaklar):
        baslangic[i] = model.NewIntVar(0, PLANLAMA_UFUKU - ucak["sure"], f"bas_{i}")
        bitis[i] = model.NewIntVar(0, PLANLAMA_UFUKU, f"bit_{i}")
        aralik[i] = model.NewIntervalVar(baslangic[i], ucak["sure"], bitis[i], f"aralik_{i}")
        slot_atama[i] = model.NewIntVar(1, TOPLAM_SLOT, f"slot_{i}")
        gecikmeler[i] = model.NewIntVar(0, PLANLAMA_UFUKU, f"gecikme_{i}")
        model.AddMaxEquality(gecikmeler[i], [0, bitis[i] - ucak["teslim_hedefi"]])

    model.AddCumulative(intervals=[aralik[i] for i in range(len(ucaklar))], demands=[1 for _ in ucaklar], capacity=TOPLAM_SLOT)
    model.AddCumulative(intervals=[aralik[i] for i in range(len(ucaklar))], demands=[ucak["adam_saat"] for ucak in ucaklar], capacity=GUNLUK_MAKS_ADAM_SAAT)

    for i, ucak in enumerate(ucaklar):
        if ucak["parca_gun"] > 0:
            model.Add(baslangic[i] >= ucak["parca_gun"])

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

    toplam_ceza = sum(gecikmeler[i] * ucaklar[i]["ceza"] for i in range(len(ucaklar)))
    model.Minimize(toplam_ceza)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    return solver, status, baslangic, bitis, gecikmeler, slot_atama

# =========================================================================
# ÇIKTI EKRANI (DASHBOARD)
# =========================================================================
if baslat_butonu:
    solver, status, baslangic, bitis, gecikmeler, slot_atama = optimizasyon_calistir(
        ucaklar, TOPLAM_SLOT, GUNLUK_MAKS_ADAM_SAAT, PLANLAMA_UFUKU
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        st.error("❌ Girilen kısıtlar altında uygun bir planlama bulunamadı! Lütfen slot sayısını veya adam/saat kapasitesini artırın.")
    else:
        # Metrik Kartları
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Toplam Gecikme Cezası", value=f"{solver.ObjectiveValue():,.0f} $")
        with col2:
            durum_metni = "En İyi Çözüm (Optimal)" if status == cp_model.OPTIMAL else "Uygun Çözüm (Feasible)"
            st.metric(label="Çözücü Durumu", value=durum_metni)

        # Tablo oluşturma
        bugun = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        veri_listesi = []
        for i, ucak in enumerate(ucaklar):
            bas_gun = solver.Value(baslangic[i])
            bit_gun = solver.Value(bitis[i])
            gec_gun = solver.Value(gecikmeler[i])
            veri_listesi.append({
                "Uçak": ucak["ad"],
                "Başlangıç": bugun + timedelta(days=bas_gun),
                "Bitiş": bugun + timedelta(days=bit_gun),
                "Slot": f"Hangar Slot {solver.Value(slot_atama[i])}",
                "Süre (Gün)": ucak["sure"],
                "Adam/Saat": ucak["adam_saat"],
                "Gecikme (Gün)": gec_gun,
                "Durum": "🔴 Gecikmeli" if gec_gun > 0 else "🟢 Zamanında",
            })
        df = pd.DataFrame(veri_listesi)

        # Gantt Şeması Çizimi
        st.subheader("📊 Optimizasyon Sonucu Gantt Şeması")
        fig = px.timeline(
            df, x_start="Başlangıç", x_end="Bitiş", y="Slot", color="Uçak", text="Uçak",
            hover_data=["Süre (Gün)", "Adam/Saat", "Gecikme (Gün)", "Durum"],
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Tarih", height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Veri Tablosu Gösterimi
        st.subheader("📋 Detaylı Planlama Tablosu")
        st.dataframe(df, use_container_width=True)
else:
    st.info("💡 Planlama sonuçlarını ve grafiği görmek için sol taraftaki menüden değerleri ayarlayıp **'Planlamayı Başlat'** butonuna basın.")
