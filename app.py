"""
app.py
Streamlit‑интерфейс к ядру Py‑qpcR с поддержкой расчета ΔΔCt
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import io
import re
from scipy import stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from qpcr_data import (
        load_qpcr_csv,
        build_dataset_from_raw,
        baseline_subtract,
        QPCRDataset,
        coerce_numeric_columns,
        convert_qiagen_to_normal,
    )
    from qpcr_models import (
        fit_curve_l4,
        fit_curve_l5,
        fit_curve_auto,
        l4_model,
        l5_model,
    )
    from qpcr_analysis import (
        batch_fit,
        calib_efficiency,
        relative_expression,
    )
except ImportError:
    # Заглушка, если модулей нет в текущей директории при тестировании
    pass

# ======================
# НАСТРОЙКА СТРАНИЦЫ
# ======================
st.set_page_config(
    page_title="Py-qpcR",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Py-qpcR – интерактивный аналог qpcR")

# --- ВЫБОР РЕЖИМА РАБОТЫ ---
app_mode = st.radio(
    "Выберите режим работы:",
    ["📈 Анализ сырых кривых (Оригинал)", "📊 Расчет экспрессии (ΔΔCt)"],
    horizontal=True
)

if app_mode == "📊 Расчет экспрессии (ΔΔCt)":
    st.markdown("### Загрузка таблицы с пороговыми циклами (Ct)")
    st.info("Загрузите CSV или Excel файл. Формат: один столбец с образцами, остальные — гены. Пропуски обозначаются как '-'.")
    
    uploaded_ct = st.file_uploader("Выберите файл данных (CSV/XLSX)", type=["csv", "xlsx"], key="ct_upload")
    
    if uploaded_ct is not None:
        try:
            # Чтение файла
            if uploaded_ct.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_ct)
            else:
                df_raw = pd.read_excel(uploaded_ct)
            
            # Очистка пустых строк и столбцов по краям
            df_raw = df_raw.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            # Если Pandas считал первые пустые строки, а заголовки оказались ниже (Unnamed: 0...)
            if any(str(c).startswith("Unnamed") for c in df_raw.columns):
                df_raw.columns = df_raw.iloc[0]
                df_raw = df_raw[1:]
            
            # --- ОЧИСТКА ДАННЫХ ---
            df = df_raw.copy()
            df = df.replace('-', np.nan)
            df = df.dropna(how='all')
            
            # Ищем и удаляем дублирующиеся заголовки в теле таблицы (например, слово "День")
            first_col = df.columns[0]
            df = df[df[first_col] != first_col].reset_index(drop=True)
            
            st.markdown("**Предпросмотр очищенных данных:**")
            st.dataframe(df.head())
            
            st.markdown("### ⚙️ Настройка расчета")
            col1, col2 = st.columns(2)
            
            with col1:
                sample_col = st.selectbox("Столбец с названиями образцов (Условия):", options=df.columns)
                gene_cols = [c for c in df.columns if c != sample_col]
                
                # Заполняем пропуски в названиях образцов (если ячейки были объединены)
                df[sample_col] = df[sample_col].ffill()
                unique_samples = df[sample_col].dropna().unique()
                
                control_sample = st.selectbox("Контрольный образец (Калибратор):", options=unique_samples)
                
            with col2:
                ref_gene = st.selectbox("Референсный ген (Housekeeping):", options=gene_cols)
                target_options = [c for c in gene_cols if c != ref_gene]
                target_genes = st.multiselect("Опытные гены (Target):", options=target_options, default=target_options)
            
            if st.button("🚀 Рассчитать относительную экспрессию (Fold Change)", type="primary"):
                if not target_genes:
                    st.warning("Пожалуйста, выберите хотя бы один опытный ген.")
                else:
                    # Конвертируем выбранные колонки в числа
                    df_calc = df.copy()
                    cols_to_convert = [ref_gene] + target_genes
                    for c in cols_to_convert:
                        df_calc[c] = pd.to_numeric(df_calc[c], errors='coerce')
                    
                    # Группируем по образцам (усредняем технические повторы)
                    df_grouped = df_calc.groupby(sample_col)[cols_to_convert].mean()
                    
                    results = []
                    # Проверяем, есть ли данные для контроля
                    if control_sample not in df_grouped.index:
                        st.error(f"Контрольный образец '{control_sample}' не найден после очистки данных.")
                    elif pd.isna(df_grouped.loc[control_sample, ref_gene]):
                        st.error(f"У контрольного образца '{control_sample}' нет значения для референсного гена '{ref_gene}'.")
                    else:
                        ref_ctrl_ct = df_grouped.loc[control_sample, ref_gene]
                        
                        for target in target_genes:
                            if pd.isna(df_grouped.loc[control_sample, target]):
                                st.warning(f"У контроля нет значения Ct для гена {target}. Расчет для этого гена пропущен.")
                                continue
                                
                            target_ctrl_ct = df_grouped.loc[control_sample, target]
                            dCt_ctrl = target_ctrl_ct - ref_ctrl_ct
                            
                            for sample in df_grouped.index:
                                target_ct = df_grouped.loc[sample, target]
                                ref_ct = df_grouped.loc[sample, ref_gene]
                                
                                if pd.isna(target_ct) or pd.isna(ref_ct):
                                    continue # Пропускаем, если нет данных
                                    
                                dCt = target_ct - ref_ct
                                ddCt = dCt - dCt_ctrl
                                fold_change = 2 ** (-ddCt)
                                
                                results.append({
                                    "Образец": sample,
                                    "Ген": target,
                                    "Ct Target": round(target_ct, 2),
                                    "Ct Ref": round(ref_ct, 2),
                                    "ΔCt": round(dCt, 3),
                                    "ΔΔCt": round(ddCt, 3),
                                    "Fold Change (2^-ΔΔCt)": round(fold_change, 3)
                                })
                        
                        if results:
                            res_df = pd.DataFrame(results)
                            st.success("Расчет успешно выполнен!")
                            
                            st.markdown("### 📋 Таблица результатов")
                            st.dataframe(res_df, use_container_width=True)
                            
                            # Кнопка скачивания результатов
                            csv_out = res_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Скачать результаты (CSV)",
                                data=csv_out,
                                file_name="ddCt_results.csv",
                                mime="text/csv",
                            )
                            
                            st.markdown("### 📊 График относительной экспрессии")
                            fig = px.bar(
                                res_df, 
                                x="Образец", 
                                y="Fold Change (2^-ΔΔCt)", 
                                color="Ген",
                                barmode="group",
                                text="Fold Change (2^-ΔΔCt)",
                                title="Относительная экспрессия генов"
                            )
                            fig.update_traces(textposition='outside')
                            fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Контроль (База=1)")
                            fig.update_layout(yaxis_title="Относительная экспрессия (Fold Change)", xaxis_title="")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Не удалось рассчитать значения. Проверьте исходные данные.")
                            
        except Exception as e:
            st.error(f"Произошла ошибка при обработке файла: {e}")
            
    # Прерываем выполнение скрипта, чтобы не отрисовывался оригинальный интерфейс ниже
    st.stop()

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================
def init_state():
            excel_file = pd.ExcelFile(uploaded_excel)
            sheet_names = excel_file.sheet_names
            st.markdown("### 📄 Выбор листа")
            selected_sheet = st.selectbox("Выберите лист с сырыми данными:", options=sheet_names)
            df_excel = pd.read_excel(uploaded_excel, sheet_name=selected_sheet)
            
            st.markdown("**Предпросмотр данных:**")
            st.dataframe(df_excel.head(10))
            
            st.markdown("### ⚙️ Настройки экспорта")
            col1, col2 = st.columns(2)
            with col1:
                csv_separator = st.selectbox(
                    "Разделитель CSV:", options=[",", ";", "\t", "|"],
                    format_func=lambda x: {",": "Запятая (,)", ";": "Точка с запятой (;)", "\t": "Табуляция (\\t)", "|": "Вертикальная черта (|)"}[x],
                    key="csv_sep"
                )
            with col2:
                include_index = st.checkbox("Включить индекс строк", value=False, key="csv_index")
                
            csv_data = df_excel.to_csv(sep=csv_separator, index=include_index, encoding="utf-8").encode("utf-8")
            original_name = uploaded_excel.name.rsplit(".", 1)[0]
            csv_filename = f"{original_name}_{selected_sheet}.csv"
            
            st.download_button("⬇️ Скачать CSV", data=csv_data, file_name=csv_filename, mime="text/csv")
        except Exception as e:
            st.error(f"Ошибка чтения Excel: {e}")