import os
import re
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dashboard import (
    DATA_SNAPSHOT,
    KPI_SPECS,
    OCCUPANCY_CAUTION,
    OCCUPANCY_RISK,
    RETRYABLE_GEMINI_ERRORS,
    generate_ai_summary,
    build_kpi_chart,
    build_occupancy_bar,
    kpi_delta,
    kpi_period,
    occupancy_status,
    read_dashboard_data,
)


ROOT = Path(__file__).resolve().parent


PAGE_HEADER = """
<div class="hero"><div class="hero-main">
<div class="hero-kicker">BUSAN NEW PORT / OPERATIONS</div>
<div class="hero-title">부산 신항 처리 효율 개선 방안</div>
<div class="hero-copy">부산 신항 부두별 선석 운영과 처리 생산성</div>
</div></div>
"""

def render_ai_summary(series):
    state_key = "eda_overview_ai_summary"
    error_key = "eda_summary_error"
    future_key = "eda_summary_future"
    if not st.session_state.get(state_key) and error_key not in st.session_state:
        future = st.session_state.get(future_key)
        if future is None:
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                try:
                    key = st.secrets.get("GEMINI_API_KEY")
                except (FileNotFoundError, KeyError):
                    key = None
            if key:
                executor = ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(generate_ai_summary, key, series)
                    st.session_state[future_key] = future
                finally:
                    executor.shutdown(wait=False)
            else:
                st.session_state[error_key] = "Gemini API 키를 설정하면 요약을 볼 수 있어요."

        if future is not None and future.done():
            try:
                st.session_state[state_key] = future.result()
            except Exception as exc:
                status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if status_code == 429:
                    message = "요약 사용량 한도에 도달했어요. 잠시 후 새로고침해 주세요."
                elif status_code in RETRYABLE_GEMINI_ERRORS:
                    message = f"Gemini 서버가 응답하지 않았어요 (HTTP {status_code}). 잠시 후 새로고침해 주세요."
                else:
                    message = "요약을 불러오지 못했어요. 잠시 후 새로고침해 주세요."
                st.session_state[error_key] = message
            finally:
                del st.session_state[future_key]

    with st.container(key="ai_summary_card", border=True):
        st.subheader("AI 분석 요약")
        if st.session_state.get(state_key):
            st.markdown(st.session_state[state_key])
        elif error_key in st.session_state:
            st.caption(st.session_state[error_key])
        else:
            st.markdown(
                '<div class="ai-summary-loading" role="status">'
                '<span class="ai-summary-spinner" aria-hidden="true"></span>'
                '<span>요약 생성중입니다...</span></div>',
                unsafe_allow_html=True,
            )


def kpi_text(value, unit):
    return f"{value:.2f}{unit}" if value is not None else "선택 기간 데이터 부족"


def delta_badge(key, delta):
    number = re.match(r"^[+-]?[0-9.]+", delta)
    if number is None:
        return ""
    change = float(number[0])
    if change == 0:
        tone = "neutral"
    elif (change < 0 if key in ("occupancy", "average_wait") else change > 0):
        tone = "positive"
    else:
        tone = "negative"
    return f'<span class="kpi-delta-badge {tone}">{escape(delta)}</span>'


def select_dashboard_option(column, label, options, saved_key):
    widget_key = f"_{saved_key}"
    saved_value = st.session_state.get(saved_key)

    def save_selection():
        st.session_state[saved_key] = st.session_state[widget_key]

    return column.selectbox(
        label, options,
        index=options.index(saved_value) if saved_value in options else 0,
        format_func=lambda value: f"{label}  {value}" if label == "기준월" else value,
        label_visibility="collapsed", key=widget_key, filter_mode=None,
        on_change=save_selection,
    )


def render_berth_comparison(series):
    months = sorted({month for month, _, _ in series["occupancy"]}, reverse=True)
    berths = sorted({berth for _, berth, _ in series["occupancy"]})
    with st.container(key="kpi_dashboard", border=True):
        heading, month_col = st.columns([5, 1.4])
        heading.subheader("부두별 비교")
        month = select_dashboard_option(month_col, "기준월", months, "overview_month")
        monthly_occupancy = {
            berth: value
            for date, berth, value in series["occupancy"]
            if date == month
        }
        st.markdown("#### 1-5부두 월별 선석 점유율")
        st.caption(f"{month} · 해당 월 작업시간 합계를 부두별 가용 선석시간으로 나눈 값")
        st.plotly_chart(
            build_occupancy_bar(
                [(berth, monthly_occupancy.get(berth)) for berth in berths],
                "부두",
            ),
            key="overall_occupancy",
        )
        render_comparison_table(series, berths, month)


def render_comparison_table(series, berths, month):
    end = pd.Period(month, freq="M")
    prior_start, prior_end = end - 12, end - 1
    rows = []
    for terminal in berths:
        row = {"부두": terminal}
        for key, title, unit, _ in KPI_SPECS:
            value, _ = kpi_period(series, terminal, key, end, end)
            previous, _ = kpi_period(series, terminal, key, prior_start, prior_end)
            cell = escape(kpi_text(value, unit))
            if value is not None:
                delta = kpi_delta(key, value, previous)
                badge = delta_badge(key, delta)
                cell += " " + (badge or f'<span class="kpi-delta-badge neutral">{escape(delta)}</span>')
            row[title] = cell
        rows.append(row)
    st.caption(
        f"증감 기준 · 기준월 {month}의 값과 기준월을 제외한 직전 12개월 "
        f"({prior_start} ~ {prior_end})의 월평균을 비교. 대기시간은 연결된 입항 기록 기준."
    )
    headings = "".join(f"<th scope='col'>{escape(label)}</th>" for label in rows[0])
    body = "".join(
        f"<tr><th scope='row'>{escape(row['부두'])}</th>"
        + "".join(f"<td>{row[title]}</td>" for _, title, _, _ in KPI_SPECS)
        + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="kpi-comparison-wrap"><table class="kpi-comparison">'
        f"<thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )

def render_kpi_panel(series):
    months = sorted({month for month, _, _ in series["occupancy"]}, reverse=True)
    berths = sorted({berth for _, berth, _ in series["occupancy"]})
    with st.container(key="kpi_dashboard_detail", border=True):
        heading, berth_col, month_col = st.columns([4, 1.2, 1.4])
        berth = select_dashboard_option(berth_col, "부두", berths, "berth_detail_port")
        month = select_dashboard_option(month_col, "기준월", months, "berth_detail_month")
        with heading.container(key="kpi_panel_title"):
            st.subheader(f"신항 처리 효율 대시보드 · {berth}")
        end = pd.Period(month, freq="M")
        prior_start = end - 12
        prior_end = end - 1
        st.caption(f"{month} 월별 KPI · 직전 12개월 월평균과 비교 ({prior_start} ~ {prior_end}) · 대기시간은 연결된 입항 기록 기준 추정값")

        for col, (key, title, unit, color) in zip(st.columns(3, border=True), KPI_SPECS):
            with col:
                render_kpi_card(series, berth, month, key, title, unit, color)

        st.markdown(f"#### {berth} 선석별 점유율")
        st.caption(f"기준월 {month}")
        slot_rows = [
            [slot, value]
            for date, port, slot, value in series["monthly_slots"]
            if date == month and port == berth
        ]
        st.plotly_chart(build_occupancy_bar(slot_rows, "선석"), key="slot_occupancy")


def render_kpi_card(series, berth, month, key, title, unit, color):
    end = pd.Period(month, freq="M")
    prior_start, prior_end = end - 12, end - 1
    value, _ = kpi_period(series, berth, key, end, end)
    previous, _ = kpi_period(series, berth, key, prior_start, prior_end)
    delta = kpi_delta(key, value, previous)
    fig = build_kpi_chart(series, berth, key, title, color, month)
    with st.container(key=f"kpi_card_{key}"):
        st.markdown(f"#### **{title}**")
        st.markdown(f"## {kpi_text(value, unit)}")
        detail = f"직전 12개월 평균 대비 {delta}" if delta.startswith(("+", "-")) else delta
        if key == "occupancy" and value is not None:
            status = occupancy_status(value)
            tone = {"유지": "positive", "주의": "caution", "위험": "negative"}[status]
            st.caption(
                f'<span class="kpi-delta-badge occupancy-status {tone}">{status}</span>'
                f" · {escape(detail)}",
                unsafe_allow_html=True,
            )
        else:
            st.caption(detail)
        st.markdown(f'<div class="kpi-accent" style="background:{color}"></div>',
                    unsafe_allow_html=True)
        if fig is not None:
            st.plotly_chart(fig, key=f"mini_{key}")
            if key == "occupancy":
                st.caption(
                    f"유지 {OCCUPANCY_CAUTION:g}% 미만 · "
                    f"주의 {OCCUPANCY_CAUTION:g}% 이상 {OCCUPANCY_RISK:g}% 미만 · "
                    f"위험 {OCCUPANCY_RISK:g}% 이상"
                )
        else:
            st.caption("선택 기간의 월별 자료가 없습니다.")

def render_dashboard(series):
    overview, detail = st.tabs(("전체", "부두별"), key="dashboard_tabs")
    with overview:
        if st.session_state.get("eda_overview_ai_summary") or "eda_summary_error" in st.session_state:
            render_ai_summary(series)
        else:
            st.fragment(render_ai_summary, run_every="1s")(series)
        render_berth_comparison(series)
    with detail:
        render_kpi_panel(series)


def main():
    st.set_page_config(page_title="부산 신항 처리 효율 개선 방안", page_icon="⚓", layout="wide")
    st.markdown(f"<style>{(ROOT / 'styles.css').read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

    try:
        series = st.cache_data(read_dashboard_data)(DATA_SNAPSHOT, DATA_SNAPSHOT.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as exc:
        st.error("대시보드 데이터를 읽을 수 없습니다.")
        st.caption(str(exc))
        st.stop()

    st.markdown(PAGE_HEADER, unsafe_allow_html=True)

    render_dashboard(series)


if __name__ == "__main__":
    main()
