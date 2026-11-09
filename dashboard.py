import csv
import json
import os
import re
from html import unescape
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent
EDA = Path(os.getenv("BUSAN_EDA_DIR", ROOT / "EDA"))
NOTEBOOK = EDA / "dashboard0.ipynb"
DATA_SNAPSHOT = ROOT / "data" / "dashboard_snapshot.csv"
SNAPSHOT_METRICS = ("occupancy", "productivity", "average_wait", "monthly_slots")
SNAPSHOT_COLUMNS = ("metric", "month", "terminal", "berth", "value")


KPI_SPECS = (
    ("occupancy", "선석 점유율", "%", "#15803d"),
    ("productivity", "체류 생산성", " TEU/h", "#2563eb"),
    ("average_wait", "평균 대기시간", "시간", "#dc2626"),
)

OCCUPANCY_CAUTION = 75.9
OCCUPANCY_RISK = 80.9

RETRYABLE_GEMINI_ERRORS = [500, 502, 503, 504]


def read_dashboard_data(path, modified_time):
    series = {metric: [] for metric in SNAPSHOT_METRICS}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(SNAPSHOT_COLUMNS):
            raise ValueError("대시보드 CSV의 열 구성이 맞지 않습니다.")
        for row in reader:
            metric = row["metric"]
            value = float(row["value"]) if row["value"] else None
            point = [row["month"], row["terminal"]]
            if metric == "monthly_slots":
                point.append(row["berth"])
            series[metric].append([*point, value])

    if not all(series.values()):
        raise ValueError("대시보드 CSV에 필요한 지표가 빠져 있습니다.")
    return series


def export_dashboard_data(source, target):
    series = read_eda_outputs(source, source.stat().st_mtime_ns)["series"]
    target.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=target.parent, delete=False
    )
    temporary = Path(file.name)
    try:
        with file:
            writer = csv.writer(file)
            writer.writerow(SNAPSHOT_COLUMNS)
            for metric in SNAPSHOT_METRICS:
                for month, terminal, *details, value in series[metric]:
                    berth = details[0] if details else ""
                    writer.writerow([metric, month, terminal, berth, value])
        read_dashboard_data(temporary, temporary.stat().st_mtime_ns)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def read_eda_outputs(path, modified_time):
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    tables = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            markup = "".join(output.get("data", {}).get("text/html", []))
            header = re.search(r"<thead>(.*?)</thead>", markup, re.S)
            if header:
                columns = re.findall(r"<th[^>]*>(.*?)</th>", header[1], re.S)
                tables.append((columns[1:], markup))

    def read_table(title, columns):
        for headers, markup in tables:
            if set(columns) <= set(headers):
                body = re.search(r"<tbody>(.*?)</tbody>", markup, re.S)[1]
                rows = [
                    [unescape(value.strip()) for value in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
                    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S)
                ]
                if not rows or any("..." in row or "…" in row for row in rows):
                    raise ValueError(f"{title} 표의 중간 행이 생략됐어요. 수정된 노트북에서 해당 셀을 실행하고 저장해 주세요.")
                return pd.DataFrame(rows, columns=headers)[columns].replace("NaN", None)
        raise ValueError(f"{title} 표가 저장돼 있지 않아요. 해당 분석 셀을 실행하고 저장해 주세요.")

    waiting = read_table("대기시간", ["월", "부두", "평균대기시간", "대기시간계산건수"])
    waiting["대기시간계산건수"] = pd.to_numeric(waiting["대기시간계산건수"])
    waiting["대기시간합계"] = pd.to_numeric(waiting["평균대기시간"]) * waiting["대기시간계산건수"]
    waiting = waiting.dropna(subset=["대기시간합계"]).groupby(["월", "부두"], as_index=False)[
        ["대기시간합계", "대기시간계산건수"]
    ].sum()

    productivity = read_table("부두별 체류 생산성", ["부두", "년월", "선석수", "체류생산성 (천TEU/시간)"])
    overall = read_table("부두별 전체 점유율", ["부두명", "선석수", "평균선석점유율_퍼센트"])
    slots = read_table("선석별 전체 점유율", ["부두", "선석", "평균선석점유율_퍼센트"])
    monthly_slots = read_table(
        "월별 선석 점유율",
        ["연도", "월", "부두", "선석", "월총시간", "총작업시간_시간", "선석점유율_퍼센트"],
    )
    monthly_slots["총작업시간_시간"] = pd.to_numeric(monthly_slots["총작업시간_시간"])
    monthly = monthly_slots.groupby(["연도", "월", "부두", "월총시간"], as_index=False)[
        "총작업시간_시간"
    ].sum(min_count=1)
    berth_counts = {port: int(count) for port, count in zip(overall["부두명"], overall["선석수"])}
    years = monthly["연도"].astype(int)

    return {
        "period": f"{years.min()}~{years.max()}년",
        "series": {
            "average_wait": [
                [month, f"{port}부두", round(total / count, 2)]
                for month, port, total, count in waiting.itertuples(index=False, name=None)
                if count > 0
            ],
            "productivity": [
                [month, f"{port.removeprefix('port')}부두", round(float(value) * 1000, 2)]
                for port, month, _, value in productivity.dropna().itertuples(index=False, name=None)
            ],
            "occupancy": [
                [f"{year}-{int(month):02d}", f"{port}부두",
                 round(total / (float(hours) * berth_counts[f"{port}부두"]) * 100, 2)]
                for year, month, port, hours, total in monthly.dropna().itertuples(index=False, name=None)
            ],
            "overall_occupancy": [
                [port, None if pd.isna(value) else round(float(value), 2)]
                for port, _, value in overall.itertuples(index=False, name=None)
            ],
            "slots": [
                [f"{port}부두", f"{slot}선석", None if pd.isna(value) else round(float(value), 2)]
                for port, slot, value in slots.itertuples(index=False, name=None)
            ],
            "monthly_slots": [
                [f"{year}-{int(month):02d}", f"{port}부두", f"{slot}선석",
                 None if pd.isna(value) else round(float(value), 2)]
                for year, month, port, slot, value in monthly_slots[
                    ["연도", "월", "부두", "선석", "선석점유율_퍼센트"]
                ].itertuples(index=False, name=None)
            ],
        },
    }


def occupancy_status(value):
    if value >= OCCUPANCY_RISK:
        return "위험"
    if value >= OCCUPANCY_CAUTION:
        return "주의"
    return "유지"


def kpi_period(series, berth, key, start, end):
    points = [
        (month, value) for month, terminal, value in series.get(key, [])
        if terminal == berth and start <= pd.Period(month, freq="M") <= end
    ]
    points.sort()
    if len(points) != end.ordinal - start.ordinal + 1:
        return None, points
    return sum(value for _, value in points) / len(points), points


def kpi_delta(key, value, previous):
    if value is None or previous is None:
        return "직전 12개월 데이터 부족"
    if key == "productivity":
        return f"{(value / previous - 1) * 100:+.1f}%" if previous else "비교 자료 없음"
    unit = "%p" if key == "occupancy" else "시간"
    return f"{value - previous:+.2f}{unit}"


def build_kpi_chart(series, berth, key, title, color, month):
    points = sorted((m, v) for m, terminal, v in series[key] if terminal == berth)
    if not points:
        return None

    frame = pd.DataFrame(points, columns=["월", title])
    frame["월"] = pd.to_datetime(frame["월"], format="%Y-%m")
    fig = px.line(frame, x="월", y=title, color_discrete_sequence=[color], markers=True)
    fig.update_traces(line=dict(width=3, dash="solid"), marker=dict(size=4))
    fig.update_layout(height=300, margin=dict(l=12, r=16, t=14, b=12),
                      xaxis_title=None, yaxis_title=None, showlegend=False)
    fig.update_xaxes(tickformat="%Y.%m", nticks=4)
    fig.add_vline(x=pd.Timestamp(month), line_color=color, opacity=0.4)
    if key == "occupancy":
        fig.add_hrect(
            y0=OCCUPANCY_CAUTION, y1=OCCUPANCY_RISK,
            fillcolor="#F58518", opacity=0.3, line_width=0, layer="below",
        )
        fig.add_hrect(
            y0=OCCUPANCY_RISK, y1=100,
            fillcolor="#a61c33", opacity=0.3, line_width=0, layer="below",
        )
        fig.add_hline(
            y=OCCUPANCY_RISK, line_color="#536373", line_width=1.5, line_dash="dot",
        )
        fig.add_annotation(
            x=0.02, xref="paper", xanchor="left",
            y=OCCUPANCY_RISK, yref="y", yanchor="bottom", yshift=2,
            text=f"{OCCUPANCY_RISK:g}%", showarrow=False,
            font=dict(color="#344454"), bgcolor="white",
        )
        fig.update_yaxes(range=[40, 100], ticksuffix="%")
    return fig


def build_occupancy_bar(rows, x_label):
    frame = pd.DataFrame(rows, columns=[x_label, "선석 점유율"])
    fig = px.bar(
        frame, x=x_label, y="선석 점유율",
        text_auto=".2f", color_discrete_sequence=["#15803d"],
    )
    fig.update_layout(
        template="plotly_white", height=400, margin=dict(l=12, r=16, t=12, b=12),
        xaxis_title=None, yaxis_title="선석 점유율 (%)",
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    fig.update_traces(hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>")
    if x_label == "부두":
        port_colors = {
            "1부두": "#4C78A8",
            "2부두": "#F58518",
            "3부두": "#54A24B",
            "4부두": "#E45756",
            "5부두": "#B279A2",
        }
        fig.update_traces(marker_color=[port_colors[port] for port, _ in rows])
    elif x_label == "선석":
        palette = px.colors.qualitative.Plotly
        fig.update_traces(
            marker_color=[
                palette[(int(slot.removesuffix("선석")) - 1) % len(palette)]
                for slot, _ in rows
            ],
        )
    fig.update_traces(width=0.45)
    for label, value in rows:
        if value is None:
            fig.add_annotation(x=label, y=0, yshift=10, text="자료 없음", showarrow=False)
    return fig


def generate_ai_summary(key, series):
    data = {
        "부두별 선석 점유율 (월, 부두, %)": series["occupancy"],
        "부두별 체류 생산성 (월, 부두, TEU/h)": series["productivity"],
        "부두별 추정 대기시간 (월, 부두, 시간)": series["average_wait"],
        "선석별 점유율 (월, 부두, 선석, %)": series["monthly_slots"],
    }
    prompt = (
        "부산 신항 운영 대시보드 데이터를 한국어로 요약하세요. "
        "아래 월별 데이터를 근거로 체류 생산성, 선석 점유율, "
        "평균 추정 대기시간의 변화와 부두 간 차이를 설명하세요. "
        "1부두의 선석 점유율 결과를 우선하고, 월별 값과 전체 기간 값을 구분하세요. "
        "수치를 제시할 때는 부두, 기간, 단위를 함께 적으세요. "
        "확인되지 않은 수치나 원인은 추측하지 마세요. "
        "수치는 소수점 둘째 자리까지 표시하고 결측값(null)은 자료 없음으로 해석하세요. "
        "제목이나 서론 없이 최대 5개 항목으로 구성되도록 요약하세요. "
        "요약 문장 외의 다른 문장은 포함하지 마세요. "
        "각 항목은 '- '로 시작하는 한 문장으로 작성하세요."
        "\n\n대시보드 월별 데이터\n"
        + json.dumps(data, ensure_ascii=False)
    )

    http_options = types.HttpOptions(
        retry_options=types.HttpRetryOptions(
            attempts=2,
            initial_delay=2.0,
            http_status_codes=RETRYABLE_GEMINI_ERRORS,
        )
    )
    with genai.Client(api_key=key, http_options=http_options) as client:
        result = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

    summary = (result.text or "").strip()
    if not summary:
        raise ValueError("응답에 요약 텍스트가 없습니다.")
    return summary


if __name__ == "__main__":
    export_dashboard_data(NOTEBOOK, DATA_SNAPSHOT)
    print(f"대시보드 데이터 저장 완료: {DATA_SNAPSHOT}")
