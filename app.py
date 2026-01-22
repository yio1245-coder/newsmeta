import os
import re
import tempfile
from collections import Counter
from urllib.request import urlopen

import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="뉴스 특성추출/키워드 탐색", layout="wide")

DATA_DEFAULT_PATH = "newsmeta.csv"


# -----------------------
# Font
# -----------------------
def find_malgun_font():
    """Try common Malgun Gothic paths (Windows)."""
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# -----------------------
# Google Drive helpers
# -----------------------
def extract_gdrive_file_id(url: str) -> str | None:
    """
    Accepts Google Drive share links such as:
      - https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
      - https://drive.google.com/open?id=<FILE_ID>
      - https://drive.google.com/uc?id=<FILE_ID>&export=download
    Returns FILE_ID or None.
    """
    if not url:
        return None

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)

    return None


def gdrive_direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_to_tempfile(url: str) -> str:
    """Stream-download to a tempfile and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()

    with urlopen(url) as r, open(tmp_path, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            f.write(chunk)

    return tmp_path


# -----------------------
# CSV reader with encoding fallback
# -----------------------
def read_csv_with_fallback(path_or_buffer, is_gz=False) -> pd.DataFrame:
    """
    Try multiple encodings to handle Korean CSVs (cp949/euc-kr).
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

    last_err = None
    for enc in encodings_to_try:
        try:
            if is_gz:
                return pd.read_csv(
                    path_or_buffer,
                    dtype=str,
                    keep_default_na=False,
                    compression="gzip",
                    encoding=enc,
                )
            else:
                return pd.read_csv(
                    path_or_buffer,
                    dtype=str,
                    keep_default_na=False,
                    encoding=enc,
                )
        except UnicodeDecodeError as e:
            last_err = e
            continue

    raise last_err


# -----------------------
# Loaders
# -----------------------
@st.cache_data(show_spinner=False)
def load_data_from_path(path_or_buffer):
    """
    Load CSV from local path or file-like object.
    Deduplicate fully-identical rows.
    Parse date -> year.
    """
    name = ""
    
    if hasattr(path_or_buffer, "name"):
        name = path_or_buffer.name.lower()
    elif isinstance(path_or_buffer, str):
        name = path_or_buffer.lower()

    is_gz = name.endswith(".gz")
    df = read_csv_with_fallback(path_or_buffer, is_gz=is_gz)

    before = len(df)
    df = df.drop_duplicates(keep="first")
    after = len(df)

    df["일자_dt"] = pd.to_datetime(df["일자"], errors="coerce")
    df["연도"] = df["일자_dt"].dt.year

    meta = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "bad_dates": int(df["일자_dt"].isna().sum()),
    }
    return df, meta


@st.cache_data(show_spinner=False)
def load_data_from_url(url: str):
    """
    Load CSV/CSV.GZ from a URL (e.g., Google Drive).
    Accepts Google Drive share link OR direct download link.
    Downloads to temp file first, then reads with encoding fallback.
    """
    used_url = url.strip()

    # Convert Google Drive share link to direct download link
    file_id = extract_gdrive_file_id(used_url)
    if file_id:
        used_url = gdrive_direct_download_url(file_id)

    tmp_path = download_to_tempfile(used_url)

    # Determine gzip by URL suffix (optional)
    is_gz = used_url.lower().endswith(".gz") or url.lower().endswith(".gz")

    try:
        df = read_csv_with_fallback(tmp_path, is_gz=is_gz)
    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    before = len(df)
    df = df.drop_duplicates(keep="first")
    after = len(df)

    df["일자_dt"] = pd.to_datetime(df["일자"], errors="coerce")
    df["연도"] = df["일자_dt"].dt.year

    meta = {
        "rows_before": before,
        "rows_after": after,
        "rows_removed": before - after,
        "bad_dates": int(df["일자_dt"].isna().sum()),
    }
    return df, meta, used_url


# -----------------------
# Search / co-occurrence utils
# -----------------------
def parse_terms(raw: str) -> list[str]:
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    # dedupe preserving order
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def filter_by_terms(df: pd.DataFrame, terms: list[str], mode: str) -> pd.Series:
    if not terms:
        return pd.Series(True, index=df.index)

    s = df["특성추출"].astype(str)
    masks = [s.str.contains(t, regex=False, na=False) for t in terms]

    if mode == "AND":
        m = masks[0]
        for mm in masks[1:]:
            m &= mm
        return m
    else:
        m = masks[0]
        for mm in masks[1:]:
            m |= mm
        return m


def tokenize_keywords(cell: str) -> set[str]:
    # Split by comma, strip, drop empties
    tokens = [t.strip() for t in str(cell).split(",") if t.strip()]
    return set(tokens)  # B: 기사 1건 내 중복 제거


def keyword_cooccurrence(df_filtered: pd.DataFrame, exclude_terms: set[str]) -> Counter:
    c = Counter()
    for cell in df_filtered["키워드"].astype(str):
        toks = tokenize_keywords(cell)
        if exclude_terms:
            toks = {t for t in toks if t not in exclude_terms}
        c.update(toks)
    return c


# -----------------------
# Sidebar: Data source
# -----------------------
st.sidebar.header("데이터 불러오기")
st.sidebar.caption(
    "GitHub에는 코드만 두고, 데이터는 Google Drive에 두는 방식입니다.\n"
    "권장: Google Drive URL로 로딩"
)

data_source = st.sidebar.radio(
    "데이터 소스",
    ["Google Drive URL (권장)", "파일 업로드", "로컬 파일(newsmeta.csv)"],
    index=0,
)

df = None
meta = None

if data_source == "Google Drive URL (권장)":
    url = st.sidebar.text_input("Google Drive 공유 링크 또는 직접 다운로드 링크", value="")
    st.sidebar.caption(
        "예: https://drive.google.com/file/d/FILE_ID/view?usp=sharing\n"
        "앱이 자동으로 직접 다운로드 링크로 변환합니다."
    )
    if not url:
        st.info("왼쪽 사이드바에 Google Drive 링크를 붙여넣어 주세요.")
        st.stop()

    try:
        with st.spinner("데이터 다운로드 및 로딩 중..."):
            df, meta, used_url = load_data_from_url(url)
        st.sidebar.success("로드 완료")
        st.sidebar.write("사용한 다운로드 URL:")
        st.sidebar.code(used_url)
    except Exception as e:
        st.sidebar.error("로드 실패")
        st.sidebar.write(str(e))
        st.stop()

elif data_source == "파일 업로드":
    uploaded = st.sidebar.file_uploader("CSV 업로드", type=["csv", "gz"])
    if uploaded is None:
        st.info("사이드바에서 CSV(또는 csv.gz)를 업로드해 주세요.")
        st.stop()

    try:
        df, meta = load_data_from_path(uploaded)
    except Exception as e:
        st.sidebar.error("업로드 파일 로딩 실패")
        st.sidebar.write(str(e))
        st.stop()

else:  # local
    if not os.path.exists(DATA_DEFAULT_PATH):
        st.error("newsmeta.csv 파일이 없어요. (로컬 파일 모드)")
        st.stop()

    df, meta = load_data_from_path(DATA_DEFAULT_PATH)


# -----------------------
# Sidebar: Data health
# -----------------------
st.sidebar.subheader("데이터 상태")
st.sidebar.write(f"- 원본 행 수: {meta['rows_before']:,}")
st.sidebar.write(f"- 중복 제거 후: {meta['rows_after']:,}")
st.sidebar.write(f"- 제거된 중복: {meta['rows_removed']:,}")
if meta["bad_dates"] > 0:
    st.sidebar.warning(f"날짜 파싱 실패: {meta['bad_dates']:,}건")


# -----------------------
# Sidebar: Search
# -----------------------
st.sidebar.header("검색")
raw_terms = st.sidebar.text_input("특성추출 검색어 (콤마로 구분)", "")
mode = st.sidebar.radio("검색 조건", ["OR", "AND"], horizontal=True)
terms = parse_terms(raw_terms)

# --- Year range (SAFE) ---
years = df["연도"].dropna()

if years.empty:
    st.sidebar.error("연도 정보를 만들 수 없어요. '일자' 형식/인코딩을 확인해주세요.")
    st.stop()

years = years.astype(int)
year_min = int(years.min())
year_max = int(years.max())

if year_min == year_max:
    year_range = (year_min, year_max)
    st.sidebar.info(f"데이터 연도가 {year_min}년 한 해만 있어요.")
else:
    year_range = st.sidebar.slider(
        "연도 범위",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )



# -----------------------
# Filtering
# -----------------------
mask_terms = filter_by_terms(df, terms, mode)
mask_year = df["연도"].between(year_range[0], year_range[1])
df_f = df.loc[mask_terms & mask_year].copy()


# -----------------------
# Main
# -----------------------
st.title("특성추출 기반 기사 분포 & 키워드 공동출현 (파일럿)")

c1, c2, c3 = st.columns(3)
c1.metric("검색 결과 기사 수", f"{len(df_f):,}")
c2.metric("연도 범위", f"{year_range[0]}–{year_range[1]}")
c3.metric("언론사 수", f"{df_f['언론사'].nunique():,}")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("연도별 기사 수")
    yearly = df_f.groupby("연도").size().reset_index(name="기사수").sort_values("연도")
    fig_year = px.bar(yearly, x="연도", y="기사수")
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader("언론사 분포 (상위 30)")
    outlet = (
        df_f.groupby("언론사")
        .size()
        .reset_index(name="기사수")
        .sort_values("기사수", ascending=False)
        .head(30)
    )
    fig_outlet = px.bar(outlet, x="언론사", y="기사수")
    fig_outlet.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_outlet, use_container_width=True)

with right:
    st.subheader("키워드 공동출현 (옵션 A)")
    topn = st.slider("Top N", 10, 50, 20, 5)
    exclude = st.toggle("공동출현에서 검색어 제외", value=True)
    exclude_terms = set(terms) if exclude else set()

    counter = keyword_cooccurrence(df_f, exclude_terms)
    co_df = pd.DataFrame(counter.most_common(topn), columns=["키워드", "빈도"])

    if not co_df.empty:
        fig_co = px.bar(co_df[::-1], x="빈도", y="키워드", orientation="h")
        st.plotly_chart(fig_co, use_container_width=True)
        st.dataframe(co_df, use_container_width=True, height=260)
    else:
        st.info("공동출현 결과가 없습니다. 검색어/연도 범위를 바꿔보세요.")

    st.subheader("워드클라우드")
    font_path = find_malgun_font()
    if font_path is None:
        st.warning("맑은고딕 폰트 경로를 찾지 못했어요. 한글이 깨질 수 있어요.")

    if counter:
        wc = WordCloud(
            font_path=font_path,
            width=900,
            height=520,
            background_color="white",
        ).generate_from_frequencies(dict(counter))

        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("워드클라우드를 만들 데이터가 없어요.")


st.subheader("기사 목록 (최대 300건)")
cols = ["일자", "연도", "언론사", "제목"]
st.dataframe(
    df_f[cols].sort_values(["일자", "언론사"], ascending=[False, True]).head(300),
    use_container_width=True,
    height=420,
)

with st.expander("현재 필터 상태 보기"):
    st.write({"terms": terms, "mode": mode, "year_range": year_range, "rows": len(df_f)})
