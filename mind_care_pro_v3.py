"""
mind_care_pro.py — 마음 온도계 Pro | 정신건강 AI 서비스 (GPT-4o mini + 공공서비스 연계)
======================================================================================
기능:
  1. 기본 인적 정보 입력 (직업·거주지역·연락처 포함 강화)
  2. 사용자 입력문장 우울 관련 감정 추출 (챗봇)
  3. 사용자 입력문장 우울 정도 분석 (NLP)
  4. 사용자 우울 자가진단 (PHQ-9)
  5. 우울 위험도 3단계 맞춤 케어 (경도/중증/고위험)
  6. 상담사 대시보드 (대화 단위 위험도, 패턴, 우선 개입 대상)

실행:
  $env:OPENAI_API_KEY = "sk-..."
  streamlit run mind_care_pro.py
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from model import load_model, get_depression_score, get_chatbot_response, INV_MAP

# ─────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="마음 온도계 Pro",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.stApp { background: #F0F4FF; }

.app-header {
    background: linear-gradient(135deg, #1e3a6e 0%, #2563EB 60%, #7C3AED 100%);
    border-radius: 18px; padding: 1.6rem 2rem; color: white; margin-bottom: 1.5rem;
}
.app-title { font-size: 1.8rem; font-weight: 900; margin-bottom: 0.2rem; }
.app-sub   { font-size: 0.9rem; opacity: 0.85; }
.step-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.2); border-radius: 99px;
    padding: 0.25rem 0.8rem; font-size: 0.82rem; font-weight: 700; color: white; margin-right: 0.4rem;
}
.card {
    background: white; border-radius: 14px; padding: 1.3rem 1.5rem;
    box-shadow: 0 4px 20px rgba(37,99,235,0.08); border: 1px solid #E2E8F0; margin-bottom: 1rem;
}
.card-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 0.7rem; }
.sec-header { font-size: 1rem; font-weight: 700; color: #1E293B; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem; }
.chat-wrap { background: white; border-radius: 14px; border: 1px solid #E2E8F0; padding: 1rem; max-height: 420px; overflow-y: auto; }
.bubble-user { display: flex; justify-content: flex-end; margin-bottom: 0.6rem; }
.bubble-user-inner { background: #2563EB; color: white; padding: 0.6rem 1rem; border-radius: 16px 16px 4px 16px; max-width: 75%; font-size: 0.9rem; line-height: 1.55; word-break: break-word; }
.bubble-bot { display: flex; align-items: flex-start; gap: 0.5rem; margin-bottom: 0.6rem; }
.bot-avatar { width: 30px; height: 30px; background: #DBEAFE; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.9rem; flex-shrink: 0; }
.bubble-bot-inner { background: #F1F5F9; color: #1E293B; padding: 0.6rem 1rem; border-radius: 4px 16px 16px 16px; max-width: 75%; font-size: 0.9rem; line-height: 1.6; word-break: break-word; white-space: pre-wrap; }
.crisis-box { background: #FFF5F5; border-left: 5px solid #DC2626; border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.88rem; line-height: 1.8; color: #7F1D1D; }
.care-card { background: #F8FAFC; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; border-left: 4px solid #2563EB; font-size: 0.88rem; }
.hotline { display: flex; align-items: center; gap: 0.6rem; background: #EFF6FF; border-radius: 10px; padding: 0.6rem 1rem; margin-bottom: 0.4rem; font-size: 0.88rem; }
.hotline-num { font-weight: 900; font-size: 1rem; color: #2563EB; }
.service-link { display: block; background: #F8FAFC; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.4rem; border-left: 4px solid #7C3AED; font-size: 0.88rem; text-decoration: none; color: #1E293B; }
.service-link:hover { background: #EFF6FF; }
.crisis-btn {
    display: block; width: 100%; background: #DC2626;
    color: white; font-size: 1.4rem; font-weight: 900;
    border-radius: 14px; padding: 1rem 1.5rem; text-align: center;
    text-decoration: none; margin-bottom: 0.6rem; border: none;
    box-shadow: 0 4px 16px rgba(220,38,38,0.35);
    letter-spacing: 0.04em;
}
.crisis-btn:hover { background: #B91C1C; color: white; }
.risk-badge-high   { background:#FEE2E2; color:#991B1B; border:1px solid #FECACA; border-radius:99px; padding:0.2rem 0.8rem; font-size:0.82rem; font-weight:700; }
.risk-badge-mid    { background:#FFEDD5; color:#9A3412; border:1px solid #FED7AA; border-radius:99px; padding:0.2rem 0.8rem; font-size:0.82rem; font-weight:700; }
.risk-badge-low    { background:#FEF9C3; color:#854D0E; border:1px solid #FDE68A; border-radius:99px; padding:0.2rem 0.8rem; font-size:0.82rem; font-weight:700; }
.risk-badge-safe   { background:#DCFCE7; color:#065F46; border:1px solid #A7F3D0; border-radius:99px; padding:0.2rem 0.8rem; font-size:0.82rem; font-weight:700; }
.counselor-row { background:#F8FAFC; border-radius:10px; padding:0.6rem 1rem; margin-bottom:0.4rem; border-left:4px solid #7C3AED; font-size:0.85rem; display:flex; align-items:center; gap:0.6rem; }
.pattern-tag { display:inline-block; background:#FEE2E2; color:#991B1B; border-radius:6px; padding:0.15rem 0.55rem; font-size:0.78rem; font-weight:600; margin:0.1rem; }
.care-section { background:#F8FAFC; border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1rem; border:1px solid #E2E8F0; }
.care-section-title { font-size:1rem; font-weight:800; color:#1E293B; margin-bottom:0.8rem; }
.self-care-item { background:white; border-radius:8px; padding:0.6rem 1rem; margin-bottom:0.4rem; border-left:3px solid #059669; font-size:0.87rem; }
.notify-box { background:#EFF6FF; border-radius:12px; padding:1rem 1.2rem; border:1px solid #BFDBFE; font-size:0.88rem; color:#1D4ED8; line-height:1.8; margin-top:0.8rem; }

.profile-tag { display: inline-block; background: #DBEAFE; color: #1D4ED8; border-radius: 99px; padding: 0.2rem 0.7rem; font-size: 0.82rem; font-weight: 600; margin: 0.15rem; }
.lsis-result { background: linear-gradient(135deg,#EFF6FF,#F5F3FF); border-radius: 12px; padding: 1rem 1.3rem; margin-top: 0.8rem; border: 1px solid #BFDBFE; }
.empty-state { text-align: center; padding: 3rem 1rem; color: #94A3B8; font-size: 0.9rem; line-height: 1.8; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1e3a6e,#2563EB) !important; }
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    border: none; border-radius: 10px; color: white; font-weight: 700;
}
section[data-testid="stSidebar"] div.stButton > button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: rgba(255,255,255,0.7) !important;
    border-radius: 6px !important; font-size: 0.72rem !important;
    padding: 0.2rem 0.5rem !important; margin-top: -0.1rem !important;
    margin-bottom: 0.35rem !important; font-weight: 400 !important; height: auto !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255,255,255,0.28) !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────
PERSONAS = {
    "🧑‍⚕️ 상담사 지우": {
        "description": "따뜻하고 공감 잘 하는 전문 심리 상담사",
        "system": "당신은 '지우'라는 이름의 전문 심리 상담사입니다.",
        "color": "#4A90D9",
    },
    "👫 친구 민준": {
        "description": "편하게 털어놓을 수 있는 동네 친구",
        "system": "너는 '민준'이라는 이름의 오랜 친구야. 반말로 대화해.",
        "color": "#27AE60",
    },
    "🤖 AI 어시스턴트 클로": {
        "description": "차분하고 정확한 AI 어시스턴트",
        "system": "당신은 '클로'라는 AI 어시스턴트입니다.",
        "color": "#8E44AD",
    },
    "📚 멘토 선생님": {
        "description": "논리적이고 조언 잘 해주는 인생 멘토",
        "system": "당신은 경험이 풍부한 인생 멘토입니다.",
        "color": "#E67E22",
    },
    "😄 개그맨 철수": {
        "description": "유머러스하고 긍정 에너지 넘치는 친구",
        "system": """너는 '철수'야. 개그맨 스타일의 유머 넘치는 친구야. 반말로 대화해.
항상 긍정적이고 웃긴 방식으로 대화하되, 상대방의 감정을 무시하지 마.
말 끝에 가끔 개그 멘트나 말장난을 넣어줘. 예: '그건 진짜 웃픈 일이네 ㅋㅋ', '야 그거 레전드다', '아 진짜? ㄷㄷ'
이모지도 많이 써줘. 😂🤣😆
근데 상대가 진짜 힘들어 보이면 유머는 잠깐 내려놓고 진심으로 공감해줘.
절대 딱딱하거나 격식체 쓰지 마.""",
        "color": "#E74C3C",
    },
}

INTERESTS = ["영화", "음악", "독서", "운동", "게임", "요리", "여행", "반려동물", "그림/예술", "기타"]

OCCUPATIONS = ["학생(중고등)", "대학생/대학원생", "직장인", "자영업자", "주부", "무직/구직중", "기타"]
REGIONS = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "세종"]

HIGH_RISK = {"자살충동", "절망감", "죄책감", "우울감", "무기력"}

# PHQ-9 문항 (Patient Health Questionnaire-9)
PHQ9_QUESTIONS = [
    "기분이 가라앉거나 우울하거나 희망이 없다고 느꼈다",
    "평소 하던 일에 대한 흥미가 없어지거나 즐거움을 느끼지 못했다",
    "잠들기가 어렵거나 자주 깼다 (혹은 너무 많이 잤다)",
    "평소보다 식욕이 줄었다 (혹은 평소보다 많이 먹었다)",
    "다른 사람들이 눈치 챌 정도로 말과 행동이 느려졌다 (혹은 너무 안절부절 못했다)",
    "피곤하고 기운이 없었다",
    "내가 잘못했거나 실패했다는 생각이 들었다 (혹은 자신과 가족을 실망시켰다고 생각했다)",
    "신문을 읽거나 TV를 보는 것과 같은 일상적인 일에도 집중할 수가 없었다",
    "차라리 죽는 것이 더 낫겠다고 생각했다 (혹은 자해할 생각을 했다)",
]

# PHQ-9 심각도 기준
PHQ9_SEVERITY = [
    (0,  4,  "없음-최소",    "#DCFCE7", "#065F46", "추가 조치 불필요"),
    (5,  9,  "경증",         "#FEF9C3", "#854D0E", "추적 관찰 시 PHQ-9 재실시"),
    (10, 14, "보통",         "#FFEDD5", "#9A3412", "치료 계획·상담·추적 관찰·처방약 고려"),
    (15, 19, "중등도",       "#FEE2E2", "#991B1B", "처방약 처방 및 상담"),
    (20, 27, "극심한 우울",  "#FEE2E2", "#7F1D1D", "처방약 처방 및 정신건강 전문가 의뢰"),
]

CARE_PROGRAMS = {
    "고위험": [
        ("🆘 자살예방상담전화", "1393 (24시간 무료)"),
        ("🏥 정신건강위기상담전화", "1577-0199 (24시간)"),
        ("🏥 지역 정신건강복지센터", "전문 상담 및 치료 연계"),
    ],
    "중증": [
        ("💬 정신건강복지센터 상담", "지역 센터 방문 권장"),
        ("📱 마인드부스터 Green", "인지행동치료 기반 우울 개선 앱"),
        ("🤝 청년 불씨 프로젝트", "고립청년 회복 지원 (정서 워크숍·소모임)"),
    ],
    "경증": [
        ("📱 마인드부스터 Green", "대학생 우울 개선 디지털 프로그램"),
        ("🌱 사회참여 활동", "취향 기반 소모임·문화 체험 참여"),
        ("📞 보건복지상담센터", "129 (복지 서비스 연계)"),
    ],
}

# 시·도별 광역 정신건강복지센터 공식 링크
REGION_CENTERS = {
    "서울": ("서울시 광역정신건강복지센터 (블루터치)", "https://blutouch.net"),
    "부산": ("부산시 광역정신건강복지센터", "https://www.bsmind.or.kr"),
    "대구": ("대구시 광역정신건강복지센터", "https://www.daegumind.or.kr"),
    "인천": ("인천시 광역정신건강복지센터", "https://www.icmind.or.kr"),
    "광주": ("광주시 광역정신건강복지센터", "https://www.gjmind.or.kr"),
    "대전": ("대전시 광역정신건강복지센터", "https://www.djmind.or.kr"),
    "울산": ("울산시 광역정신건강복지센터", "https://www.usmind.or.kr"),
    "경기": ("경기도 광역정신건강복지센터", "https://www.mentalhealth.or.kr"),
    "강원": ("강원도 광역정신건강복지센터", "https://www.gwmind.or.kr"),
    "충북": ("충북 광역정신건강복지센터", "https://www.cbmind.or.kr"),
    "충남": ("충남 광역정신건강복지센터", "https://www.cnmind.or.kr"),
    "전북": ("전북 광역정신건강복지센터", "https://www.jbmind.or.kr"),
    "전남": ("전남 광역정신건강복지센터", "https://www.jnmind.or.kr"),
    "경북": ("경북 광역정신건강복지센터", "https://www.gbmind.or.kr"),
    "경남": ("경남 광역정신건강복지센터", "https://www.gnmind.or.kr"),
    "제주": ("제주도 광역정신건강복지센터", "https://www.jejumind.or.kr"),
    "세종": ("세종시 광역정신건강복지센터", "https://www.sjmind.or.kr"),
}

# 시군구 센터 검색은 국가정신건강정보포털 이용
SIGUNGU_SEARCH_URL = "https://www.mentalhealth.go.kr/portal/health/fac/PotalHealthFacListTab1.do"
NCMH_URL = "https://www.ncmh.go.kr"
BLUTOUCH_URL = "https://blutouch.net/facility/center"

# 지역사회 회복 지원 프로그램 (검증된 링크만)
COMMUNITY_PROGRAMS = [
    {
        "name": "📌 국립정신건강센터",
        "desc": "국가 정신건강 전문기관 — 진료·연구·교육·정신건강증진사업",
        "link": "https://www.ncmh.go.kr",
        "link_text": "공식 홈페이지 바로가기",
    },
    {
        "name": "📌 국가정신건강정보포털",
        "desc": "정신건강 정보 통합 제공 — 질병정보·자가진단·의료기관 검색",
        "link": "https://www.mentalhealth.go.kr",
        "link_text": "포털 바로가기",
    },
    {
        "name": "📌 복지로 — 복지서비스 찾기",
        "desc": "생애주기별 복지서비스 통합 검색 — 외로움·고립 지원 포함",
        "link": "https://www.bokjiro.go.kr",
        "link_text": "복지 서비스 찾기",
    },
    {
        "name": "📌 청년 불씨 프로젝트 (남양주시)",
        "desc": "고립청년 회복 지원 — 정서 워크숍·취향 소모임·일상회복 챌린지 (지역 문의 필요)",
        "link": "https://www.nyj.go.kr",
        "link_text": "남양주시 홈페이지",
    },
    {
        "name": "📌 마인드부스터 Green (앱스토어)",
        "desc": "대학생 우울 개선 — 인지행동치료 기반, 28회기, 개인 맞춤형 (앱 검색: 마인드부스터)",
        "link": "https://apps.apple.com/kr/app/%EB%A7%88%EC%9D%B8%EB%93%9C%EB%B6%80%EC%8A%A4%ED%84%B0/id1490078741",
        "link_text": "앱스토어 바로가기",
    },
]


# ─────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()


# ─────────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────────
defaults = {
    "page": "💬 일상 대화",
    "profile_done": False,
    "user_profile": {},
    "persona": "🧑‍⚕️ 상담사 지우",
    "messages": [],
    "api_history": [],
    "last_analysis": None,
    "score_history": [],
    "probs_history": [],   # 매 대화별 19개 감정 확률 누적
    "risk_history": [],    # (턴번호, level, score, top_emo) 기록 — 상담사 대시보드용
    "phq9_done": False,
    "phq9_answers": [0] * 9,
    "phq9_total": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v




# ─────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────
def render_chat(messages, profile):
    persona_name = st.session_state.persona.split(" ", 1)[1] if st.session_state.persona else "AI"
    html = '<div class="chat-wrap">'
    if not messages:
        name = profile.get("nickname", "")
        interests = ", ".join(profile.get("interests", []))
        html += f'''<div class="empty-state">💙<br>
        안녕하세요{" " + name + "님" if name else ""}! 저는 {persona_name}입니다.<br>
        {"관심사(" + interests + ")에 대해 " if interests else ""}편하게 이야기 나눠요 😊</div>'''
    for msg in messages:
        c = msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
        if msg["role"] == "user":
            html += f'<div class="bubble-user"><div class="bubble-user-inner">{c}</div></div>'
        else:
            html += f'<div class="bubble-bot"><div class="bot-avatar">💙</div><div class="bubble-bot-inner">{c}</div></div>'
    html += '</div>'
    return html


def get_care_level(level_str):
    if "고위험" in level_str: return "고위험"
    if "중증"   in level_str: return "중증"
    if "경증"   in level_str: return "경증"
    return "양호"


def get_avg_probs():
    """전체 대화의 감정별 평균 확률 계산 (일상 제외)"""
    if not st.session_state.probs_history:
        return None
    stacked = np.stack(st.session_state.probs_history, axis=0)
    avg = stacked.mean(axis=0)
    # 일상(index 11) 제외
    result = {}
    for i, name in INV_MAP.items():
        if i != 11:  # 일상 제외
            result[name] = round(float(avg[i]) * 100, 1)
    return result


# ─────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💙 마음 온도계")
    st.markdown("**정신건강 AI 케어 서비스**")
    st.markdown("---")

    st.markdown("#### 📌 서비스 단계")
    pages = [
        ("💬 일상 대화",    "1단계", "챗봇 상담"),
        ("📊 감정 분석",    "2단계", "NLP 리포트"),
        ("🌱 맞춤 케어",    "3단계", "개인 솔루션"),
        ("📝 PHQ-9 검진",   "4단계", "우울 자가검진"),
        ("🏥 공공 서비스",  "5단계", "기관 연계"),
        ("🩺 상담사 대시보드", "전문가", "위험도·패턴 분석"),
    ]
    for pname, step, desc in pages:
        active = st.session_state.page == pname
        bg     = "rgba(255,255,255,0.25)" if active else "rgba(255,255,255,0.07)"
        border = "2px solid rgba(255,255,255,0.8)" if active else "1px solid rgba(255,255,255,0.2)"
        weight = "700" if active else "400"
        prefix = "▶ " if active else "　"
        # 카드 + 버튼 합쳐서 클릭 가능하게
        clicked = st.button(
            f"{prefix}{step}  {pname}\n{desc}",
            key=f"nav_{pname}",
            use_container_width=True,
        )
        if clicked:
            st.session_state.page = pname
            st.rerun()

    st.markdown("---")

    if st.session_state.page == "💬 일상 대화":
        st.markdown("#### 🎭 대화 상대")
        for pname, pinfo in PERSONAS.items():
            short = pname.split(" ", 1)[1]
            icon  = pname.split()[0]
            if st.button(f"{icon} {short}", key=f"p_{pname}", use_container_width=True):
                if st.session_state.persona != pname:
                    st.session_state.persona = pname
                    st.session_state.messages = []
                    st.session_state.api_history = []
                    st.rerun()
            if st.session_state.persona == pname:
                st.markdown(f'<div style="font-size:0.72rem;opacity:0.8;margin-top:-0.5rem;margin-bottom:0.2rem;padding-left:0.3rem;">✓ {pinfo["description"]}</div>', unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("#### 🆘 위기 상담")
    st.markdown('<div style="font-size:0.82rem;line-height:2.1;">☎ 자살예방 <b>1393</b><br>☎ 정신건강위기 <b>1577-0199</b><br>☎ 복지상담 <b>129</b></div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ 대화 초기화", key="auto____대화_초기화_1", use_container_width=True):
        st.session_state.messages = []
        st.session_state.api_history = []
        st.session_state.last_analysis = None
        st.session_state.score_history = []
        st.session_state.probs_history = []
        st.session_state.risk_history = []
        st.rerun()


# ─────────────────────────────────────────────────
# 메인 헤더
# ─────────────────────────────────────────────────
page_info = {
    "💬 일상 대화":        ("1단계", "일상 대화 · 상담형 챗봇"),
    "📊 감정 분석":        ("2단계", "위험 신호 조기 탐지 · NLP 감정 분석"),
    "🌱 맞춤 케어":        ("3단계", "고립 유형별 · 생애주기별 맞춤 케어"),
    "📝 PHQ-9 검진":       ("4단계", "우울 자가검진 · PHQ-9"),
    "🏥 공공 서비스":      ("5단계", "지역별 정신건강 기관 연계"),
    "🩺 상담사 대시보드":  ("전문가", "대화 단위 위험도 · 개입 우선순위 분석"),
}
step_label, step_desc = page_info[st.session_state.page]

st.markdown(f"""
<div class="app-header">
    <div class="app-title">💙 마음 온도계 Pro</div>
    <div class="app-sub">우울·사회적 고립 위험 신호 탐지 · 3단계 맞춤 케어 · 상담사 대시보드</div>
    <div style="margin-top:0.8rem;">
        <span class="step-badge">STEP {step_label[0]}</span>
        <span style="font-size:0.9rem; opacity:0.9;">{step_desc}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 1: 일상 대화
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "💬 일상 대화":

    col_chat, col_right = st.columns([1.6, 1], gap="large")

    with col_chat:
        if not st.session_state.profile_done:
            st.markdown('<div class="sec-header">👤 먼저 간단히 알려주세요!</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.83rem;color:#64748B;margin-bottom:0.8rem;">입력 정보는 맞춤 케어 제공에만 활용됩니다. 민감한 정보는 선택 사항입니다.</div>', unsafe_allow_html=True)
            with st.form("profile_form"):
                col_n, col_region = st.columns(2)
                with col_n:
                    nickname = st.text_input("닉네임 (선택)", placeholder="예: 민수, 지은...")
                with col_region:
                    region = st.selectbox("거주 지역", ["선택 안함"] + REGIONS)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    age_group = st.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대 이상"])
                with col_b:
                    gender = st.selectbox("성별", ["남성", "여성", "기타/비공개"])
                with col_c:
                    occupation = st.selectbox("직업", OCCUPATIONS)

                interests = st.multiselect("관심사 (복수 선택 가능)", INTERESTS, default=["영화"])

                contact = st.text_input("비상 연락처 (선택 — 고위험 시 지인 알리기 기능에 활용)", placeholder="010-XXXX-XXXX")

                submitted = st.form_submit_button("대화 시작하기 💙", type="primary", use_container_width=True)
            if submitted:
                st.session_state.user_profile = {
                    "nickname":   nickname,
                    "age_group":  age_group,
                    "gender":     gender,
                    "occupation": occupation,
                    "region":     region if region != "선택 안함" else "",
                    "interests":  interests,
                    "contact":    contact,
                }
                st.session_state.profile_done = True
                st.rerun()

        else:
            p = st.session_state.user_profile

            # ── 프로필 태그 + 수정 버튼 ──────────────
            tags = ""
            if p.get("nickname"): tags += f'<span class="profile-tag">👤 {p["nickname"]}</span>'
            tags += f'<span class="profile-tag">🎂 {p.get("age_group","")}</span>'
            tags += f'<span class="profile-tag">⚧ {p.get("gender","")}</span>'
            if p.get("occupation"): tags += f'<span class="profile-tag">💼 {p["occupation"]}</span>'
            if p.get("region"):     tags += f'<span class="profile-tag">📍 {p["region"]}</span>'
            for i in p.get("interests", []):
                tags += f'<span class="profile-tag">🎯 {i}</span>'

            tag_col, btn_col = st.columns([5, 1])
            with tag_col:
                st.markdown(f'<div style="margin-bottom:0.3rem;">{tags}</div>', unsafe_allow_html=True)
            with btn_col:
                if st.button("✏️ 수정", key="edit_profile"):
                    st.session_state.profile_done = False
                    st.rerun()

            persona = PERSONAS[st.session_state.persona]
            picon   = st.session_state.persona.split()[0]
            pname   = st.session_state.persona.split(" ", 1)[1]
            st.markdown(f"""
            <div style="background:{persona['color']}18; border:1.5px solid {persona['color']}44;
                        border-radius:10px; padding:0.6rem 1rem; margin-bottom:0.7rem;
                        display:flex; align-items:center; gap:0.6rem;">
                <span style="font-size:1.3rem;">{picon}</span>
                <span style="font-weight:700; color:{persona['color']};">{pname}</span>
                <span style="color:#64748B; font-size:0.83rem;">— {persona['description']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(render_chat(st.session_state.messages, p), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "메시지", placeholder="오늘 어떤 하루였나요? 편하게 이야기해 보세요...",
                    height=80, label_visibility="collapsed",
                )
                c1, c2 = st.columns([5, 1])
                with c1:
                    submit = st.form_submit_button("전송 ➤", use_container_width=True, type="primary")
                with c2:
                    reset = st.form_submit_button("초기화", use_container_width=True)

            if reset:
                st.session_state.messages = []
                st.session_state.api_history = []
                st.session_state.last_analysis = None
                st.session_state.score_history = []
                st.session_state.probs_history = []
                st.session_state.risk_history = []
                st.rerun()

            if submit and user_input.strip():
                try:
                    model, tokenizer, inv_map, run_cfg, device = get_model()
                except Exception as e:
                    st.error(f"모델 로드 실패: {e}")
                    st.stop()

                analysis = get_depression_score(
                    text=user_input.strip(),
                    model=model, tokenizer=tokenizer, device=device, inv_map=inv_map,
                    max_len=run_cfg.get("max_len", 64),
                    threshold=run_cfg.get("multi_threshold", 3.0),
                )

                with st.spinner("답변 생성 중..."):
                    try:
                        bot_reply = get_chatbot_response(
                            user_text=user_input.strip(),
                            analysis=analysis,
                            conversation_history=st.session_state.api_history,
                            persona_system=persona["system"],
                        )
                    except Exception as e:
                        bot_reply = f"죄송해요, 오류가 발생했어요. ({e})"

                st.session_state.messages.append({"role": "user",      "content": user_input.strip()})
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.session_state.api_history.append({"role": "user",      "content": user_input.strip()})
                st.session_state.api_history.append({"role": "assistant", "content": bot_reply})
                st.session_state.last_analysis = analysis
                st.session_state.score_history.append((len(st.session_state.score_history)+1, analysis["score"]))
                # 감정 확률 누적 저장 (평균 계산용)
                st.session_state.probs_history.append(analysis["probs"])
                # 상담사 대시보드용 턴별 위험도 기록
                top_emos = sorted(
                    [(INV_MAP[i], float(analysis["probs"][i])) for i in range(len(analysis["probs"])) if i != 11],
                    key=lambda x: x[1], reverse=True
                )[:3]
                st.session_state.risk_history.append({
                    "turn":    len(st.session_state.risk_history) + 1,
                    "level":   analysis["level"],
                    "score":   analysis["score"],
                    "text":    user_input.strip()[:60],
                    "top_emos": top_emos,
                })
                st.rerun()

            # 고위험 감정 감지 시 위기 안내 (대화 중에만)
            a = st.session_state.last_analysis
            if a and (a["score"] >= 60 or any(e in HIGH_RISK for e in a.get("multi", []))):
                st.markdown("""
                <div class="crisis-box">
                    🚨 <b>위기 상담 안내</b><br>
                    지금 많이 힘드신가요? 혼자 감당하지 않아도 됩니다.<br><br>
                    ☎ <b>자살예방 1393</b> (24시간) &nbsp;|&nbsp; ☎ <b>정신건강위기 1577-0199</b>
                </div>
                """, unsafe_allow_html=True)

            # 감정 분석 이동 유도
            if st.session_state.probs_history:
                st.markdown("---")
                if st.button("📊 대화 감정 분석 보기 →", key="auto___대화_감정_분석_보기___1", type="primary"):
                    st.session_state.page = "📊 감정 분석"
                    st.rerun()

    # 오른쪽: 대화 중에는 간단한 안내만 표시
    with col_right:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:1.5rem;border:1px solid #E2E8F0;
                    text-align:center;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">💬</div>
            <div style="font-weight:700;font-size:0.95rem;color:#1E293B;margin-bottom:0.5rem;">
                대화 후 감정 분석
            </div>
            <div style="font-size:0.85rem;color:#64748B;line-height:1.7;">
                대화가 끝나면<br>
                <b>감정 분석 페이지</b>에서<br>
                19가지 감정별 평균 확률을<br>
                확인할 수 있어요 😊
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 위기 상담 핫라인 항상 표시
        st.markdown('<div class="sec-header">🆘 위기 상담 핫라인</div>', unsafe_allow_html=True)
        for num, name, desc in [
            ("1393", "자살예방상담전화", "24시간 무료"),
            ("1577-0199", "정신건강위기상담", "24시간 운영"),
            ("129", "보건복지상담센터", "복지 서비스 연계"),
            ("1388", "청소년상담전화", "24시간 운영"),
        ]:
            st.markdown(f"""
            <div class="hotline">
                <span style="font-size:1.1rem;">📞</span>
                <span class="hotline-num">{num}</span>
                <span><b>{name}</b><br>
                <span style="font-size:0.78rem;color:#64748B;">{desc}</span></span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 2: 감정 분석 — 대화 후 평균 확률 표시
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "📊 감정 분석":

    if not st.session_state.probs_history:
        st.markdown('<div class="empty-state">📊<br>1단계 일상 대화에서 대화를 시작하면<br>여기서 감정 분석 리포트를 확인할 수 있어요.</div>', unsafe_allow_html=True)
        if st.button("💬 일상 대화 시작하기", key="auto___일상_대화_시작하기_1", type="primary"):
            st.session_state.page = "💬 일상 대화"
            st.rerun()
    else:
        n_turns = len(st.session_state.probs_history)
        st.markdown(f'<div class="sec-header">💭 전체 대화 감정 분석 결과 (총 {n_turns}회 대화 기준)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#EFF6FF;border-radius:10px;padding:0.7rem 1rem;margin-bottom:1rem;
                    font-size:0.85rem;color:#1D4ED8;border-left:4px solid #2563EB;">
            💡 아래 결과는 전체 대화의 <b>감정별 평균 확률</b>입니다. 일상 감정은 제외되었습니다.
        </div>
        """, unsafe_allow_html=True)

        avg_probs = get_avg_probs()

        dc1, dc2 = st.columns([1.5, 1], gap="large")

        with dc1:
            # 19개 감정 평균 확률 바차트
            st.markdown('<div class="sec-header">📊 19가지 감정별 평균 확률</div>', unsafe_allow_html=True)

            # 확률 내림차순 정렬
            sorted_emotions = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
            emo_names = [e[0] for e in sorted_emotions]
            emo_vals  = [e[1] for e in sorted_emotions]
            bar_colors = ["#DC2626" if n in HIGH_RISK else "#2563EB" for n in emo_names]

            fig_bar = go.Figure(go.Bar(
                x=emo_vals, y=emo_names, orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in emo_vals],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                height=500, margin=dict(l=0, r=60, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#F1F5F9", title="평균 확률 (%)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # 대화 이력
            st.markdown('<div class="sec-header">📋 대화 이력</div>', unsafe_allow_html=True)
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            for i, msg in enumerate(user_msgs):
                st.markdown(f"""
                <div style="background:#F8FAFC;border-radius:8px;padding:0.5rem 0.9rem;
                            margin-bottom:0.3rem;font-size:0.84rem;border-left:3px solid #2563EB;">
                    <span style="color:#94A3B8;font-size:0.74rem;">#{i+1}번째 대화</span><br>
                    {msg["content"][:80]}{"..." if len(msg["content"])>80 else ""}
                </div>
                """, unsafe_allow_html=True)

        with dc2:
            # 파이차트 (상위 6개)
            st.markdown('<div class="sec-header">🥧 감정 분포 (상위 6개)</div>', unsafe_allow_html=True)
            top6 = sorted_emotions[:6]
            others_val = sum(e[1] for e in sorted_emotions[6:])
            pie_labels = [e[0] for e in top6] + (["기타"] if others_val > 0.5 else [])
            pie_vals   = [e[1] for e in top6] + ([round(others_val, 1)] if others_val > 0.5 else [])
            colors = ["#2563EB","#7C3AED","#06B6D4","#059669","#D97706","#DC2626","#94A3B8"]
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels, values=pie_vals, hole=0.42,
                marker_colors=colors[:len(pie_labels)],
                textinfo="label+percent", textfont_size=10,
            ))
            fig_pie.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                  paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

            # 주요 감정 카드
            st.markdown('<div class="sec-header">🏆 가장 많이 감지된 감정 Top 3</div>', unsafe_allow_html=True)
            for rank, (emo, val) in enumerate(sorted_emotions[:3], 1):
                color = "#DC2626" if emo in HIGH_RISK else "#2563EB"
                medal = ["🥇", "🥈", "🥉"][rank-1]
                st.markdown(f"""
                <div style="background:#F8FAFC;border-radius:10px;padding:0.7rem 1rem;
                            margin-bottom:0.4rem;border-left:4px solid {color};">
                    <span style="font-size:1.1rem;">{medal}</span>
                    <b style="color:{color};margin-left:0.4rem;">{emo}</b>
                    <span style="float:right;font-weight:700;color:{color};">{val:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

            # 고위험 감정 경고
            high_risk_detected = [(e, v) for e, v in sorted_emotions if e in HIGH_RISK and v > 5.0]
            if high_risk_detected:
                st.markdown("---")
                st.markdown('<div class="sec-header">⚠️ 주의 감정 감지</div>', unsafe_allow_html=True)
                for emo, val in high_risk_detected:
                    st.warning(f"**{emo}** 감정이 평균 **{val:.1f}%** 감지되었습니다.")
                st.markdown("""
                <div class="crisis-box">
                    🚨 전문 상담 연계를 권장합니다.<br>
                    ☎ <b>자살예방 1393</b> (24시간)
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🌱 맞춤 케어 보기 →", key="auto___맞춤_케어_보기___1", type="primary"):
            st.session_state.page = "🌱 맞춤 케어"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 3: 맞춤 케어
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "🌱 맞춤 케어":

    a = st.session_state.last_analysis
    p = st.session_state.user_profile

    if not a:
        st.markdown('<div class="empty-state">🌱<br>먼저 1단계 일상 대화를 진행해 주세요.</div>', unsafe_allow_html=True)
        if st.button("💬 일상 대화 시작하기", key="auto___일상_대화_시작하기_2", type="primary"):
            st.session_state.page = "💬 일상 대화"
            st.rerun()
    else:
        age_group = p.get("age_group", "")

        # PHQ-9 완료 여부 확인
        phq9_done  = st.session_state.phq9_done
        phq9_total = st.session_state.phq9_total if phq9_done else None

        # PHQ-9 점수 기반 케어 레벨 결정
        def get_phq9_care_level(score):
            if score is None:   return None
            if score >= 15:     return "고위험"
            elif score >= 10:   return "중증"
            elif score >= 5:    return "경증"
            else:               return "양호"

        care_level = get_phq9_care_level(phq9_total)

        if not phq9_done:
            # PHQ-9 미완료 시 안내
            st.markdown("""
            <div style="background:#EFF6FF;border-radius:14px;padding:1.5rem 1.8rem;
                        border:1px solid #BFDBFE;text-align:center;">
                <div style="font-size:1.5rem;margin-bottom:0.5rem;">📝</div>
                <div style="font-weight:700;font-size:1rem;color:#1D4ED8;margin-bottom:0.5rem;">
                    PHQ-9 우울 자가검진을 먼저 진행해 주세요
                </div>
                <div style="font-size:0.88rem;color:#64748B;line-height:1.7;">
                    맞춤 케어는 PHQ-9 검진 결과를 기반으로 제공됩니다.<br>
                    검진 후 이 페이지로 돌아오면 맞춤 케어를 확인할 수 있어요.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📝 PHQ-9 검진 하러 가기", type="primary", use_container_width=True):
                st.session_state.page = "📝 PHQ-9 검진"
                st.rerun()

        elif care_level == "양호":
            st.markdown(f"""
            <div style="background:#DCFCE7;border-radius:12px;padding:1rem 1.3rem;
                        margin-bottom:1rem;border-left:5px solid #065F46;">
                <b style="color:#065F46;font-size:1rem;">PHQ-9 점수: {phq9_total}점 — 정상 범위</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style="background:linear-gradient(135deg,#DCFCE7,#D1FAE5);border-radius:14px;
                        padding:1.5rem 1.8rem;text-align:center;border:1px solid #A7F3D0;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🌟</div>
                <div style="font-weight:900;font-size:1.1rem;color:#065F46;margin-bottom:0.4rem;">
                    현재 마음 상태가 안정적이에요!
                </div>
                <div style="font-size:0.9rem;color:#047857;line-height:1.7;">
                    지금처럼 규칙적인 일상과 사회적 교류를 유지해 주세요 😊<br><br>
                    규칙적인 운동 · 충분한 수면 · 좋아하는 활동 즐기기
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            cc1, cc2 = st.columns([1, 1], gap="large")

            with cc1:
                # ── 경도 위험 ─────────────────────────────────────────
                if care_level == "경증":
                    st.markdown(f"""
                    <div class="care-section">
                        <div class="care-section-title">🟡 경도 위험 — 주의 요망 (PHQ-9: {phq9_total}점)</div>
                        <div style="font-size:0.85rem;color:#854D0E;margin-bottom:1rem;line-height:1.7;">
                            PHQ-9 점수 5~9점 구간으로, 우울 증상이 일부 나타나고 있습니다.
                            자살 관련 표현은 없지만 꾸준한 자기관리가 필요합니다.
                        </div>
                        <div style="font-weight:700;font-size:0.88rem;color:#374151;margin-bottom:0.5rem;">🌙 자기관리 3가지 핵심</div>
                    """, unsafe_allow_html=True)
                    for item in [
                        ("😴 수면", "매일 같은 시간에 자고 일어나는 수면 리듬을 만드세요. 취침 1시간 전 스마트폰을 끄고, 7~8시간 숙면을 목표로 합니다."),
                        ("🍚 식사", "하루 세 끼를 규칙적으로 드세요. 식사를 거르면 기분 저하와 에너지 부족이 심해질 수 있습니다."),
                        ("🌅 일상 리듬", "기상 후 햇빛 쬐기, 짧은 산책, 좋아하는 활동 한 가지를 루틴으로 만드세요."),
                    ]:
                        st.markdown(f"""
                        <div class="self-care-item">
                            <b>{item[0]}</b><br>
                            <span style="color:#64748B;font-size:0.82rem;">{item[1]}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("""
                    <div class="care-section">
                        <div class="care-section-title">📞 온라인·전화 상담 기관</div>
                    """, unsafe_allow_html=True)
                    for name, num, desc in [
                        ("정신건강복지센터", "지역번호 + 1577-0199", "무료 전화 상담, 방문 연계"),
                        ("대학 상담센터", "재학 학교 홈페이지 확인", "재학생 무료, 비밀 보장"),
                        ("직장 EAP(근로자지원)", "HR 또는 사내 복지 포털", "직장인 무료 심리 상담 제도"),
                        ("마음이음 (온라인)", "www.mind-i.com", "24시간 채팅 상담 가능"),
                    ]:
                        st.markdown(f"""
                        <div style="background:white;border-radius:8px;padding:0.55rem 0.9rem;margin-bottom:0.35rem;border-left:3px solid #D97706;font-size:0.84rem;">
                            <b style="color:#D97706;">{name}</b> — {num}<br>
                            <span style="color:#64748B;font-size:0.79rem;">{desc}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ── 중증 위험 ─────────────────────────────────────────
                elif care_level == "중증":
                    st.markdown(f"""
                    <div class="care-section" style="border-left:4px solid #EA580C;">
                        <div class="care-section-title" style="color:#9A3412;">🟠 중등도 위험 — 전문가 상담 권고 (PHQ-9: {phq9_total}점)</div>
                        <div style="font-size:0.85rem;color:#9A3412;margin-bottom:1rem;line-height:1.7;">
                            PHQ-9 점수 10~14점 구간으로, 우울 증상이 상당히 나타나고 있습니다.
                            전문 상담이 필요한 상태입니다.
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div style="font-weight:700;font-size:0.88rem;color:#374151;margin-bottom:0.5rem;">🏥 정신건강복지센터 이용 방법</div>', unsafe_allow_html=True)
                    for step, desc in [
                        ("1. 전화 예약", "지역 정신건강복지센터에 전화(1577-0199)로 초기 상담 예약"),
                        ("2. 비용", "초기 상담 무료 / 추가 프로그램은 센터별 상이 (대부분 무료 또는 실비)"),
                        ("3. 운영 시간", "평일 09:00~18:00 기본 / 야간·주말 운영 여부는 각 센터 확인"),
                    ]:
                        st.markdown(f"""
                        <div class="self-care-item" style="border-left-color:#EA580C;">
                            <b>{step}</b> — {desc}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("""
                    <div class="care-section" style="border-left:4px solid #7C3AED;">
                        <div class="care-section-title">🤝 무료 집단 프로그램 안내</div>
                    """, unsafe_allow_html=True)
                    for prog in [
                        ("지역 자살예방센터 집단 프로그램", "각 시·도 자살예방센터에서 무료 집단 상담·마음건강 교육 운영"),
                        ("청년 불씨 프로젝트", "고립청년 회복 지원 — 정서 워크숍·취향 소모임 (남양주시 등 지역 문의)"),
                        ("마인드부스터 Green", "CBT 기반 우울 개선 앱 — 28회기 개인 맞춤형 (앱 검색: 마인드부스터)"),
                    ]:
                        st.markdown(f"""
                        <div style="background:white;border-radius:8px;padding:0.55rem 0.9rem;margin-bottom:0.35rem;border-left:3px solid #7C3AED;font-size:0.84rem;">
                            <b style="color:#7C3AED;">{prog[0]}</b><br>
                            <span style="color:#64748B;font-size:0.79rem;">{prog[1]}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # 구체적 시점 제안
                    st.markdown("""
                    <div style="background:#FFEDD5;border-radius:12px;padding:1rem 1.2rem;border-left:5px solid #EA580C;font-size:0.88rem;color:#7C2D12;line-height:1.8;">
                        ⏰ <b>이 상태가 2주 이상 지속된다면</b>, 더 이상 혼자 감당하지 마시고
                        반드시 정신건강복지센터나 정신건강의학과 전문의와 상의하세요.<br>
                        <span style="font-size:0.82rem;">※ 조기에 도움을 받을수록 회복이 빠릅니다.</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── 고위험 ────────────────────────────────────────────
                elif care_level == "고위험":
                    st.markdown(f"""
                    <div style="background:#FFF5F5;border-radius:14px;padding:1.3rem 1.5rem;border:2px solid #DC2626;margin-bottom:1rem;">
                        <div style="font-size:1rem;font-weight:900;color:#991B1B;margin-bottom:0.8rem;">🔴 고위험 — 즉각적 도움이 필요합니다 (PHQ-9: {phq9_total}점)</div>
                        <div style="font-size:0.87rem;color:#7F1D1D;margin-bottom:1rem;line-height:1.7;">
                            PHQ-9 점수 15점 이상으로, 심각한 우울 증상이 나타나고 있습니다.
                            <b>지금 바로 아래 번호로 연락해 주세요.</b>
                        </div>
                    """, unsafe_allow_html=True)
                    for num, name in [("1393", "자살예방상담전화"), ("1577-0199", "정신건강위기상담전화")]:
                        st.markdown(f'<a class="crisis-btn" href="tel:{num}">📞 {num} — {name} (24시간 무료)</a>', unsafe_allow_html=True)
                    st.markdown("""
                        <div style="font-size:0.85rem;color:#991B1B;font-weight:700;margin-top:0.8rem;line-height:1.7;">
                            📱 지금 이 앱보다 <u>사람과 직접 통화하는 것</u>이 훨씬 더 안전합니다.<br>
                            전화가 어렵다면 문자(#1393)나 카카오톡 채널 '마음이음'을 이용하세요.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 지인 알리기 기능
                    p_local = st.session_state.user_profile
                    contact = p_local.get("contact", "")
                    st.markdown("""
                    <div class="notify-box">
                        💌 <b>믿을 만한 친구나 가족에게 지금 힘든 상태라는 사실을 알려보세요.</b><br>
                        <span style="font-size:0.83rem;">혼자 감당하려 하지 않아도 됩니다.
                        아래 문장을 복사해서 보내보세요:</span><br><br>
                        <i style="font-size:0.84rem;color:#1D4ED8;">
                        "나 요즘 많이 힘들어. 잠깐 얘기할 수 있어?"</i>
                    </div>
                    """, unsafe_allow_html=True)
                    if contact:
                        st.markdown(f"""
                        <div style="background:#FEF9C3;border-radius:10px;padding:0.7rem 1rem;margin-top:0.5rem;font-size:0.85rem;color:#854D0E;border-left:4px solid #D97706;">
                            📱 등록된 비상 연락처: <b>{contact}</b><br>
                            <span style="font-size:0.79rem;">지금 위 메시지를 이 번호로 보내보세요.</span>
                        </div>
                        """, unsafe_allow_html=True)

            with cc2:
                st.markdown('<div class="sec-header">🆘 위기 상담 핫라인</div>', unsafe_allow_html=True)
                for num, name, desc in [
                    ("1393","자살예방상담전화","24시간 무료"),
                    ("1577-0199","정신건강위기상담","24시간 운영"),
                    ("129","보건복지상담센터","복지 서비스 연계"),
                    ("1388","청소년상담전화","24시간 운영"),
                ]:
                    st.markdown(f"""
                    <div class="hotline">
                        <span style="font-size:1.1rem;">📞</span>
                        <span class="hotline-num">{num}</span>
                        <span><b>{name}</b><br>
                        <span style="font-size:0.78rem;color:#64748B;">{desc}</span></span>
                    </div>
                    """, unsafe_allow_html=True)

                # 생애주기별 맞춤 안내
                st.markdown('<div class="sec-header" style="margin-top:1rem;">📋 생애주기별 맞춤 안내</div>', unsafe_allow_html=True)
                lifecycle = {
                    "10대":      "학교 상담실, 청소년 상담전화(1388), 또래 멘토링 프로그램",
                    "20대":      "청년 불씨 프로젝트, 대학 상담센터, 마인드부스터 Green",
                    "30대":      "직장인 EAP(근로자지원프로그램), 지역 정신건강복지센터",
                    "40대":      "중년 심리지원 프로그램, 정신건강복지센터 상담",
                    "50대 이상": "노인 돌봄 서비스, AI 스피커 기반 독거노인 지원, 복지관 프로그램",
                }
                desc_life = lifecycle.get(age_group, "지역 정신건강복지센터 및 공공 상담 서비스 활용을 권장합니다.")
                st.markdown(f"""
                <div style="background:#F0F4FF;border-radius:10px;padding:0.8rem 1rem;font-size:0.88rem;line-height:1.7;margin-bottom:0.8rem;">
                    <b style="color:#2563EB;">{age_group if age_group else "맞춤"} 추천</b><br>{desc_life}
                </div>
                """, unsafe_allow_html=True)

                if p.get("interests"):
                    st.markdown('<div class="sec-header">🎯 관심사 기반 활동</div>', unsafe_allow_html=True)
                    interest_care = {
                        "영화":"영화 동아리·시네마테라피 프로그램 참여",
                        "음악":"음악치료 프로그램, 합창단·밴드 소모임",
                        "독서":"독서치료, 북클럽·도서관 독서 모임",
                        "운동":"운동치료, 지역 스포츠 클럽·요가 클래스",
                        "게임":"디지털 힐링 프로그램, 보드게임 카페 소모임",
                        "요리":"요리 클래스, 푸드테라피 프로그램",
                        "여행":"치유 여행 프로그램, 지역 탐방 소모임",
                        "반려동물":"동물매개치료, 반려동물 커뮤니티 활동",
                        "그림/예술":"미술치료, 공방 클래스, 전시 관람 소모임",
                        "기타":"지역 커뮤니티 센터 프로그램 탐색",
                    }
                    for interest in p["interests"]:
                        care_desc = interest_care.get(interest, "관련 커뮤니티 활동 탐색")
                        st.markdown(f"""
                        <div style="background:#F0FDF4;border-radius:8px;padding:0.5rem 0.9rem;
                                    margin-bottom:0.3rem;font-size:0.85rem;border-left:3px solid #059669;">
                            <b style="color:#059669;">🎯 {interest}</b> — {care_desc}
                        </div>
                        """, unsafe_allow_html=True)


        st.markdown("---")
        if st.button("📝 PHQ-9 우울 자가검진 하기 →", type="primary", key="btn_care_to_phq"):
            st.session_state.page = "📝 PHQ-9 검진"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 4: 공공 서비스 연계
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "🏥 공공 서비스":

    sc1, sc2 = st.columns([1, 1], gap="large")

    with sc1:
        # 위기 상담 핫라인
        st.markdown('<div class="sec-header">🆘 위기 상담 핫라인</div>', unsafe_allow_html=True)
        for num, name, desc in [
            ("1393", "자살예방상담전화", "24시간 무료, 전국 어디서나"),
            ("1577-0199", "정신건강위기상담전화", "24시간 정신건강 위기 대응"),
            ("129", "보건복지상담센터", "복지 서비스 연계 및 상담"),
            ("1388", "청소년상담전화", "청소년 위기 상담 (24시간)"),
        ]:
            st.markdown(f"""
            <div class="hotline">
                <span style="font-size:1.2rem;">📞</span>
                <span class="hotline-num">{num}</span>
                <span><b>{name}</b><br>
                <span style="font-size:0.8rem;color:#64748B;">{desc}</span></span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 지역 정신건강복지센터 — 지도 검색으로 통일
        st.markdown('<div class="sec-header">🏥 내 지역 정신건강복지센터 찾기</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#F0F4FF;border-radius:12px;padding:1rem 1.2rem;
                    border:1px solid #BFDBFE;margin-bottom:0.8rem;font-size:0.88rem;line-height:1.7;">
            <b style="color:#1D4ED8;">📍 시·군·구까지 상세 검색 가능</b><br>
            <span style="color:#374151;">국가정신건강정보포털에서 내 지역 정신건강복지센터를<br>
            시·군·구 단위로 검색할 수 있어요.</span>
        </div>
        """, unsafe_allow_html=True)

        # 지도 검색 링크 (네이버/구글)
        # 네이버: 정신건강복지센터 검색 (현재 위치 기반)
        NAVER_MAP_URL = "https://map.naver.com/v5/search/%EC%A0%95%EC%8B%A0%EA%B1%B4%EA%B0%95%EB%B3%B5%EC%A7%80%EC%84%BC%ED%84%B0"
        GOOGLE_MAP_URL = "https://www.google.com/maps/search/%EC%A0%95%EC%8B%A0%EA%B1%B4%EA%B0%95%EB%B3%B5%EC%A7%80%EC%84%BC%ED%84%B0"

        st.markdown(f"""
        <div style="display:flex;flex-direction:column;gap:0.5rem;">
            <div style="font-size:0.82rem;color:#64748B;margin-bottom:0.2rem;">
                📍 지도에서 내 주변 센터 찾기
            </div>
            <div style="display:flex;gap:0.5rem;">
                <a href="{NAVER_MAP_URL}" target="_blank"
                   style="flex:1;display:block;background:#03C75A;color:white;padding:0.65rem 0.5rem;
                          border-radius:10px;text-decoration:none;font-size:0.88rem;
                          font-weight:700;text-align:center;">
                    🗺️ 네이버 지도 검색
                </a>
                <a href="{GOOGLE_MAP_URL}" target="_blank"
                   style="flex:1;display:block;background:#EA4335;color:white;padding:0.65rem 0.5rem;
                          border-radius:10px;text-decoration:none;font-size:0.88rem;
                          font-weight:700;text-align:center;">
                    🌍 구글 지도 검색
                </a>
            </div>
            <a href="{SIGUNGU_SEARCH_URL}" target="_blank"
               style="display:block;background:#2563EB;color:white;padding:0.65rem 1rem;
                      border-radius:10px;text-decoration:none;font-size:0.9rem;
                      font-weight:700;text-align:center;">
                🔍 시·군·구 정신건강복지센터 검색 (국가포털)
            </a>
            <a href="{NCMH_URL}" target="_blank"
               style="display:block;background:#059669;color:white;padding:0.65rem 1rem;
                      border-radius:10px;text-decoration:none;font-size:0.9rem;
                      font-weight:700;text-align:center;">
                🏛️ 국립정신건강센터 공식 홈페이지
            </a>
            <a href="https://www.mentalhealth.go.kr" target="_blank"
               style="display:block;background:#7C3AED;color:white;padding:0.65rem 1rem;
                      border-radius:10px;text-decoration:none;font-size:0.9rem;
                      font-weight:700;text-align:center;">
                💻 국가정신건강정보포털
            </a>
            <a href="https://www.bokjiro.go.kr" target="_blank"
               style="display:block;background:#D97706;color:white;padding:0.65rem 1rem;
                      border-radius:10px;text-decoration:none;font-size:0.9rem;
                      font-weight:700;text-align:center;">
                🤝 복지로 — 복지서비스 통합 검색
            </a>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        # 지역사회 회복 지원 프로그램
        st.markdown('<div class="sec-header">🌱 지역사회 회복 지원 프로그램</div>', unsafe_allow_html=True)
        for prog in COMMUNITY_PROGRAMS:
            st.markdown(f"""
            <div style="background:#F8FAFC;border-radius:12px;padding:0.9rem 1.1rem;
                        margin-bottom:0.6rem;border-left:4px solid #7C3AED;">
                <div style="font-weight:700;font-size:0.9rem;color:#1E293B;margin-bottom:0.3rem;">
                    {prog["name"]}
                </div>
                <div style="font-size:0.83rem;color:#64748B;margin-bottom:0.5rem;line-height:1.6;">
                    {prog["desc"]}
                </div>
                <a href="{prog["link"]}" target="_blank"
                   style="background:#7C3AED;color:white;padding:0.3rem 0.8rem;border-radius:6px;
                          text-decoration:none;font-size:0.8rem;font-weight:600;">
                    🔗 {prog["link_text"]}
                </a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # PHQ-9 기반 추천
        st.markdown('<div class="sec-header">🎯 PHQ-9 결과 기반 추천</div>', unsafe_allow_html=True)
        if st.session_state.phq9_done:
            phq9_total = st.session_state.phq9_total
            if phq9_total >= 15:
                st.error(f"🔴 고위험 (PHQ-9: {phq9_total}점) — 자살예방상담전화(1393)에 즉시 연락하세요.")
            elif phq9_total >= 10:
                st.warning(f"🟠 중등도 (PHQ-9: {phq9_total}점) — 지역 정신건강복지센터 방문을 권장합니다.")
            elif phq9_total >= 5:
                st.info(f"🟡 경도 (PHQ-9: {phq9_total}점) — 마인드부스터 Green 등 디지털 프로그램을 활용해 보세요.")
            else:
                st.success(f"🟢 정상 (PHQ-9: {phq9_total}점) — 긍정적인 사회 참여 활동을 유지하세요.")
        else:
            st.info("💡 PHQ-9 검진을 완료하면 결과 기반 맞춤 서비스를 추천해 드립니다.")

    st.markdown("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("📝 PHQ-9 우울 자가검진 하기 →", key="auto___PHQ_9_우울_자가검진_하기___1", type="primary", use_container_width=True):
            st.session_state.page = "📝 PHQ-9 검진"
            st.rerun()
    with col_b2:
        if st.button("🌱 맞춤 케어 보기 →", key="auto___맞춤_케어_보기___2", use_container_width=True):
            st.session_state.page = "🌱 맞춤 케어"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 5: PHQ-9 우울 자가검진
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "📝 PHQ-9 검진":

    st.markdown("""
    <div class="card">
        <div class="card-title">PHQ-9란?</div>
        <div style="font-size:0.9rem;line-height:1.8;color:#374151;">
            <b>우울 자가검사 (Patient Health Questionnaire-9, PHQ-9)</b>는
            총 9개 문항으로 구성된 자기보고식 설문지입니다.
            지난 2주간 우울증 증상(흥미 저하, 우울감, 수면 장애, 피로 등)의 빈도를 측정하며,
            총점(0~27점)이 높을수록 우울증 위험이 높습니다.
            <b>10점 이상이면 우울증을 의심하고 전문가 상담을 권장합니다.</b><br><br>
            ⚠️ <span style="color:#DC2626;">PHQ-9는 확진 도구가 아닌 선별 검사입니다.
            높은 점수가 나왔다면 반드시 정신건강의학과 전문의나 상담 센터 등 전문가의 도움을 받으세요.</span><br><br>
            <span style="color:#64748B;font-size:0.82rem;">
                측정 기간: 지난 2주간
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">📝 지난 2주간, 얼마나 자주 다음과 같은 문제들로 곤란을 겪으셨습니까?</div>', unsafe_allow_html=True)

    radio_options = ["없음 (0일)", "2~6일", "7~12일", "거의 매일"]
    answers = []

    for i, q in enumerate(PHQ9_QUESTIONS):
        if i == 8:
            st.markdown('<div style="background:#FFF5F5;border-radius:8px;padding:0.6rem 1rem;margin:0.8rem 0 0.3rem;border-left:4px solid #DC2626;font-size:0.83rem;color:#7F1D1D;">⚠️ 아래 문항은 민감한 내용을 포함합니다. 솔직하게 응답해 주세요.</div>', unsafe_allow_html=True)

        st.markdown(f'<div style="background:#F8FAFC;border-radius:10px;padding:0.7rem 1rem;margin-bottom:0.3rem;border-left:3px solid #2563EB;font-size:0.9rem;font-weight:500;">Q{i+1}. {q}</div>', unsafe_allow_html=True)

        saved_val = st.session_state.phq9_answers[i]
        val = st.radio(
            f"q{i+1}",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["0 — 없음 (0일)", "1 — 2~6일", "2 — 7~12일", "3 — 거의 매일"][x],
            index=saved_val,
            key=f"phq9_{i}",
            horizontal=True,
            label_visibility="collapsed",
        )
        answers.append(val)
        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    if st.button("📊 결과 확인하기", key="auto___결과_확인하기_1", type="primary", use_container_width=True):
        st.session_state.phq9_answers = answers
        total = sum(answers)
        st.session_state.phq9_total = total
        st.session_state.phq9_done = True

        severity_label = ""
        severity_bg    = ""
        severity_color = ""
        intervention   = ""
        for s_min, s_max, s_label, s_bg, s_color, s_inter in PHQ9_SEVERITY:
            if s_min <= total <= s_max:
                severity_label = s_label
                severity_bg    = s_bg
                severity_color = s_color
                intervention   = s_inter
                break

        is_high_risk_phq = total >= 10

        st.markdown("---")
        st.markdown('<div class="sec-header">📋 PHQ-9 검진 결과</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns(2)
        rc1.metric("PHQ-9 총점", f"{total}점 / 27점")
        rc2.metric("우울 심각도", severity_label)

        st.markdown(f"""
        <div style="background:{severity_bg};border-radius:12px;padding:1rem 1.3rem;
                    margin-top:0.5rem;border-left:5px solid {severity_color};">
            <b style="color:{severity_color};font-size:1rem;">{severity_label} ({total}점)</b><br>
            <span style="color:{severity_color};font-size:0.88rem;">권장 개입: {intervention}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">📊 PHQ-9 점수</div>', unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=total,
            number={"suffix": "점", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 27], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar":  {"color": severity_color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  4],  "color": "#DCFCE7"},
                    {"range": [5,  9],  "color": "#FEF9C3"},
                    {"range": [10, 14], "color": "#FFEDD5"},
                    {"range": [15, 19], "color": "#FEE2E2"},
                    {"range": [20, 27], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": severity_color, "width": 3}, "thickness": 0.75, "value": total},
            },
            title={"text": severity_label, "font": {"size": 16}},
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=40,b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("""
        <div style="font-size:0.78rem;color:#94A3B8;text-align:center;margin-top:-0.5rem;">
            🟢 없음(0~4) &nbsp; 🟡 경증(5~9) &nbsp; 🟠 보통(10~14) &nbsp; 🔴 중등도(15~19) &nbsp; 🔴 극심(20~27)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">📋 문항별 응답</div>', unsafe_allow_html=True)
        short_labels = ["우울감", "흥미 저하", "수면 장애", "식욕 변화", "행동 변화", "피로", "죄책감", "집중 곤란", "자해 생각"]
        fig_bar = go.Figure(go.Bar(
            x=short_labels, y=answers,
            marker_color=["#DC2626" if v >= 2 else "#2563EB" for v in answers],
            text=answers, textposition="outside",
        ))
        fig_bar.update_layout(
            height=260, margin=dict(l=0,r=0,t=10,b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0,3.5], tickvals=[0,1,2,3],
                       ticktext=["없음","2~6일","7~12일","거의 매일"], gridcolor="#F1F5F9"),
            xaxis=dict(gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # AI 모델과 비교
        if st.session_state.probs_history:
            avg_probs = get_avg_probs()
            depression_emos = ["우울감", "슬픔", "무기력", "절망감", "죄책감"]
            model_depression_score = sum(avg_probs.get(e, 0) for e in depression_emos)
            st.markdown("---")
            st.markdown('<div class="sec-header">🔗 AI 모델 감정 분석과의 비교</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            col_a.metric("PHQ-9 총점", f"{total}점 / 27점", severity_label)
            col_b.metric("AI 우울 관련 감정 확률 합", f"{model_depression_score:.1f}%", "대화 기반 추정")
            st.markdown(f"""
            <div style="background:#EFF6FF;border-radius:10px;padding:0.8rem 1rem;
                        font-size:0.85rem;color:#1D4ED8;border-left:4px solid #2563EB;margin-top:0.5rem;">
                💡 <b>PHQ-9 {total}점</b> ({severity_label})과
                <b>AI 모델 우울 관련 감정 {model_depression_score:.1f}%</b>를 함께 확인하여
                보다 정확한 심리 상태를 파악할 수 있습니다.
                {"<br>⚠️ 두 지표 모두 우울 위험 신호를 보이고 있습니다. 전문가 상담을 권장합니다." if is_high_risk_phq and model_depression_score > 20 else ""}
            </div>
            """, unsafe_allow_html=True)

        if answers[8] >= 1:
            st.markdown("""
            <div class="crisis-box" style="margin-top:1rem;">
                🚨 <b>즉각적인 도움이 필요합니다</b><br>
                자해나 자살에 대한 생각이 있으신가요? 혼자 감당하지 않아도 됩니다.<br><br>
                ☎ <b>자살예방상담전화 1393</b> (24시간, 무료)<br>
                ☎ <b>정신건강위기상담전화 1577-0199</b> (24시간)
            </div>
            """, unsafe_allow_html=True)
        elif is_high_risk_phq:
            st.warning("⚠️ PHQ-9 점수가 10점 이상입니다. 정신건강의학과 전문의 또는 상담 센터 방문을 권장합니다.")

        st.markdown("---")
        if st.button("🏥 공공 서비스 연계 보기 →", type="primary", key="btn_phq_to_service"):
            st.session_state.page = "🏥 공공 서비스"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# 상담사 대시보드
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "🩺 상담사 대시보드":

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a6e,#7C3AED);border-radius:14px;
                padding:1rem 1.5rem;color:white;margin-bottom:1.2rem;">
        <div style="font-size:1.1rem;font-weight:900;margin-bottom:0.3rem;">🩺 상담사 전용 대시보드</div>
        <div style="font-size:0.83rem;opacity:0.85;">대화 단위 위험도 · 우울·고립 표현 패턴 · 우선 개입 대상 확인</div>
    </div>
    """, unsafe_allow_html=True)

    rh = st.session_state.risk_history
    p  = st.session_state.user_profile

    if not rh:
        st.markdown('<div class="empty-state">🩺<br>아직 대화 데이터가 없습니다.<br>사용자가 1단계 일상 대화를 진행하면 여기서 분석 결과를 확인할 수 있습니다.</div>', unsafe_allow_html=True)
        if st.button("💬 일상 대화로 이동", type="primary"):
            st.session_state.page = "💬 일상 대화"
            st.rerun()
    else:
        # ── 상단 요약 지표 ─────────────────────────────────────
        total_turns = len(rh)
        high_turns  = sum(1 for r in rh if r["score"] >= 60 or any(e in HIGH_RISK for e in r.get("top_emos", [])))
        mid_turns   = sum(1 for r in rh if 35 <= r["score"] < 60)
        avg_score   = sum(r["score"] for r in rh) / total_turns if total_turns else 0
        max_score   = max(r["score"] for r in rh) if rh else 0

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("총 대화 턴", f"{total_turns}회")
        sm2.metric("🔴 고위험 턴", f"{high_turns}회", f"전체의 {high_turns/total_turns*100:.0f}%" if total_turns else "")
        sm3.metric("🟠 중등도 턴", f"{mid_turns}회")
        sm4.metric("평균 우울 점수", f"{avg_score:.1f}점")

        # 우선 개입 필요 여부 배너
        if high_turns > 0:
            st.markdown(f"""
            <div style="background:#FEF2F2;border:2px solid #DC2626;border-radius:12px;
                        padding:1rem 1.4rem;margin:0.8rem 0;font-size:0.9rem;color:#7F1D1D;line-height:1.8;">
                🚨 <b>우선 개입 대상</b> — 고위험 수준 발화가 <b>{high_turns}회</b> 감지되었습니다.
                즉각적인 상담 연계 또는 위기 대응이 필요합니다.<br>
                <span style="font-size:0.82rem;">☎ 자살예방상담전화 1393 / 정신건강위기상담 1577-0199</span>
            </div>
            """, unsafe_allow_html=True)
        elif mid_turns >= 2:
            st.warning(f"⚠️ 중증 위험 표현이 {mid_turns}회 감지되었습니다. 전문 상담 연계를 권고합니다.")

        dash_c1, dash_c2 = st.columns([1.4, 1], gap="large")

        with dash_c1:
            # ── 대화 단위 위험도 타임라인 ─────────────────────
            st.markdown('<div class="sec-header">📈 대화 단위 위험도 타임라인</div>', unsafe_allow_html=True)

            level_color_map = {"고위험": "#DC2626", "중증": "#EA580C", "경증": "#D97706", "양호": "#059669"}

            def score_to_color(score):
                if score >= 60:   return "#DC2626"
                elif score >= 35: return "#EA580C"
                elif score >= 15: return "#D97706"
                else:             return "#059669"

            turns_x  = [r["turn"] for r in rh]
            scores_y = [r["score"] for r in rh]
            colors_c = [score_to_color(r["score"]) for r in rh]

            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(
                x=turns_x, y=scores_y, mode="lines+markers",
                line=dict(color="#7C3AED", width=2),
                marker=dict(color=colors_c, size=10, line=dict(color="white", width=1.5)),
                hovertemplate="턴 %{x}<br>위험 점수: %{y:.2f}<extra></extra>",
                name="위험 점수",
            ))
            fig_timeline.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="대화 턴", gridcolor="#F1F5F9", dtick=1),
                yaxis=dict(title="위험 점수", gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # ── 대화 단위 위험도 테이블 ───────────────────────
            st.markdown('<div class="sec-header">📋 대화 단위 위험도 상세</div>', unsafe_allow_html=True)
            for r in rh:
                score = r["score"]
                if score >= 60:
                    badge = '<span style="background:#FEE2E2;color:#991B1B;border:1px solid #FECACA;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.82rem;font-weight:700;">🔴 고위험</span>'
                elif score >= 35:
                    badge = '<span style="background:#FFEDD5;color:#9A3412;border:1px solid #FED7AA;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.82rem;font-weight:700;">🟠 중등도</span>'
                elif score >= 15:
                    badge = '<span style="background:#FEF9C3;color:#854D0E;border:1px solid #FDE68A;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.82rem;font-weight:700;">🟡 경도</span>'
                else:
                    badge = '<span style="background:#DCFCE7;color:#065F46;border:1px solid #A7F3D0;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.82rem;font-weight:700;">🟢 양호</span>'
                top_emo_str = " · ".join([f"{e[0]}({e[1]*100:.0f}%)" for e in r["top_emos"][:2]])
                st.markdown(
                    f'<div style="background:#F8FAFC;border-radius:10px;padding:0.6rem 1rem;'
                    f'margin-bottom:0.4rem;border-left:4px solid #7C3AED;font-size:0.85rem;display:flex;align-items:center;gap:0.6rem;">'
                    f'<span style="font-weight:700;color:#7C3AED;min-width:40px;">#{r["turn"]}</span>'
                    f'{badge}'
                    f'<span style="color:#64748B;font-size:0.83rem;flex:1;">{r["text"]}{"..." if len(r["text"]) >= 60 else ""}</span>'
                    f'<span style="font-size:0.78rem;color:#94A3B8;">{top_emo_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with dash_c2:
            # ── 우울·고립 관련 표현 발생 패턴 ────────────────
            st.markdown('<div class="sec-header">🔍 위험 감정 발생 패턴</div>', unsafe_allow_html=True)

            # 전체 top 감정 집계
            all_emos: dict = {}
            for r in rh:
                for emo, prob in r["top_emos"]:
                    if emo not in all_emos:
                        all_emos[emo] = 0
                    all_emos[emo] += prob

            if all_emos:
                sorted_emos = sorted(all_emos.items(), key=lambda x: x[1], reverse=True)
                for emo, total_prob in sorted_emos[:8]:
                    is_risk = emo in HIGH_RISK
                    color   = "#DC2626" if is_risk else "#2563EB"
                    pct     = min(total_prob / (total_turns * 1.0) * 100, 100)
                    bar_w   = max(int(pct * 1.8), 4)
                    tag     = '<span style="background:#FEE2E2;color:#991B1B;border-radius:6px;padding:0.15rem 0.55rem;font-size:0.78rem;font-weight:600;margin:0.1rem;">⚠ 위험</span>' if is_risk else ""
                    st.markdown(
                        f'<div style="background:white;border-radius:8px;padding:0.5rem 0.8rem;'
                        f'margin-bottom:0.35rem;border-left:3px solid {color};">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.2rem;">'
                        f'<span style="font-weight:600;font-size:0.85rem;color:{color};">{emo}</span>'
                        f'{tag}'
                        f'<span style="font-size:0.78rem;color:#94A3B8;">{total_prob*100:.0f}% 누적</span>'
                        f'</div>'
                        f'<div style="background:#F1F5F9;border-radius:4px;height:5px;">'
                        f'<div style="background:{color};height:5px;border-radius:4px;width:{bar_w}%;max-width:100%;"></div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

            # ── 사용자 프로필 요약 ─────────────────────────────
            st.markdown('<div class="sec-header" style="margin-top:1rem;">👤 내담자 기본 정보</div>', unsafe_allow_html=True)
            if p:
                rows = [
                    ("닉네임", p.get("nickname", "—")),
                    ("연령대",  p.get("age_group", "—")),
                    ("성별",    p.get("gender",    "—")),
                    ("직업",    p.get("occupation","—")),
                    ("거주지",  p.get("region",    "—")),
                    ("관심사",  ", ".join(p.get("interests", [])) or "—"),
                    ("비상연락", p.get("contact", "—")),
                ]
                for label, val in rows:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;font-size:0.84rem;
                                padding:0.3rem 0;border-bottom:1px solid #F1F5F9;">
                        <span style="color:#64748B;">{label}</span>
                        <span style="font-weight:600;color:#1E293B;">{val}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#94A3B8;font-size:0.85rem;">프로필 정보 없음</span>', unsafe_allow_html=True)

            # PHQ-9 결과 요약
            if st.session_state.phq9_done:
                total_phq = st.session_state.phq9_total
                st.markdown(f"""
                <div style="background:#F0F4FF;border-radius:10px;padding:0.8rem 1rem;
                            margin-top:0.8rem;border-left:4px solid #2563EB;font-size:0.86rem;">
                    📋 <b>PHQ-9 결과</b>: {total_phq}점 /27점
                    {"— 🔴 전문가 의뢰 권장" if total_phq >= 15 else ("— 🟠 상담 권고" if total_phq >= 10 else ("— 🟡 주의 관찰" if total_phq >= 5 else "— 🟢 정상 범위"))}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("💬 대화 화면으로", use_container_width=True):
                st.session_state.page = "💬 일상 대화"
                st.rerun()
        with d2:
            if st.button("🌱 맞춤 케어 보기", type="primary", use_container_width=True):
                st.session_state.page = "🌱 맞춤 케어"
                st.rerun()