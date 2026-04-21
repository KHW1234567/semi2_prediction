"""
streamlit_app.py — 마음 온도계 | 정신건강 AI 서비스
=====================================================
아키텍처:
  1. 일상 대화 (Chatbot)  — 관심사/성별/나이 설정 + 페르소나 챗봇
  2. 감정 분석 (NLP)      — 감정 분류 + 우울 점수 + 리포트
  3. 맞춤 케어 (Program)  — 유형별 케어 + 위기 상담 안내
  4. LSIS 자가검진        — 외로움·사회적 고립 척도 (후반 배치)

실행:
  streamlit run streamlit_app.py

필요 패키지:
  pip install streamlit plotly transformers torch
"""

import streamlit as st
import plotly.graph_objects as go
from model import load_model, get_depression_score, get_chatbot_response, INV_MAP

# ─────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="마음 온도계",
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

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}
.stApp { background: #F0F4FF; }

/* 헤더 */
.app-header {
    background: linear-gradient(135deg, #1e3a6e 0%, #2563EB 60%, #7C3AED 100%);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    color: white;
    margin-bottom: 1.5rem;
}
.app-title { font-size: 1.8rem; font-weight: 900; margin-bottom: 0.2rem; }
.app-sub   { font-size: 0.9rem; opacity: 0.85; }

/* 단계 뱃지 */
.step-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.2);
    border-radius: 99px; padding: 0.25rem 0.8rem;
    font-size: 0.82rem; font-weight: 700; color: white;
    margin-right: 0.4rem;
}

/* 카드 */
.card {
    background: white; border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 4px 20px rgba(37,99,235,0.08);
    border: 1px solid #E2E8F0;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #64748B; margin-bottom: 0.7rem;
}
.sec-header {
    font-size: 1rem; font-weight: 700;
    color: #1E293B; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.4rem;
}

/* 채팅 */
.chat-wrap {
    background: white; border-radius: 14px;
    border: 1px solid #E2E8F0;
    padding: 1rem; max-height: 420px;
    overflow-y: auto;
}
.bubble-user {
    display: flex; justify-content: flex-end; margin-bottom: 0.6rem;
}
.bubble-user-inner {
    background: #2563EB; color: white;
    padding: 0.6rem 1rem;
    border-radius: 16px 16px 4px 16px;
    max-width: 75%; font-size: 0.9rem;
    line-height: 1.55; word-break: break-word;
}
.bubble-bot {
    display: flex; align-items: flex-start;
    gap: 0.5rem; margin-bottom: 0.6rem;
}
.bot-avatar {
    width: 30px; height: 30px; background: #DBEAFE;
    border-radius: 50%; display: flex;
    align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0;
}
.bubble-bot-inner {
    background: #F1F5F9; color: #1E293B;
    padding: 0.6rem 1rem;
    border-radius: 4px 16px 16px 16px;
    max-width: 75%; font-size: 0.9rem;
    line-height: 1.6; word-break: break-word;
    white-space: pre-wrap;
}

/* 위기 박스 */
.crisis-box {
    background: #FFF5F5;
    border-left: 5px solid #DC2626;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.88rem; line-height: 1.8; color: #7F1D1D;
}

/* 케어 카드 */
.care-card {
    background: #F8FAFC; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    border-left: 4px solid #2563EB; font-size: 0.88rem;
}
.hotline {
    display: flex; align-items: center; gap: 0.6rem;
    background: #EFF6FF; border-radius: 10px;
    padding: 0.6rem 1rem; margin-bottom: 0.4rem;
    font-size: 0.88rem;
}
.hotline-num { font-weight: 900; font-size: 1rem; color: #2563EB; }

/* 감정 태그 */
.etag {
    display: inline-block; border-radius: 99px;
    padding: 0.15rem 0.6rem; font-size: 0.8rem;
    font-weight: 600; margin: 0.1rem;
}

/* 프로필 태그 */
.profile-tag {
    display: inline-block;
    background: #DBEAFE; color: #1D4ED8;
    border-radius: 99px; padding: 0.2rem 0.7rem;
    font-size: 0.82rem; font-weight: 600; margin: 0.15rem;
}

/* LSIS */
.lsis-result {
    background: linear-gradient(135deg,#EFF6FF,#F5F3FF);
    border-radius: 12px; padding: 1rem 1.3rem;
    margin-top: 0.8rem; border: 1px solid #BFDBFE;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1e3a6e,#2563EB) !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }

/* 빈 상태 */
.empty-state {
    text-align: center; padding: 3rem 1rem;
    color: #94A3B8; font-size: 0.9rem; line-height: 1.8;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#2563EB,#7C3AED);
    border: none; border-radius: 10px; color: white;
    font-weight: 700;
}
section[data-testid="stSidebar"] div.stButton > button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: rgba(255,255,255,0.7) !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
    padding: 0.2rem 0.5rem !important;
    margin-top: -0.1rem !important;
    margin-bottom: 0.35rem !important;
    font-weight: 400 !important;
    height: auto !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255,255,255,0.28) !important;
    color: white !important;
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
        "system": "너는 '철수'야. 유머 넘치는 친구야. 반말로 대화해.",
        "color": "#E74C3C",
    },
}

INTERESTS = ["영화", "음악", "독서", "운동", "게임", "요리", "여행", "반려동물", "그림/예술", "기타"]

HIGH_RISK = {"자살충동", "절망감", "죄책감", "우울감", "무기력"}

LSIS_QUESTIONS = [
    ("외로움",      "나는 외로움을 느낀다"),
    ("외로움",      "나는 혼자라는 느낌이 든다"),
    ("사회적지지",  "나를 이해해주는 사람들이 있다"),
    ("사회적지지",  "나는 필요할 때 도움을 요청할 수 있는 사람이 있다"),
    ("사회적관계망","나는 정기적으로 만나는 사람들이 있다"),
    ("사회적관계망","나는 내가 속한 모임이나 집단이 있다"),
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
    "양호": [
        ("🌟 긍정 활동 유지", "규칙적인 운동·사회적 교류 권장"),
        ("📖 자기계발 프로그램", "관심사 기반 활동 참여"),
        ("💚 일상 돌봄 유지", "정기적 자가점검 추천"),
    ],
}


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
    "lsis_done": False,
    "lsis_answers": [2, 2, 3, 3, 3, 3],
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


# ─────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💙 마음 온도계")
    st.markdown("**정신건강 AI 케어 서비스**")
    st.markdown("---")

    st.markdown("#### 📌 서비스 단계")
    pages = [
        ("💬 일상 대화",  "1단계", "챗봇 상담"),
        ("📊 감정 분석",  "2단계", "NLP 리포트"),
        ("🌱 맞춤 케어",  "3단계", "개인 솔루션"),
        ("📝 LSIS 검진",  "4단계", "자가검진"),
    ]
    for pname, step, desc in pages:
        active = st.session_state.page == pname
        bg     = "rgba(255,255,255,0.25)" if active else "rgba(255,255,255,0.07)"
        border = "2px solid rgba(255,255,255,0.8)" if active else "1px solid rgba(255,255,255,0.2)"
        weight = "700" if active else "400"
        prefix = "▶ " if active else "　"
        st.markdown(f"""
        <div style="background:{bg};border:{border};border-radius:10px;
                    padding:0.6rem 0.9rem;margin-bottom:0.3rem;">
            <div style="font-size:0.68rem;color:rgba(255,255,255,0.6);letter-spacing:0.05em;">{step}</div>
            <div style="font-weight:{weight};font-size:0.9rem;color:white;">{prefix}{pname}</div>
            <div style="font-size:0.73rem;color:rgba(255,255,255,0.65);margin-top:0.1rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"→ {pname}", key=f"nav_{pname}", use_container_width=True):
            st.session_state.page = pname
            st.rerun()

    st.markdown("---")

    # 페르소나 선택 (1단계에서만)
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
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.api_history = []
        st.session_state.last_analysis = None
        st.session_state.score_history = []
        st.rerun()


# ─────────────────────────────────────────────────
# 메인 헤더
# ─────────────────────────────────────────────────
page_info = {
    "💬 일상 대화": ("1단계", "일상 대화 · 상담형 챗봇"),
    "📊 감정 분석": ("2단계", "위험 신호 조기 탐지 · NLP 감정 분석"),
    "🌱 맞춤 케어": ("3단계", "고립 유형별 · 생애주기별 맞춤 케어"),
    "📝 LSIS 검진": ("4단계", "외로움 & 사회적 고립 척도 자가검진"),
}
step_label, step_desc = page_info[st.session_state.page]

st.markdown(f"""
<div class="app-header">
    <div class="app-title">💙 마음 온도계</div>
    <div class="app-sub">사용자 입력 문장에서 우울·사회적 고립 위험 신호를 탐지하는 정신건강 AI 모델</div>
    <div style="margin-top:0.8rem;">
        <span class="step-badge">STEP {step_label[0]}</span>
        <span style="font-size:0.9rem; opacity:0.9;">{step_desc}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 1: 일상 대화 (Chatbot)
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "💬 일상 대화":

    col_chat, col_right = st.columns([1.6, 1], gap="large")

    with col_chat:

        # ── 프로필 설정 (처음 1회) ──────────────────
        if not st.session_state.profile_done:
            st.markdown('<div class="sec-header">👤 먼저 간단히 알려주세요!</div>', unsafe_allow_html=True)
            with st.form("profile_form"):
                nickname = st.text_input("닉네임 (선택)", placeholder="예: 민수, 지은...")
                col_a, col_b = st.columns(2)
                with col_a:
                    age_group = st.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대 이상"])
                with col_b:
                    gender = st.selectbox("성별", ["남성", "여성", "기타/비공개"])
                interests = st.multiselect(
                    "관심사 (복수 선택 가능)",
                    INTERESTS,
                    default=["영화"],
                )
                submitted = st.form_submit_button("대화 시작하기 💙", type="primary", use_container_width=True)

            if submitted:
                st.session_state.user_profile = {
                    "nickname": nickname,
                    "age_group": age_group,
                    "gender": gender,
                    "interests": interests,
                }
                st.session_state.profile_done = True
                st.rerun()

        else:
            # ── 프로필 태그 표시 ──────────────────────
            p = st.session_state.user_profile
            tags = ""
            if p.get("nickname"): tags += f'<span class="profile-tag">👤 {p["nickname"]}</span>'
            tags += f'<span class="profile-tag">🎂 {p.get("age_group","")}</span>'
            tags += f'<span class="profile-tag">⚧ {p.get("gender","")}</span>'
            for i in p.get("interests", []):
                tags += f'<span class="profile-tag">🎯 {i}</span>'
            st.markdown(f'<div style="margin-bottom:0.7rem;">{tags}</div>', unsafe_allow_html=True)

            # ── 페르소나 표시 ─────────────────────────
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

            # ── 채팅 화면 ─────────────────────────────
            st.markdown(render_chat(st.session_state.messages, p), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "메시지",
                    placeholder="오늘 어떤 하루였나요? 편하게 이야기해 보세요...",
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
                        )
                    except Exception as e:
                        bot_reply = f"죄송해요, 오류가 발생했어요. ({e})"

                st.session_state.messages.append({"role": "user",      "content": user_input.strip()})
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.session_state.api_history.append({"role": "user",      "content": user_input.strip()})
                st.session_state.api_history.append({"role": "assistant", "content": bot_reply})
                st.session_state.last_analysis = analysis
                turn = len(st.session_state.score_history) + 1
                st.session_state.score_history.append((turn, analysis["score"]))
                st.rerun()

            # ── 분석 결과 요약 ─────────────────────────
            if st.session_state.last_analysis:
                a = st.session_state.last_analysis
                st.markdown("---")
                st.markdown('<div class="sec-header">🔍 마지막 입력 분석</div>', unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("우울 점수", f"{a['score']}점")
                m2.metric("위험 등급", a['level'])
                m3.metric("주요 감정", a['top3'][0][0] if a['top3'] else "-")
                if a["multi"]:
                    tags_html = "".join(
                        f'<span class="etag" style="background:{"#FEE2E2" if e in HIGH_RISK else "#DBEAFE"};color:{"#991B1B" if e in HIGH_RISK else "#1D4ED8"};">{e}</span>'
                        for e in a["multi"]
                    )
                    st.markdown(f'<div style="margin-top:0.3rem;"><b style="font-size:0.83rem;">복합 감정:</b> {tags_html}</div>', unsafe_allow_html=True)

                # 감정 분석 페이지로 이동 유도
                if st.button("📊 상세 감정 분석 보기 →", type="primary"):
                    st.session_state.page = "📊 감정 분석"
                    st.rerun()

    # ── 오른쪽: 미니 게이지 ────────────────────────
    with col_right:
        st.markdown('<div class="card-title">📊 실시간 우울 위험 점수</div>', unsafe_allow_html=True)
        analysis = st.session_state.last_analysis
        score = analysis["score"] if analysis else 0
        level = analysis["level"] if analysis else "대화를 시작하세요"
        bar_color = (
            "#DC2626" if score >= 60 else
            "#D97706" if score >= 35 else
            "#EAB308" if score >= 15 else "#059669"
        )
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            number={"suffix": "점", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#CBD5E1"},
                "bar":  {"color": bar_color},
                "bgcolor": "white",
                "steps": [
                    {"range": [0,  15], "color": "#DCFCE7"},
                    {"range": [15, 35], "color": "#FEF9C3"},
                    {"range": [35, 60], "color": "#FFEDD5"},
                    {"range": [60,100], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": bar_color, "width": 3}, "thickness": 0.75, "value": score},
            },
            title={"text": level, "font": {"size": 14}},
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=15,r=15,t=35,b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("""
        <div style="font-size:0.74rem;color:#94A3B8;text-align:center;margin-top:-0.5rem;">
            🟢 양호(0~14) 🟡 경증(15~34) 🟠 중증(35~59) 🔴 고위험(60+)
        </div>""", unsafe_allow_html=True)

        # 점수 추이
        if len(st.session_state.score_history) >= 2:
            st.markdown('<div class="card-title" style="margin-top:1rem;">📈 점수 추이</div>', unsafe_allow_html=True)
            turns  = [h[0] for h in st.session_state.score_history]
            scores = [h[1] for h in st.session_state.score_history]
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=turns, y=scores, mode="lines+markers",
                line=dict(color="#2563EB", width=2),
                marker=dict(size=6),
                hovertemplate="입력 %{x}번: %{y}점<extra></extra>",
            ))
            fig_line.add_hline(y=60, line_dash="dot", line_color="#DC2626")
            fig_line.add_hline(y=35, line_dash="dot", line_color="#D97706")
            fig_line.update_layout(
                height=160, margin=dict(l=0,r=30,t=5,b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0,100], gridcolor="#F1F5F9"),
                xaxis=dict(gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # 고위험 위기 안내
        if analysis and ("고위험" in analysis["level"] or
                         any(e in HIGH_RISK for e in analysis.get("multi", []))):
            st.markdown("""
            <div class="crisis-box">
                🚨 <b>위기 상담 안내</b><br>
                지금 많이 힘드신가요?<br>혼자 감당하지 않아도 됩니다.<br><br>
                ☎ <b>자살예방 1393</b> (24시간)<br>
                ☎ <b>정신건강위기 1577-0199</b>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 2: 감정 분석 (NLP)
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "📊 감정 분석":

    if not st.session_state.score_history:
        st.markdown('<div class="empty-state">📊<br>1단계 일상 대화에서 대화를 시작하면<br>여기서 감정 분석 리포트를 확인할 수 있어요.</div>', unsafe_allow_html=True)
        if st.button("💬 일상 대화 시작하기", type="primary"):
            st.session_state.page = "💬 일상 대화"
            st.rerun()
    else:
        a = st.session_state.last_analysis
        scores_list = [h[1] for h in st.session_state.score_history]

        # 요약 지표
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 대화 횟수", f"{len(scores_list)}회")
        m2.metric("평균 위험 점수", f"{round(sum(scores_list)/len(scores_list),1)}점")
        m3.metric("최고 위험 점수", f"{max(scores_list)}점")
        m4.metric("현재 위험 등급", a["level"] if a else "-")

        st.markdown("---")
        dc1, dc2 = st.columns([1.5, 1], gap="large")

        with dc1:
            # 점수 추이 (전체)
            st.markdown('<div class="sec-header">📈 우울 위험 점수 추이</div>', unsafe_allow_html=True)
            turns  = [h[0] for h in st.session_state.score_history]
            scores = [h[1] for h in st.session_state.score_history]
            fig_full = go.Figure()
            for y0, y1, col, lab in [
                (0,15,"rgba(5,150,105,0.08)","양호"),
                (15,35,"rgba(234,179,8,0.08)","경증"),
                (35,60,"rgba(217,119,6,0.08)","중증"),
                (60,100,"rgba(220,38,38,0.08)","고위험"),
            ]:
                fig_full.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0,
                                   annotation_text=lab, annotation_position="left",
                                   annotation_font_size=9, annotation_font_color="#94A3B8")
            fig_full.add_trace(go.Scatter(
                x=turns, y=scores, mode="lines+markers",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=8, color=scores, colorscale="RdYlGn_r", cmin=0, cmax=100),
                hovertemplate="입력 %{x}번<br>점수: %{y}점<extra></extra>",
            ))
            fig_full.update_layout(
                height=260, margin=dict(l=0,r=60,t=10,b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0,100], gridcolor="#F1F5F9"),
                xaxis=dict(title="입력 번호", gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_full, use_container_width=True)

            # 대화 이력
            st.markdown('<div class="sec-header">📋 대화 이력</div>', unsafe_allow_html=True)
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            for i, (msg, sc) in enumerate(zip(user_msgs, scores_list)):
                st.markdown(f"""
                <div style="background:#F8FAFC;border-radius:8px;padding:0.5rem 0.9rem;
                            margin-bottom:0.3rem;font-size:0.84rem;border-left:3px solid #2563EB;">
                    <span style="color:#94A3B8;font-size:0.74rem;">#{i+1} 사용자 입력</span>
                    <span style="float:right;font-weight:700;color:#2563EB;">{sc}점</span><br>
                    {msg["content"][:80]}{"..." if len(msg["content"])>80 else ""}
                </div>
                """, unsafe_allow_html=True)

        with dc2:
            # 감정 분포 파이차트
            if a:
                st.markdown('<div class="sec-header">💭 감지된 감정 분포</div>', unsafe_allow_html=True)
                probs = a["probs"]
                sorted_idx = probs.argsort()[::-1]
                top6 = sorted_idx[:6]
                other = probs[sorted_idx[6:]].sum()
                labels = [INV_MAP[i] for i in top6] + (["기타"] if other > 0.01 else [])
                values = [round(float(probs[i])*100,1) for i in top6] + ([round(float(other)*100,1)] if other > 0.01 else [])
                colors = ["#2563EB","#7C3AED","#06B6D4","#059669","#D97706","#DC2626","#94A3B8"]
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values, hole=0.42,
                    marker_colors=colors[:len(labels)],
                    textinfo="label+percent", textfont_size=10,
                ))
                fig_pie.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
                                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

                # 감정 바차트
                st.markdown('<div class="sec-header">📊 감정별 확률</div>', unsafe_allow_html=True)
                top8 = sorted_idx[:8]
                emo_names = [INV_MAP[i] for i in top8]
                emo_vals  = [round(float(probs[i])*100,1) for i in top8]
                bar_colors = ["#DC2626" if n in HIGH_RISK else "#2563EB" for n in emo_names]
                fig_bar = go.Figure(go.Bar(
                    x=emo_vals, y=emo_names, orientation="h",
                    marker_color=bar_colors,
                    hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                ))
                fig_bar.update_layout(
                    height=280, margin=dict(l=0,r=10,t=5,b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="#F1F5F9", title="%"),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # 맞춤 케어로 이동 유도
        st.markdown("---")
        st.info("💡 분석 결과를 바탕으로 맞춤 케어 프로그램을 확인해 보세요!")
        if st.button("🌱 맞춤 케어 보기 →", type="primary"):
            st.session_state.page = "🌱 맞춤 케어"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 3: 맞춤 케어 (Program)
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "🌱 맞춤 케어":

    a = st.session_state.last_analysis
    p = st.session_state.user_profile

    if not a:
        st.markdown('<div class="empty-state">🌱<br>먼저 1단계 일상 대화를 진행해 주세요.<br>감정 분석 결과를 바탕으로 맞춤 케어를 제안드립니다.</div>', unsafe_allow_html=True)
        if st.button("💬 일상 대화 시작하기", type="primary"):
            st.session_state.page = "💬 일상 대화"
            st.rerun()
    else:
        care_level = get_care_level(a["level"])
        age_group  = p.get("age_group", "")

        # 현재 상태 요약
        level_colors = {"고위험": "#FEE2E2", "중증": "#FFEDD5", "경증": "#FEF9C3", "양호": "#DCFCE7"}
        level_text   = {"고위험": "#991B1B",  "중증": "#9A3412",  "경증": "#854D0E",  "양호": "#065F46"}
        st.markdown(f"""
        <div style="background:{level_colors[care_level]};border-radius:12px;
                    padding:1rem 1.3rem;margin-bottom:1rem;
                    border-left:5px solid {level_text[care_level]};">
            <b style="color:{level_text[care_level]};font-size:1rem;">
                현재 위험 등급: {a["level"]}
            </b>
            <span style="color:{level_text[care_level]};font-size:0.88rem;margin-left:0.8rem;">
                우울 점수 {a["score"]}점 / 100점
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── 점수 추이 상세 표 ─────────────────────────
        scores_list = [h[1] for h in st.session_state.score_history]
        if scores_list:
            st.markdown('<div class="sec-header">📈 우울 점수 추이 상세</div>', unsafe_allow_html=True)

            # 추이 그래프
            turns = [h[0] for h in st.session_state.score_history]
            fig_trend = go.Figure()
            for y0, y1, col, lab in [
                (0,15,"rgba(5,150,105,0.08)","양호"),
                (15,35,"rgba(234,179,8,0.08)","경증"),
                (35,60,"rgba(217,119,6,0.08)","중증"),
                (60,100,"rgba(220,38,38,0.08)","고위험"),
            ]:
                fig_trend.add_hrect(y0=y0, y1=y1, fillcolor=col, line_width=0,
                                    annotation_text=lab, annotation_position="left",
                                    annotation_font_size=9, annotation_font_color="#94A3B8")
            fig_trend.add_trace(go.Scatter(
                x=turns, y=scores_list, mode="lines+markers+text",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=9, color=scores_list, colorscale="RdYlGn_r", cmin=0, cmax=100),
                text=[f"{s}점" for s in scores_list],
                textposition="top center",
                textfont=dict(size=10),
                hovertemplate="입력 %{x}번<br>점수: %{y}점<extra></extra>",
            ))
            fig_trend.update_layout(
                height=260, margin=dict(l=0,r=60,t=20,b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0,100], gridcolor="#F1F5F9", title="위험 점수"),
                xaxis=dict(title="대화 횟수", gridcolor="#F1F5F9", dtick=1),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # 대화별 점수 상세 테이블
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            st.markdown('<div class="sec-header">📋 대화별 점수 상세</div>', unsafe_allow_html=True)
            for i, (msg, sc) in enumerate(zip(user_msgs, scores_list)):
                # 이전 대비 변화
                if i == 0:
                    delta_html = '<span style="color:#94A3B8;font-size:0.75rem;">첫 번째</span>'
                else:
                    diff = sc - scores_list[i-1]
                    if diff > 0:
                        delta_html = f'<span style="color:#DC2626;font-size:0.75rem;">▲ +{diff:.1f}점</span>'
                    elif diff < 0:
                        delta_html = f'<span style="color:#059669;font-size:0.75rem;">▼ {diff:.1f}점</span>'
                    else:
                        delta_html = '<span style="color:#94A3B8;font-size:0.75rem;">— 변화없음</span>'

                # 위험 등급 색상
                sc_color = (
                    "#DC2626" if sc >= 60 else
                    "#D97706" if sc >= 35 else
                    "#EAB308" if sc >= 15 else "#059669"
                )
                sc_level = (
                    "🔴 고위험" if sc >= 60 else
                    "🟠 중증"   if sc >= 35 else
                    "🟡 경증"   if sc >= 15 else "🟢 양호"
                )
                st.markdown(f"""
                <div style="background:#F8FAFC;border-radius:10px;padding:0.65rem 1rem;
                            margin-bottom:0.4rem;border-left:4px solid {sc_color};">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="color:#94A3B8;font-size:0.75rem;font-weight:600;">#{i+1}번째 대화</span>
                        <span style="display:flex;gap:0.6rem;align-items:center;">
                            {delta_html}
                            <span style="font-weight:900;color:{sc_color};font-size:1rem;">{sc}점</span>
                            <span style="font-size:0.8rem;">{sc_level}</span>
                        </span>
                    </div>
                    <div style="font-size:0.85rem;color:#374151;margin-top:0.3rem;">
                        {msg["content"][:80]}{"..." if len(msg["content"])>80 else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 요약 통계
            avg_sc  = round(sum(scores_list)/len(scores_list), 1)
            max_sc  = max(scores_list)
            min_sc  = min(scores_list)
            trend   = scores_list[-1] - scores_list[0] if len(scores_list) > 1 else 0
            trend_txt = f"▲ {trend:+.1f}점 (악화)" if trend > 0 else (f"▼ {trend:.1f}점 (개선)" if trend < 0 else "변화없음")
            trend_color = "#DC2626" if trend > 0 else ("#059669" if trend < 0 else "#94A3B8")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#EFF6FF,#F5F3FF);border-radius:12px;
                        padding:0.9rem 1.2rem;margin-top:0.5rem;border:1px solid #BFDBFE;">
                <div style="font-weight:700;font-size:0.85rem;color:#1E293B;margin-bottom:0.5rem;">📊 종합 통계</div>
                <div style="display:flex;gap:1.5rem;flex-wrap:wrap;font-size:0.85rem;">
                    <span>평균 <b style="color:#2563EB;">{avg_sc}점</b></span>
                    <span>최고 <b style="color:#DC2626;">{max_sc}점</b></span>
                    <span>최저 <b style="color:#059669;">{min_sc}점</b></span>
                    <span>전체 추이 <b style="color:{trend_color};">{trend_txt}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── 양호 vs 우울 분기 ─────────────────────────
        if care_level == "양호":
            # 양호: 케어 프로그램 미표시, 긍정 메시지만
            st.markdown("""
            <div style="background:linear-gradient(135deg,#DCFCE7,#D1FAE5);border-radius:14px;
                        padding:1.5rem 1.8rem;text-align:center;border:1px solid #A7F3D0;">
                <div style="font-size:2rem;margin-bottom:0.5rem;">🌟</div>
                <div style="font-weight:900;font-size:1.1rem;color:#065F46;margin-bottom:0.4rem;">
                    현재 마음 상태가 안정적이에요!
                </div>
                <div style="font-size:0.9rem;color:#047857;line-height:1.7;">
                    우울 위험 점수가 낮은 상태입니다.<br>
                    지금처럼 규칙적인 일상과 사회적 교류를 유지해 주세요 😊<br><br>
                    <b>지속적인 자기관리 TIP</b><br>
                    규칙적인 운동 · 충분한 수면 · 좋아하는 활동 즐기기<br>
                    소중한 사람들과 대화 나누기
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # 경증/중증/고위험: 케어 프로그램 표시
            cc1, cc2 = st.columns([1, 1], gap="large")

            with cc1:
                st.markdown(f'<div class="sec-header">🎯 맞춤 케어 프로그램 ({age_group if age_group else "전 연령"})</div>', unsafe_allow_html=True)
                for pname, pdesc in CARE_PROGRAMS[care_level]:
                    st.markdown(f"""
                    <div class="care-card">
                        <b>{pname}</b><br>
                        <span style="font-size:0.83rem;color:#64748B;">{pdesc}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # 생애주기별 안내
                st.markdown('<div class="sec-header" style="margin-top:1rem;">📋 생애주기별 맞춤 안내</div>', unsafe_allow_html=True)
                lifecycle = {
                    "10대": "학교 상담실, 청소년 상담전화(1388), 또래 멘토링 프로그램",
                    "20대": "청년 불씨 프로젝트, 대학 상담센터, 마인드부스터 Green",
                    "30대": "직장인 EAP(근로자지원프로그램), 지역 정신건강복지센터",
                    "40대": "중년 심리지원 프로그램, 정신건강복지센터 상담",
                    "50대 이상": "노인 돌봄 서비스, AI 스피커 기반 독거노인 지원, 복지관 프로그램",
                }
                desc = lifecycle.get(age_group, "지역 정신건강복지센터 및 공공 상담 서비스 활용을 권장합니다.")
                st.markdown(f"""
                <div style="background:#F0F4FF;border-radius:10px;padding:0.8rem 1rem;
                            font-size:0.88rem;line-height:1.7;">
                    <b style="color:#2563EB;">{age_group if age_group else "맞춤"} 추천</b><br>
                    {desc}
                </div>
                """, unsafe_allow_html=True)

            with cc2:
                # 위기 상담 핫라인
                st.markdown('<div class="sec-header">🆘 위기 상담 핫라인</div>', unsafe_allow_html=True)
                hotlines = [
                    ("1393",      "자살예방상담전화",  "24시간 무료"),
                    ("1577-0199", "정신건강위기상담",  "24시간 운영"),
                    ("129",       "보건복지상담센터",  "복지 서비스 연계"),
                    ("1388",      "청소년상담전화",    "24시간 운영"),
                ]
                for num, name, desc in hotlines:
                    st.markdown(f"""
                    <div class="hotline">
                        <span style="font-size:1.1rem;">📞</span>
                        <span class="hotline-num">{num}</span>
                        <span><b>{name}</b><br>
                        <span style="font-size:0.78rem;color:#64748B;">{desc}</span></span>
                    </div>
                    """, unsafe_allow_html=True)

                # 관심사 기반 활동 추천
                if p.get("interests"):
                    st.markdown('<div class="sec-header" style="margin-top:1rem;">🎯 관심사 기반 활동</div>', unsafe_allow_html=True)
                    interest_care = {
                        "영화": "영화 동아리·시네마테라피 프로그램 참여",
                        "음악": "음악치료 프로그램, 합창단·밴드 소모임",
                        "독서": "독서치료, 북클럽·도서관 독서 모임",
                        "운동": "운동치료, 지역 스포츠 클럽·요가 클래스",
                        "게임": "디지털 힐링 프로그램, 보드게임 카페 소모임",
                        "요리": "요리 클래스, 푸드테라피 프로그램",
                        "여행": "치유 여행 프로그램, 지역 탐방 소모임",
                        "반려동물": "동물매개치료, 반려동물 커뮤니티 활동",
                        "그림/예술": "미술치료, 공방 클래스, 전시 관람 소모임",
                        "기타": "지역 커뮤니티 센터 프로그램 탐색",
                    }
                    for interest in p["interests"]:
                        care_desc = interest_care.get(interest, "관련 커뮤니티 활동 탐색")
                        st.markdown(f"""
                        <div style="background:#F0FDF4;border-radius:8px;padding:0.5rem 0.9rem;
                                    margin-bottom:0.3rem;font-size:0.85rem;border-left:3px solid #059669;">
                            <b style="color:#059669;">🎯 {interest}</b> — {care_desc}
                        </div>
                        """, unsafe_allow_html=True)

            # 고위험 강조 안내
            if care_level == "고위험":
                st.markdown("""
                <div class="crisis-box" style="margin-top:1rem;">
                    🚨 <b>즉각적인 도움이 필요합니다</b><br>
                    지금 바로 자살예방상담전화 <b>1393</b>에 전화하거나
                    가까운 정신건강복지센터를 방문해 주세요.<br>
                    혼자 감당하지 않아도 됩니다. 전문가가 도와드립니다.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.info("📝 더 정확한 고립 위험도 측정을 원하시면 LSIS 자가검진을 해보세요!")
        if st.button("📝 LSIS 자가검진 하기 →", type="primary"):
            st.session_state.page = "📝 LSIS 검진"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 4: LSIS 자가검진
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "📝 LSIS 검진":

    st.markdown("""
    <div class="card">
        <div class="card-title">LSIS란?</div>
        <div style="font-size:0.9rem;line-height:1.8;color:#374151;">
            <b>외로움과 사회적 고립 척도 (Loneliness and Social Isolation Scale)</b>는
            총 6문항으로 구성된 자기보고식 설문지입니다.
            한국 사회·문화 배경을 반영하여 개발되었으며,
            외로움과 사회적 고립 고위험군을 선별하기 위해 지역사회에서 쉽게 적용할 수 있습니다.<br><br>
            <span style="color:#64748B;font-size:0.82rem;">
                출처: 홍진표 et al. (2021). 외로움과 사회적 고립 척도의 개발 및 타당화 연구.
                신경정신의학, 60(4), 291-297.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">📝 최근 한 달간 자신의 상태를 체크해 주세요</div>', unsafe_allow_html=True)

    answer_labels = ["전혀 아니다 (1)", "가끔 그렇다 (2)", "자주 그렇다 (3)", "항상 그렇다 (4)"]
    answers = []

    categories = [
        ("😔 외로움 (1~2번)", 0, 2),
        ("🤝 사회적 지지 (3~4번)", 2, 4),
        ("👥 사회적 관계망 (5~6번)", 4, 6),
    ]

    for cat_label, start, end in categories:
        st.markdown(f'<div style="font-weight:700;color:#2563EB;font-size:0.88rem;margin:0.8rem 0 0.3rem;">{cat_label}</div>', unsafe_allow_html=True)
        for i in range(start, end):
            _, q = LSIS_QUESTIONS[i]
            val = st.select_slider(
                f"Q{i+1}. {q}",
                options=[1, 2, 3, 4],
                value=st.session_state.lsis_answers[i],
                format_func=lambda x: answer_labels[x - 1],
                key=f"lsis_{i}",
            )
            answers.append(val)

    if st.button("📊 결과 확인하기", type="primary", use_container_width=True):
        st.session_state.lsis_answers = answers
        st.session_state.lsis_done = True

        lonely_score  = answers[0] + answers[1]
        support_score = answers[2] + answers[3]
        network_score = answers[4] + answers[5]

        lonely_high   = lonely_score >= 3
        isolated_risk = support_score <= 4 and network_score <= 4

        st.markdown("---")
        st.markdown('<div class="sec-header">📋 LSIS 검진 결과</div>', unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("외로움", f"{lonely_score}점 / 8점",
                   delta="고위험" if lonely_high else "정상",
                   delta_color="inverse" if lonely_high else "normal")
        rc2.metric("사회적 지지", f"{support_score}점 / 8점",
                   delta="취약" if support_score <= 4 else "양호",
                   delta_color="inverse" if support_score <= 4 else "normal")
        rc3.metric("사회적 관계망", f"{network_score}점 / 8점",
                   delta="취약" if network_score <= 4 else "양호",
                   delta_color="inverse" if network_score <= 4 else "normal")

        st.markdown('<div class="lsis-result">', unsafe_allow_html=True)
        if lonely_high:
            st.warning("⚠️ **외로움 고위험군** — 외로움 점수가 3점 이상입니다. 정서적 연결이 필요한 상태일 수 있습니다.")
        else:
            st.success("✅ 외로움 점수 정상 범위입니다.")
        if isolated_risk:
            st.warning("⚠️ **사회적 고립 고위험군** — 사회적 지지와 관계망 점수가 낮습니다.")
        else:
            st.success("✅ 사회적 고립 위험이 낮습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 레이더 차트
        fig_radar = go.Figure(go.Scatterpolar(
            r=[lonely_score/8*100, support_score/8*100, network_score/8*100, lonely_score/8*100],
            theta=["외로움", "사회적 지지", "사회적 관계망", "외로움"],
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line_color="#2563EB",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=280, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40,r=40,t=30,b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        if lonely_high or isolated_risk:
            st.info("💡 맞춤 케어 페이지에서 외로움·고립 유형별 지원 서비스를 확인해 보세요.")
            if st.button("🌱 맞춤 케어 보기 →", type="primary"):
                st.session_state.page = "🌱 맞춤 케어"
                st.rerun()