"""
goseumdochi.py — 고슴도치 | 정신건강 AI 서비스 (GPT-4o mini + 공공서비스 연계)
======================================================================================
기능:
  1. 기본 인적 정보 입력
  2. 사용자 입력문장 감정 추출 (챗봇)
  3. 사용자 입력문장 NLP 감정 분석 리포트 (전용 페이지)
  4. 사용자 우울 자가진단 (PHQ-9)
  5. 우울 위험도 3단계 맞춤 케어
  6. 공공 서비스 연계
  7. 상담사 대시보드

실행:
  $env:OPENAI_API_KEY = "sk-..."
  streamlit run goseumdochi.py
"""

import random
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from model import (
    load_model, get_depression_score, get_chatbot_response, INV_MAP,
    remove_stopwords_with_exceptions, count_valid_words,
)

try:
    import io, re
    from collections import Counter
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    _WORDCLOUD_OK = True
except Exception as _e:
    _WORDCLOUD_OK = False; _WORDCLOUD_ERR = str(_e)

try:
    from kiwipiepy import Kiwi as _Kiwi
    _kiwi_instance = _Kiwi()
    _KIWI_OK = True
except Exception:
    _kiwi_instance = None
    _KIWI_OK = False

# ── 입력 검증 기준 (Jin et al. 2023 + Chen 2021) ───────────────────
# Jin et al.(2023): 단일 발화 기반 분류의 한계 → 최소 3턴 이상 누적 필요
# Chen(2021): 전처리 후 평균 7.9단어에서 88% 이상 정확도 → 누적 10단어 기준
MIN_TURNS_FOR_ANALYSIS  = 3    # 최소 대화 턴 수
MIN_TURNS_FOR_WORDCLOUD = 3    # 워드클라우드 최소 턴
MIN_WORDS_PER_TURN      = 2    # 발화당 최소 유효 단어 (한국어 최소 명제 단위 기준: 내용어+서술어)
MIN_CHARS_FOR_WORDCLOUD = 50   # fallback용 (Kiwi 미설치 시)
MIN_CUMULATIVE_WORDS    = 10   # 누적 유효 단어 수 (Chen 2021의 7.9단어 기준 적용)

# 팔레트: merged3 색상 (블루·퍼플·그린 계열)
P_BLUE   = "#6096C8"
P_PURPLE = "#8B7BAD"
P_GREEN  = "#5B9E7A"
P_ORANGE = "#C4956B"
P_RED    = "#C07070"
P_TEAL   = "#5B9999"
CHART_PRIMARY_COLOR = P_BLUE
P_CLAY=P_BLUE; P_MAUVE=P_PURPLE; P_DARK="#4A7099"; P_BLUSH="#E8EFF7"
P_DEEP=P_RED; P_SAGE=P_GREEN; P_CORAL=P_BLUE; P_PEACH=P_PURPLE
P_ROSE=P_RED; P_PINK=P_ORANGE; P_WARM=P_RED

st.set_page_config(page_title="말랑해도 돼", page_icon="🫧", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
html,body,[class*="css"]{{font-family:'Noto Sans KR',sans-serif;}}
.stApp{{background:#F7F9FC;}}
.app-header{{background:linear-gradient(135deg,#5B82B5 0%,#7B9FD0 60%,#8B7BAD 100%);border-radius:16px;padding:1.4rem 2rem;color:white;margin-bottom:1.4rem;}}
.app-title{{font-size:1.7rem;font-weight:900;margin-bottom:0.2rem;}}
.app-sub{{font-size:0.88rem;opacity:0.88;}}
.step-badge{{display:inline-flex;align-items:center;background:rgba(255,255,255,0.22);border-radius:99px;padding:0.2rem 0.75rem;font-size:0.8rem;font-weight:700;color:white;margin-right:0.4rem;}}
.card{{background:white;border-radius:14px;padding:1.2rem 1.4rem;box-shadow:0 2px 10px rgba(96,150,200,0.07);border:1px solid #E5EEF7;margin-bottom:0.9rem;}}
.card-title{{font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#7A8FA6;margin-bottom:0.6rem;}}
.sec-header{{font-size:0.97rem;font-weight:700;color:#2D3748;margin-bottom:0.55rem;display:flex;align-items:center;gap:0.4rem;}}
.chat-wrap{{background:white;border-radius:14px;border:1px solid #E5EEF7;padding:1rem;max-height:440px;overflow-y:auto;}}
.bubble-user{{display:flex;justify-content:flex-end;margin-bottom:0.75rem;}}
.bubble-user-inner{{background:{P_BLUE};color:white;padding:0.65rem 1rem;border-radius:16px 16px 4px 16px;max-width:75%;font-size:0.9rem;line-height:1.55;word-break:break-word;}}
.bubble-bot{{display:flex;align-items:flex-start;gap:0.5rem;margin-bottom:0.75rem;}}
.bot-avatar{{width:28px;height:28px;background:#D8E8F4;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0;}}
.bubble-bot-inner{{background:#F3F7FC;color:#2D3748;padding:0.65rem 1rem;border-radius:4px 16px 16px 16px;max-width:75%;font-size:0.9rem;line-height:1.6;word-break:break-word;white-space:pre-wrap;}}
.input-highlight{{border:2px solid {P_PURPLE}55;border-radius:14px;padding:0.8rem;background:white;box-shadow:0 2px 12px rgba(139,123,173,0.12);margin-bottom:0.5rem;}}
.crisis-box{{background:#EEF4FF;border-left:4px solid {P_BLUE};border-radius:10px;padding:0.9rem 1.1rem;font-size:0.87rem;line-height:1.8;color:#2D4A7A;}}
.crisis-btn{{display:block;width:100%;background:linear-gradient(90deg,#6096C8,#8B7BAD);color:white !important;font-size:0.95rem;font-weight:700;border-radius:10px;padding:0.8rem 1.1rem;text-align:center;text-decoration:none !important;margin-bottom:0.4rem;border:none;box-shadow:0 3px 10px rgba(96,150,200,0.3);cursor:pointer;}}
.crisis-btn *{{color:white !important;text-decoration:none !important;}}
.crisis-btn:hover{{background:linear-gradient(90deg,#5088BB,#7B6BAD);color:white !important;transform:translateY(-1px);box-shadow:0 5px 14px rgba(96,150,200,0.4);}}
.hotline{{display:flex;align-items:center;gap:0.6rem;background:#F0F6FF;border-radius:10px;padding:0.55rem 0.9rem;margin-bottom:0.35rem;font-size:0.87rem;}}
.hotline-num{{font-weight:600;font-size:0.92rem;color:{P_BLUE};}}
.risk-badge-high{{background:#FFD8D8;color:#8B2020;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.8rem;font-weight:700;}}
.risk-badge-mid{{background:#FFE0D0;color:#8B3A20;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.8rem;font-weight:700;}}
.risk-badge-low{{background:#FFF0D8;color:#8B5A20;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.8rem;font-weight:700;}}
.risk-badge-safe{{background:#E8F5EF;color:#2A6049;border-radius:99px;padding:0.2rem 0.8rem;font-size:0.8rem;font-weight:700;}}
.care-section{{background:#F5F8FC;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:0.9rem;border:1px solid #E5EEF7;}}
.care-section-title{{font-size:0.97rem;font-weight:800;color:#2D3748;margin-bottom:0.7rem;}}
.self-care-item{{background:white;border-radius:8px;padding:0.55rem 0.9rem;margin-bottom:0.35rem;border-left:3px solid {P_GREEN};font-size:0.86rem;}}
.profile-tag{{display:inline-block;background:#D8E8F4;color:#1D4ED8;border-radius:99px;padding:0.18rem 0.65rem;font-size:0.8rem;font-weight:600;margin:0.12rem;}}
.empty-state{{text-align:center;padding:2.5rem 1rem;color:#A0AEC0;font-size:0.9rem;line-height:1.8;}}
.stat-card{{background:white;border-radius:12px;padding:0.8rem 1rem;border:1px solid #E5EEF7;margin-bottom:0.5rem;box-shadow:0 1px 6px rgba(96,150,200,0.06);}}
.stat-label{{font-size:0.76rem;color:#7A8FA6;margin-bottom:0.15rem;}}
.stat-value{{font-size:1.2rem;font-weight:800;color:#2D3748;}}
.cond-met{{background:#EAF5EF;border-left:4px solid {P_GREEN};border-radius:10px;padding:0.6rem 0.9rem;font-size:0.83rem;color:#2A6049;margin-bottom:0.5rem;}}
.cond-unmet{{background:#F5F8FC;border-left:4px solid #B8C8D8;border-radius:10px;padding:0.6rem 0.9rem;font-size:0.83rem;color:#5A7A96;margin-bottom:0.5rem;}}
.notify-box{{background:#EFF6FF;border-radius:12px;padding:0.9rem 1.1rem;border:1px solid #BFD8EE;font-size:0.87rem;color:#2D5A8E;line-height:1.8;margin-top:0.7rem;border-left:4px solid #6096C8;}}
.stTabs [data-baseweb="tab-list"]{{gap:0.5rem;border-bottom:2px solid #E5EEF7;}}
.stTabs [data-baseweb="tab"]{{background:transparent;border:none;border-radius:8px 8px 0 0;padding:0.5rem 1.2rem;font-size:0.88rem;color:#7A8FA6;font-weight:600;}}
.stTabs [aria-selected="true"]{{background:white;color:{P_BLUE};border-bottom:3px solid {P_BLUE};font-weight:700;}}
[data-baseweb="tag"]{{background-color:#D8E8F4 !important;border-radius:6px !important;border:none !important;}}
[data-baseweb="tag"] span{{color:#1D4ED8 !important;font-weight:500 !important;font-size:0.82rem !important;}}
div[data-testid="stTextArea"] > div > textarea{{border-color:#C8C8DC !important;border-radius:10px !important;outline:none !important;}}
div[data-testid="stTextArea"] > div > textarea:focus,div[data-testid="stTextArea"] > div > textarea:focus-visible{{border-color:{P_PURPLE} !important;outline:none !important;box-shadow:0 0 0 2px rgba(139,123,173,0.25) !important;}}
textarea:focus{{outline:none !important;border-color:{P_PURPLE} !important;box-shadow:0 0 0 2px rgba(139,123,173,0.25) !important;}}
[data-baseweb="textarea"] textarea{{border-color:#C8C8DC !important;}}
[data-baseweb="textarea"] textarea:focus{{border-color:{P_PURPLE} !important;box-shadow:0 0 0 2px rgba(139,123,173,0.25) !important;}}
section[data-testid="stSidebar"]{{background:linear-gradient(180deg,#5B82B5,#7B9FD0) !important;}}
section[data-testid="stSidebar"] *{{color:white !important;}}
section[data-testid="stSidebar"] hr{{border-color:rgba(255,255,255,0.2);}}
div.stButton > button{{border-radius:9px !important;font-weight:600 !important;transition:all 0.15s ease !important;}}
div.stButton > button[kind="primary"],div.stFormSubmitButton > button[kind="primaryFormSubmit"],div.stFormSubmitButton > button{{background:linear-gradient(90deg,{P_BLUE},{P_PURPLE}) !important;border:none !important;color:white !important;font-weight:700 !important;border-radius:10px !important;}}
div.stButton > button[kind="primary"]:hover,div.stFormSubmitButton > button:hover{{background:linear-gradient(90deg,#5088BB,#7B6BAD) !important;box-shadow:0 3px 10px rgba(96,150,200,0.3) !important;}}
div.stButton > button:not([kind="primary"]){{background:white !important;border:1.5px solid #D0DFF0 !important;color:{P_BLUE} !important;}}
div.stButton > button:not([kind="primary"]):hover{{background:#F0F6FF !important;border-color:{P_BLUE} !important;}}
section[data-testid="stSidebar"] div.stButton > button{{background:rgba(255,255,255,0.13) !important;border:1px solid rgba(255,255,255,0.28) !important;color:rgba(255,255,255,0.85) !important;border-radius:7px !important;font-size:0.75rem !important;padding:0.22rem 0.5rem !important;margin-bottom:0.3rem !important;font-weight:400 !important;height:auto !important;}}
section[data-testid="stSidebar"] div.stButton > button:hover{{background:rgba(255,255,255,0.28) !important;color:white !important;}}
</style>
""", unsafe_allow_html=True)

COMMON_PERSONA_GUIDE = """
[공통 응답 규칙 — 반드시 지켜야 함]

[대화의 핵심 목적]
- 당신의 가장 중요한 역할은 사용자가 자신의 감정을 더 구체적으로 표현하도록 도와, 우울 관련 감정 라벨을 더 정확히 파악할 수 있게 하는 것입니다.
- 감정 분석 모델은 19가지 감정 라벨(불안, 분노, 슬픔, 상처, 당황, 기쁨, 감사, 사랑, 외로움, 우울감, 무기력, 절망감, 죄책감, 자살충동, 스트레스, 자존감 저하, 대인관계, 일상, 기타)을 감지합니다.
- 매 응답은 사용자의 발화를 위 19개 감정 중 더 구체적인 감정으로 좁혀 가는 방향이어야 합니다.

[감정 탐색 우선순위]
- 우선적으로 우울 관련 감정(우울감, 외로움, 무기력, 절망감, 죄책감, 자존감 저하, 자살충동, 슬픔, 상처, 스트레스, 대인관계 문제)을 세분화해 탐색하세요.
- 사용자가 긍정 감정(기쁨, 감사, 사랑)을 말할 때도 놓치지 말고, 현재의 보호요인 또는 회복 단서로 탐색하세요.
- 사용자의 표현이 모호하면 감정을 추정하지 말고, 비슷한 감정 사이를 구별하는 질문을 하세요.
  예: "그 감정이 우울함에 더 가깝나요, 아니면 외로움이나 무기력에 더 가까운가요?"

[질문 방식]
- 한 번에 질문은 1개만 하세요.
- 질문은 다음 다섯 축 중 하나를 선택해 짧게 물으세요:
  1) 감정 이름  2) 상황/사건  3) 떠오른 생각  4) 몸의 감각  5) 행동 변화
- 사용자의 말이 짧거나 단답이면 반드시 감정 탐색 질문을 하나 덧붙이세요.
- 사용자의 발화에서 감정 단어를 포착하면, 그 감정을 반영하는 공감 문장을 먼저 말하고 그 뒤에 열린 질문을 이어가세요.

[우울 intent 세분화 유도]
- 사용자가 "힘들다", "지친다", "그냥 별로다", "아무것도 하기 싫다"처럼 모호하게 말하면,
  우울감 / 무기력 / 외로움 / 절망감 / 스트레스 / 자존감 저하 중 무엇에 가까운지 구체화하도록 도와주세요.
- 사용자가 사람 문제를 말하면 대인관계 라벨 가능성을 염두에 두고, 감정(상처, 분노, 외로움, 죄책감)과 분리해서 탐색하세요.
- 사용자가 자기비난을 하면 죄책감·자존감 저하 가능성을 살피세요.
- 사용자가 포기, 의미 없음, 미래 없음 같은 표현을 하면 절망감 가능성을 살피세요.

[위험 신호 대응]
- 우울감, 무기력, 절망감, 자살충동 같은 위험 신호가 보이면 절대 가볍게 넘기지 마세요.
- 바로 전화번호부터 제시하지 말고 반드시 공감 → 현재 상태 확인 → 도움 리소스 안내 순서를 지키세요.
- 자살충동, 자해, 죽고 싶음, 사라지고 싶음, 구체적 계획 언급이 보이면 현재의 안전 여부를 먼저 확인하세요.
- 이 서비스는 진단이 아니라 감정 탐색과 도움 연결을 돕는 도구라는 점을 유지하세요.

[표현 방식]
- 답변은 자연스러운 문장으로만 작성하세요. 목록, 불릿, 기계적인 감정 라벨 나열은 피하세요.
- 사용자가 스스로 감정을 말하도록 돕되, 정답을 강요하지 마세요.
"""

PERSONAS = {
    "🧑‍⚕️ 상담사 지우": {
        "description": "따뜻하고 공감 잘 하는 전문 심리 상담사",
        "system": """당신은 '지우'라는 이름의 전문 심리 상담사입니다.

[역할]
- 내담자의 감정을 판단 없이 경청하고, 안전한 공간을 제공합니다.
- 존댓말만 사용합니다. 반말 금지.
- 전문 용어보다 쉬운 일상어를 쓰되, 필요하면 전문가 의뢰를 부드럽게 제안합니다.
- 가능하면 사용자의 감정을 우울감, 외로움, 무기력, 절망감, 죄책감, 자존감 저하 중 어떤 쪽에 가까운지 부드럽게 구체화하도록 도와주세요.

[말투·길이]
- 한 번 응답할 때 150자~250자 사이로 작성합니다.
- 공감 → 감정 반영 → 열린 질문 순서의 3문장 구조를 유지하세요.
- 감정이 모호하면 사용자가 스스로 더 정확한 감정을 말할 수 있도록 짧은 구별 질문을 사용하세요.
- 예: "그런 마음이 드셨군요. 많이 지치고 마음이 가라앉은 느낌이 있었던 것 같아요. 그 감정이 우울함에 더 가까운지, 아니면 무기력에 더 가까운지 조금 더 들려주실 수 있을까요?"
""" + COMMON_PERSONA_GUIDE,
        "color": "#6096C8",
    },

    "🦔 고슴도치 또치 ": {
        "description": "편하게 털어놓을 수 있는 동네 친구",
        "system": """너는 '고슴도치 또치'이라는 이름의 오래된 친구야. 반말로 대화해.

[역할]
- 상담사처럼 딱딱하게 말하지 말고, 진짜 친구가 옆에서 들어주듯 편하게 반응해.
- 판단하지 않고 "야 그랬구나", "진짜 힘들었겠다" 같은 친근한 말투를 써.
- 격식체·존댓말 금지. '~요' 끝맺음 금지.
- 가볍게 맞장구만 치고 끝내지 말고, 사용자가 감정을 더 구체적으로 말하도록 한 단계 더 물어봐.
- 특히 우울한지, 외로운지, 너무 지친 건지처럼 감정을 생활 언어로 구체화하도록 도와줘.

[말투·길이]
- 한 번 응답할 때 80자~150자 사이로 짧고 친근하게 작성해.
-
- 구조: 공감 한 마디 → 감정 또는 상황을 더 구체화하는 질문 한 문장.
- 예: "아 진짜 많이 답답했겠다. 그게 그냥 힘든 느낌이야, 아니면 좀 외롭고 허한 쪽에 더 가까워?"
""" + COMMON_PERSONA_GUIDE,
        "color": "#7A5045",
    },

    "🤖 AI 어시스턴트 클로": {
        "description": "차분하고 정확한 AI 어시스턴트",
        "system": """당신은 '클로'라는 AI 어시스턴트입니다.

[역할]
- 차분하고 중립적인 톤으로 사용자의 상황을 정리해 줍니다.
- 감정을 요약하고, 필요한 정보나 자원을 논리적으로 안내합니다.
- 존댓말을 사용합니다.
- 사용자의 감정을 모호하게 요약하지 말고, 우울감/무기력/외로움/절망감처럼 비슷한 감정을 구분할 수 있도록 짧은 확인 질문을 하세요.

[말투·길이]
- 한 번 응답할 때 120자~200자 사이로 작성합니다.
- 구조: 감정 요약 → 감정 구별 또는 사실 확인 질문 → (필요 시) 간단한 자원 제안.
- 예: "지금 많이 지치고 마음이 가라앉아 있는 상태로 들립니다. 이 감정이 우울함에 더 가까운지, 아니면 아무것도 하기 싫은 무기력에 더 가까운지 말씀해 주실 수 있을까요?"
""" + COMMON_PERSONA_GUIDE,
        "color": "#8B7BAD",
    },

    "🧑‍⚕️멘토 선생님": {
        "description": "논리적이고 조언 잘 해주는 인생 멘토",
        "system": """당신은 경험이 풍부한 인생 멘토입니다.

[역할]
- 사용자의 고민을 큰 그림에서 바라보고, 가볍게 프레임을 제안합니다.
- 설교하지 말고, 먼저 경청한 뒤 한 가지 관점을 부드럽게 제시합니다.
- 존댓말을 사용합니다.
- 조언보다 먼저 감정을 세분화하는 데 집중하세요.
- 사용자의 감정이 충분히 구체화되기 전에는 해결책 제안을 서두르지 마세요.

[말투·길이]
- 한 번 응답할 때 150자~250자 사이로 작성합니다.
- 구조: 공감 → 감정 또는 상황을 해석할 수 있는 짧은 관점 → 열린 질문.
- 예: "많이 지치셨던 것 같습니다. 때로는 같은 힘듦 안에도 우울함, 무기력, 외로움이 조금씩 다르게 섞여 있기도 합니다. 지금은 어떤 감정이 가장 크게 느껴지시는지 말씀해 주실 수 있을까요?"
""" + COMMON_PERSONA_GUIDE,
        "color": "#E67E22",
    },

    "😄 개그맨 철수": {
        "description": "유머러스하고 긍정 에너지 넘치는 친구",
        "system": """너는 '철수'야. 유머 감각 있는 친구지만, 상대 기분을 잘 살피는 편이야. 반말로 대화해.

[역할]
- 가볍고 친근한 에너지를 줄 수는 있지만, 상대 감정을 절대 가볍게 소비하지 마.
- 일상 이야기나 가벼운 스트레스 상황에서는 약한 유머를 써도 된다.
- 우울감, 무기력, 절망감, 죄책감, 자존감 저하, 자살충동 표현이 보이면 즉시 유머를 멈추고 진지하게 공감해.
- 사용자가 감정을 더 구체적으로 말하도록 도와야 하며, 웃음으로 주제를 넘기면 안 된다.

[말투·길이]
- 한 번 응답할 때 80자~150자 사이로 짧고 자연스럽게 작성해.
- 구조: 짧은 공감 → 감정을 구체화하는 질문 한 문장.
- 위험 신호가 있으면 유머 없이 진지하게 반응해.
- 예(저위험 상황): "아 그거 진짜 웃픈 상황이긴 하다. 근데 너 그때 그냥 짜증난 거였어, 아니면 좀 서운하고 허무했어?"
- 예(위험 신호 상황): "그 말은 가볍게 들리지 않는다. 지금 마음이 얼마나 무너진 느낌인지 조금만 더 말해줄래?"
""" + COMMON_PERSONA_GUIDE,
        "color": "#E74C3C",
    },
}
INTERESTS   = ["영화","음악","독서","운동","게임","요리","여행","반려동물","그림/예술","기타"]
OCCUPATIONS = ["대학생/대학원생","직장인","자영업자","주부","무직/구직중","기타"]
REGIONS     = ["서울","부산","대구","인천","광주","대전","울산","경기","강원","충북","충남","전북","전남","경북","경남","제주","세종"]
HIGH_RISK   = {"자살충동","절망감","죄책감","우울감","무기력"}

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
PHQ9_SEVERITY = [
    (0,4,  "양호", "#E8F5EF","#2A6049","추가 조치 불필요"),
    (5,9,  "경도", "#FEFAE8","#6B5A20","자기관리 권장"),
    (10,14,"보통", "#FFF0D8","#8B5A20","상담·치료 계획 고려"),
    (15,19,"중증", "#FFE0D0","#8B3A20","전문가 상담 권고"),
    (20,27,"위험", "#FFD8D8","#8B2020","즉각적 전문가 의뢰"),
]
COMMUNITY_PROGRAMS = [
    {"name":"📌 국립정신건강센터","desc":"국가 정신건강 전문기관 — 진료·연구·교육·위기대응","link":"https://www.ncmh.go.kr","link_text":"홈페이지"},
    {"name":"📌 국가정신건강정보포털","desc":"질병정보·자가진단·의료기관 통합 검색","link":"https://www.mentalhealth.go.kr","link_text":"포털"},
    {"name":"📌 복지로","desc":"생애주기별 복지서비스 통합 검색 (외로움·고립 지원 포함)","link":"https://www.bokjiro.go.kr","link_text":"서비스 찾기"},
    {"name":"📌 청년 불씨 프로젝트","desc":"고립청년 회복 지원 — 정서 워크숍·취향 소모임·일상회복 챌린지","link":"https://www.nyj.go.kr","link_text":"남양주시"},
    {"name":"📌 마인드부스터 Green","desc":"대학생 우울 개선 — 인지행동치료 기반 28회기 앱","link":"https://apps.apple.com/kr/app/%EB%A7%88%EC%9D%B8%EB%93%9C%EB%B6%80%EC%8A%A4%ED%84%B0/id1490078741","link_text":"앱스토어"},
    {"name":"📌 마음이음 (채팅 상담)","desc":"24시간 채팅 기반 심리 상담 — 카카오톡 채널 또는 앱","link":"https://www.mind-i.com","link_text":"상담 채널"},
]
SIGUNGU_SEARCH_URL = "https://www.mentalhealth.go.kr/portal/health/fac/PotalHealthFacListTab1.do"
NCMH_URL = "https://www.ncmh.go.kr"

@st.cache_resource
def get_model():
    return load_model()

defaults = {
    "page":"💬 일상 대화","profile_done":False,"user_profile":{},
    "persona":"🧑‍⚕️ 상담사 지우","messages":[],"api_history":[],
    "last_analysis":None,"score_history":[],"probs_history":[],"risk_history":[],
    "phq9_done":False,"phq9_answers":[0]*9,"phq9_total":None,
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v


# ─────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────
def render_chat(messages, profile):
    pn = st.session_state.persona.split(" ",1)[1] if st.session_state.persona else "AI"
    html = '<div class="chat-wrap">'
    if not messages:
        name = profile.get("nickname","")
        interests = profile.get("interests", [])
        # 관심사 안내 문구
        if interests:
            interest_str = ", ".join(interests)
            interest_hint = (
                f'<div style="margin-top:0.6rem;background:#EEF4FF;border-radius:10px;'
                f'padding:0.55rem 0.9rem;font-size:0.83rem;color:#3A5A9A;line-height:1.6;">'
                f'💡 <b>{interest_str}</b>에 관한 이야기를 편하게 나눠 보세요!<br>'
                f'<span style="color:#7A8FA6;font-size:0.79rem;">관심 있는 주제부터 시작해도 좋아요 😊</span>'
                f'</div>'
            )
        else:
            interest_hint = ""
        html += (
            f'<div class="empty-state">🫧<br>'
            f'안녕하세요{" "+name+"님" if name else ""}! 저는 {pn}입니다.<br>'
            f'편하게 이야기 나눠요 😊'
            f'</div>'
            f'{interest_hint}'
        )
    for msg in messages:
        c = msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
        if msg["role"]=="user":
            html += f'<div class="bubble-user"><div class="bubble-user-inner">{c}</div></div>'
        else:
            html += f'<div class="bubble-bot"><div class="bot-avatar">🫧</div><div class="bubble-bot-inner">{c}</div></div>'
    html += '</div>'
    return html

def get_avg_probs(exclude_daily=False, min_pct=0.0):
    if not st.session_state.probs_history: return None
    avg = np.stack(st.session_state.probs_history, axis=0).mean(axis=0)
    return {INV_MAP[i]: round(float(avg[i])*100,1)
            for i in INV_MAP
            if not(exclude_daily and i==11) and round(float(avg[i])*100,1)>=min_pct}

def build_dynamic_system(base, profile, analysis):
    parts = [base]
    if profile:
        ctx = "\n[사용자 정보]\n"
        for k,label in [("nickname","호칭"),("age_group","연령대"),("gender","성별"),("occupation","직업"),("region","지역")]:
            if profile.get(k): ctx += f"- {label}: {profile[k]}\n"
        if profile.get("interests"): ctx += f"- 관심사: {', '.join(profile['interests'])}\n"
        parts.append(ctx)
    if analysis and analysis.get("probs") is not None:
        probs = analysis["probs"]
        top = sorted([(INV_MAP[i],float(probs[i])) for i in range(len(probs)) if i!=11],
                     key=lambda x:x[1], reverse=True)[:3]
        parts.append(f"\n[직전 감정]\n- 위험도:{analysis['level']}\n- 상위:{', '.join(f'{n}({v*100:.0f}%)' for n,v in top)}\n")
    return "\n".join(parts)

def _find_ko_font():
    import os
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
              "/Library/Fonts/AppleSDGothicNeo.ttc",
              "C:/Windows/Fonts/malgun.ttf","C:/Windows/Fonts/NanumGothic.ttf"]:
        if os.path.exists(p): return p
    return None

# ── 불용어 처리 기준 (Chen 2021 + Jin et al. 2023) ─────────────────────
_NEGATION_FORMS = {"안", "못", "없", "아니", "아니다", "없다", "없어"}
_EMPHASIS_FORMS = {"너무", "정말", "매우", "엄청", "진짜", "완전", "참", "정말로"}
# NNG: 일반명사, NNP: 고유명사만 유지
# 부정어·감정강조어는 아래 _NEGATION_FORMS/_EMPHASIS_FORMS에서 별도 보존
# 동사(VV), 형용사(VA), 조사, 어미, 부사 일반 → 전부 제거
# 근거: Chen(2021) 불용어 제거 후 명사 중심 7.9단어로 88% 정확도
_KEEP_TAGS      = {"NNG", "NNP"}   # 명사만 (kiwi_tokenize 전용)

# 워드클라우드용 넓은 품사 기준
# Chen (2021): 짧은 비격식 텍스트에서 형용사·동사가 감정의 핵심
# Nandwani & Verma (2021): 품사별 감정 기여도 순서 VA > MAG > VV > NNG 참고
_KEEP_TAGS_WC   = {"NNG", "NNP", "VV", "VA", "MAG", "MAJ"}


def kiwi_tokenize(text: str) -> list:
    """
    kiwipiepy 형태소 분석 기반 토크나이저.

    - 부정어("안","못","없","아니"): 제거하지 않음 → 감정 방향성 유지
    - 감정강조어("너무","정말","매우","엄청"): 제거하지 않음 → 감정 강도 반영
    - 조사·어미 일반 부사: 제거
    - 접두사(XPN) + 명사(NNG/NNP) 자동 결합
      예) '무(XPN)' + '기력(NNG)' → '무기력'  /  '불(XPN)' + '안정(NNG)' → '불안정'
      단어를 수동으로 나열하지 않아도, Kiwi가 XPN으로 분류한 토큰을
      바로 뒤에 오는 명사와 코드가 자동으로 합쳐줌.
    """
    if not _KIWI_OK or not text.strip():
        tokens = text.split()
        return [t for t in tokens if t in _NEGATION_FORMS or t in _EMPHASIS_FORMS
                or (len(t) >= 2 and t not in {"그리고","그런데","하지만","그래서","그냥","이렇게","그렇게"})]
    result = []
    prefix = ""   # XPN 접두사 임시 저장소
    for t in _kiwi_instance.tokenize(text, stopwords=None):
        form = t.form
        tag  = str(t.tag)
        # ① 접두사(XPN)이면 임시 저장 후 다음 토큰으로 넘어감
        if tag == "XPN":
            prefix = form
            continue
        # ② 부정어·강조어는 품사 무관하게 보존
        if form in _NEGATION_FORMS or form in _EMPHASIS_FORMS:
            result.append(form)
            prefix = ""
        # ③ 명사이면 앞 접두사와 자동 결합 후 추가
        elif tag in _KEEP_TAGS:
            combined = prefix + form
            if len(combined) >= 2:
                result.append(combined)
            prefix = ""
        else:
            prefix = ""   # 명사·보존어 외 다른 태그가 오면 접두사 무효화
    return result


def kiwi_tokenize_for_wordcloud(text: str) -> list:
    """
    워드클라우드 전용 토크나이저. (lemma, pos_tag) 튜플 리스트 반환.

    Chen (2021) + Nandwani & Verma (2021) 근거:
    - 원형(lemma) 기반: 슬퍼/슬프다/슬펐다 → 모두 '슬프다'로 통합
    - 품사 범위: NNG, NNP, VV, VA, MAG, MAJ
    - 부정어·강조어: 품사 무관 보존
    - XPN 접두사 + NNG/NNP 자동 결합
    """
    if not _KIWI_OK or not text.strip():
        result = []
        for tok in text.split():
            if tok in _NEGATION_FORMS or tok in _EMPHASIS_FORMS:
                result.append((tok, "MAG"))
            elif len(tok) >= 2:
                result.append((tok, "NNG"))
        return result

    result = []
    prefix = ""
    for t in _kiwi_instance.tokenize(text, stopwords=None):
        form  = t.form
        lemma = t.lemma if hasattr(t, "lemma") and t.lemma else t.form
        tag   = str(t.tag)

        if tag == "XPN":
            prefix = form; continue

        if form in _NEGATION_FORMS:
            result.append((form, "MAG")); prefix = ""; continue

        if form in _EMPHASIS_FORMS:
            result.append((form, "MAG")); prefix = ""; continue

        if tag in _KEEP_TAGS_WC:
            # score 기반 오타·파편 필터
            # Kiwi가 형태소를 억지로 분류할 때 score가 -20 이하로 떨어짐
            # 정상 단어: -6 ~ -15 / 오타·파편: -20 이하 (경험적 설정값)
            if t.score < -20:
                prefix = ""; continue
            if tag in {"NNG", "NNP"}:
                combined = prefix + form
                if len(combined) >= 2:
                    result.append((combined, tag))
            else:
                if len(lemma) >= 2:
                    result.append((lemma, tag))
            prefix = ""
        else:
            prefix = ""

    return result


def count_valid_words_kiwi(text: str) -> int:
    """
    한 발화의 유효 단어 수 반환.
    넓은 품사 기준(NNG/NNP/VV/VA/MAG/MAJ) 사용 → 감정 표현 발화 정상 카운트.
    경험적 기준: MIN_WORDS_PER_TURN=2 (한국어 최소 명제 단위 기준)
    """
    return len(kiwi_tokenize_for_wordcloud(text))

# ── 욕설 감지 (Korean Profanity Detection) ──────────────────────────────
# 한국어 전용 라이브러리가 없으므로 어근 목록 + 변형 패턴으로 처리
# 원문 + 공백 제거 버전 모두 체크 (ㅅ ㅂ 처럼 띄어쓴 경우 대응)
_PROFANITY_LIST = [
    # 대표 욕설 어근
    "씨발", "씨팔", "시발", "시팔",
    "개새", "개새끼", "새끼",
    "병신", "등신",
    "미친", "미친놈", "미친년",
    "존나", "지랄",
    "닥쳐", "꺼져",
    "개같", "개년", "개놈", "개소리",
    "애미", "니애미", "어미뒤",
    "보지", "자지", "섹스", "씹",
    # 초성 축약형
    "ㅅㅂ", "ㅆㅂ", "ㅅㄲ", "ㄱㅅㄲ",
    "ㅂㅅ", "ㅁㅊ", "ㅈㄴ", "ㅈㄹ",
]

def detect_profanity(text: str) -> tuple:
    """
    한국어 욕설 감지 함수.

    처리 방식:
    - 원문에서 직접 어근 탐색
    - 공백 제거 버전 탐색 (ㅅ ㅂ, 씨 발 처럼 띄어쓴 변형 대응)
    - 대소문자 무관 처리

    Returns:
        (감지됨: bool, 감지된 단어: str)
    """
    import re as _re
    text_nospace = _re.sub(r'\s+', '', text)
    for word in _PROFANITY_LIST:
        if word in text or word in text_nospace:
            return True, word
    return False, ""




def make_wordcloud_png(texts: list):
    """
    논문 기반 워드클라우드 생성.

    전처리 파이프라인:
    1단계 — 텍스트 정제: 특수문자·이모지 제거, 반복문자 정규화, 숫자 제거
            Nandwani & Verma (2021) 전처리 단계 순서 참고
    2단계 — Kiwi 형태소 분석 + 원형(lemma) 추출
            Chen (2021) 형태소 기반 전처리 참고
            원형 기반 추출로 활용형 자동 통합 (슬퍼/슬프다/슬펐다 → 슬프다)
    3단계 — 품사 필터: NNG, NNP, VV, VA, MAG, MAJ
            Nandwani & Verma (2021) 감성 분석에서의 품사별 감정 기여도 참고
            (VA > MAG > VV > NNG 순)
    4단계 — 빈도 계산 + 적응형 최소 빈도 필터
            누적 토큰 30개 미만: 1회 이상 표시 (소량 데이터 대응)
            누적 토큰 30개 이상: 2회 이상 표시 (경험적 설정값)
    5단계 — 워드클라우드 생성 (빈도 기반, 상위 30단어)
    """
    if not _WORDCLOUD_OK or not texts:
        return None, 0

    # 1단계: 텍스트 정제 (Nandwani & Verma, 2021 참고)
    def clean_text(t: str) -> str:
        t = re.sub(r"[^\w가-힣\s]", " ", t)   # 특수문자·이모지 제거
        t = re.sub(r"(.)\1{2,}", r"\1", t)    # 반복문자 정규화 (ㅠㅠㅠ → ㅠ)
        t = re.sub(r"\d+", " ", t)            # 숫자 제거
        return t.strip()

    # 2~3단계: 형태소 분석 → (lemma, tag) 리스트 (Chen, 2021 참고)
    all_pairs = []
    for t in texts:
        all_pairs.extend(kiwi_tokenize_for_wordcloud(clean_text(t)))

    if not all_pairs:
        return None, 0

    unique_count = len(set(lemma for lemma, _ in all_pairs))

    # 4단계: 원형 기준 빈도 계산
    raw_freq = Counter()
    for lemma, tag in all_pairs:
        if len(lemma) >= 2:
            raw_freq[lemma] += 1

    if not raw_freq:
        return None, unique_count

    # 적응형 최소 빈도 필터 (경험적 설정값)
    min_freq = 2 if len(all_pairs) >= 30 else 1
    freq = {word: cnt for word, cnt in raw_freq.items() if cnt >= min_freq}

    if not freq:
        return None, unique_count

    # 5단계: 워드클라우드 생성
    try:
        wc = WordCloud(
            font_path=_find_ko_font(), width=700, height=420,
            background_color="white", colormap="PuBu",
            prefer_horizontal=0.9, relative_scaling=0.2, max_words=30,
        ).generate_from_frequencies(freq)
    except:
        return None, unique_count

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), unique_count


# ─────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫧 말랑해도 돼")
    st.markdown("**마음을 말랑하게 유연하게**")
    st.markdown("---")
    st.markdown("#### 📌 서비스 단계")
    pages = [
        ("💬 일상 대화",       "1단계", "챗봇 상담"),
        ("📊 감정 분석",       "2단계", "NLP 리포트"),
        ("📝 PHQ-9",      "3단계", "우울 자가검진"),
        ("🌱 맞춤 케어",        "4단계", "개인 솔루션"),
        ("🏥 공공 서비스",     "5단계", "기관 연계"),
        ("🩺 상담사 대시보드", "전문가", "위험도 패턴 분석"),
    ]
    for pname,step,desc in pages:
        active = st.session_state.page==pname
        clean_name = pname.split(" ",1)[1] if " " in pname else pname
        if st.button(f"{'▶ ' if active else '　'}{step}  {clean_name}\n{desc}", key=f"nav_{pname}", use_container_width=True):
            st.session_state.page=pname; st.rerun()
    st.markdown("---")
    if st.session_state.page=="💬 일상 대화":
        st.markdown("#### 🎭 대화 상대")
        for pname,pinfo in PERSONAS.items():
            short=pname.split(" ",1)[1]; icon=pname.split()[0]
            if st.button(f"{icon} {short}",key=f"p_{pname}",use_container_width=True):
                if st.session_state.persona!=pname:
                    st.session_state.persona=pname; st.rerun()
            if st.session_state.persona==pname:
                st.markdown(f'<div style="font-size:0.72rem;opacity:0.82;margin-top:-0.4rem;margin-bottom:0.2rem;padding-left:0.3rem;">✓ {pinfo["description"]}</div>', unsafe_allow_html=True)
        st.markdown("---")
    st.markdown("#### 🆘 위기 상담")
    st.markdown('<div style="font-size:0.82rem;line-height:2.1;">☎ 자살예방 <b>1393</b><br>☎ 정신건강위기 <b>1577-0199</b><br>☎ 복지상담 <b>129</b></div>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🗑️ 대화 초기화",key="sidebar_reset",use_container_width=True):
        for k in ["messages","api_history","score_history","probs_history","risk_history"]: st.session_state[k]=[]
        st.session_state.last_analysis=None; st.rerun()


# ─────────────────────────────────────────────────
# 메인 헤더
# ─────────────────────────────────────────────────
page_info = {
    "💬 일상 대화":       ("1단계","일상 대화 · 상담형 챗봇"),
    "📊 감정 분석":       ("2단계","위험 신호 조기 탐지 · NLP 감정 분석"),
    "📝 PHQ-9":          ("3단계","PHQ-9 우울 자가검진"),
    "🌱 맞춤 케어":       ("4단계","PHQ-9 우울 자가검진 맞춤케어"),
    "🏥 공공 서비스":     ("5단계","지역별 정신건강 기관 연계"),
    "🩺 상담사 대시보드": ("전문가","대화 단위 위험도 · 개입 우선순위"),
}
step_label, step_desc = page_info[st.session_state.page]
# "전문가"는 숫자 단계가 아니므로 STEP 뱃지 대신 그대로 표시
_badge = step_label if step_label == "전문가" else f"STEP {step_label[0]}"
st.markdown(f'''<div class="app-header">
    <div class="app-title">🫧 말랑해도 돼</div>
    <div class="app-sub">어떤 감정이든 괜찮아요 · 천천히, 함께 풀어가요</div>
    <div style="margin-top:0.7rem;">
        <span class="step-badge">{_badge}</span>
        <span style="font-size:0.88rem;opacity:0.92;">{step_desc}</span>
    </div>
</div>''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# STEP 1: 일상 대화
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "💬 일상 대화":
    col_chat, col_right = st.columns([1.65, 1], gap="large")

    with col_chat:
        if not st.session_state.profile_done:
            st.markdown('<div class="sec-header">👤 먼저 간단히 알려주세요!</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.82rem;color:#8A7060;margin-bottom:0.7rem;">입력 정보는 맞춤 케어 제공에만 활용됩니다.</div>', unsafe_allow_html=True)
            with st.form("profile_form"):
                cn, cr = st.columns(2)
                with cn: nickname = st.text_input("닉네임 (선택)", placeholder="예: 민수, 지은...")
                with cr: region   = st.selectbox("거주 지역", ["선택 안함"]+REGIONS)
                ca, cb, cc = st.columns(3)
                with ca: age_group  = st.selectbox("연령대", ["20대","30대","40대","50대","60대 이상"])
                with cb: gender     = st.selectbox("성별", ["남성","여성","기타/비공개"])
                with cc: occupation = st.selectbox("직업", OCCUPATIONS)
                interests = st.multiselect("관심사 (복수 선택)", INTERESTS, default=["영화"])
                contact   = st.text_input("비상 연락처 (선택)", placeholder="010-XXXX-XXXX")
                submitted = st.form_submit_button("대화 시작하기 🫧", type="primary", use_container_width=True)
            if submitted:
                st.session_state.user_profile = {"nickname":nickname,"age_group":age_group,"gender":gender,"occupation":occupation,"region":region if region!="선택 안함" else "","interests":interests,"contact":contact}
                st.session_state.profile_done = True; st.rerun()
        else:
            p = st.session_state.user_profile
            tags = ""
            if p.get("nickname"): tags += f'<span class="profile-tag">👤 {p["nickname"]}</span>'
            tags += f'<span class="profile-tag">🎂 {p.get("age_group","")}</span><span class="profile-tag">⚧ {p.get("gender","")}</span>'
            if p.get("occupation"): tags += f'<span class="profile-tag">💼 {p["occupation"]}</span>'
            if p.get("region"):     tags += f'<span class="profile-tag">📍 {p["region"]}</span>'
            for i in p.get("interests",[]): tags += f'<span class="profile-tag">🎯 {i}</span>'
            tc, bc = st.columns([5,1])
            with tc: st.markdown(f'<div style="margin-bottom:0.3rem;">{tags}</div>', unsafe_allow_html=True)
            with bc:
                if st.button("✏️ 수정", key="edit_profile"): st.session_state.profile_done=False; st.rerun()

            persona = PERSONAS[st.session_state.persona]
            picon   = st.session_state.persona.split()[0]
            pname_  = st.session_state.persona.split(" ",1)[1]
            st.markdown(f'<div style="background:{persona["color"]}18;border:1.5px solid {persona["color"]}40;border-radius:10px;padding:0.55rem 0.9rem;margin-bottom:0.6rem;display:flex;align-items:center;gap:0.5rem;"><span style="font-size:1.2rem;">{picon}</span><span style="font-weight:700;color:{persona["color"]};">{pname_}</span><span style="color:#7A8FA6;font-size:0.82rem;">— {persona["description"]}</span></div>', unsafe_allow_html=True)
            st.markdown(render_chat(st.session_state.messages, p), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="input-highlight">', unsafe_allow_html=True)
            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_area("메시지", placeholder="오늘 어떤 하루였나요? 편하게 이야기해 보세요...", height=85, label_visibility="collapsed")
                c1, c2 = st.columns([5,1])
                with c1: submit = st.form_submit_button("전송 ➤", use_container_width=True, type="primary")
                with c2: reset  = st.form_submit_button("초기화", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if reset:
                for k in ["messages","api_history","score_history","probs_history","risk_history"]: st.session_state[k]=[]
                st.session_state.last_analysis=None; st.rerun()

            if submit and user_input.strip():
                # 욕설 감지
                _has_profanity, _bad_word = detect_profanity(user_input.strip())
                if _has_profanity:
                    st.warning("⚠️ 부적절한 표현이 포함되어 있어요. 편하게, 하지만 안전한 언어로 이야기해 주세요 😊")
                    st.stop()
                try: model,tokenizer,inv_map,run_cfg,device=get_model()
                except Exception as e: st.error(f"모델 로드 실패: {e}"); st.stop()
                analysis=get_depression_score(text=user_input.strip(),model=model,tokenizer=tokenizer,device=device,inv_map=inv_map,max_len=run_cfg.get("max_len",64),threshold=run_cfg.get("multi_threshold",3.0))
                with st.spinner("답변 생성 중..."):
                    try:
                        dyn=build_dynamic_system(persona["system"],p,analysis)
                        bot_reply=get_chatbot_response(user_text=user_input.strip(),analysis=analysis,conversation_history=st.session_state.api_history,persona_system=dyn)
                    except Exception as e: bot_reply=f"죄송해요, 오류가 발생했어요. ({e})"
                # 또치 이모티콘 50% 확률 부착
                if st.session_state.persona == "🦔 고슴도치 또치" and random.random() < 0.5:
                    bot_reply = bot_reply + " " + random.choice(["🦔", "🐾", "🍀"])
                st.session_state.messages.append({"role":"user","content":user_input.strip()})
                st.session_state.messages.append({"role":"assistant","content":bot_reply})
                st.session_state.api_history.append({"role":"user","content":user_input.strip()})
                st.session_state.api_history.append({"role":"assistant","content":bot_reply})
                st.session_state.last_analysis=analysis
                _now = datetime.now()
                st.session_state.score_history.append({
                    "turn":  len(st.session_state.score_history) + 1,
                    "score": analysis["score"],
                    "ts":    _now,
                })
                st.session_state.probs_history.append(analysis["probs"])
                top_emos=sorted([(INV_MAP[i],float(analysis["probs"][i])) for i in range(len(analysis["probs"])) if i!=11],key=lambda x:x[1],reverse=True)[:3]
                st.session_state.risk_history.append({
                    "turn":     len(st.session_state.risk_history) + 1,
                    "level":    analysis["level"],
                    "score":    analysis["score"],
                    "text":     user_input.strip()[:60],
                    "top_emos": top_emos,
                    "ts":       _now,
                })
                st.rerun()

            # 고위험 감지 시 위기 안내 (채팅 중 자동 표시 비활성화)

            # 감정 분석 이동 버튼
            n_cur = len(st.session_state.probs_history)
            if n_cur > 0:
                st.markdown("---")
                if n_cur >= MIN_TURNS_FOR_ANALYSIS:
                    if st.button("📊 대화 감정 분석 보기 →", key="go_to_analysis", type="primary"):
                        st.session_state.page="📊 감정 분석"; st.rerun()
                else:
                    remain = MIN_TURNS_FOR_ANALYSIS - n_cur
                    st.info(f"📊 감정 분석은 최소 {MIN_TURNS_FOR_ANALYSIS}회 이상 대화했을 때 가능해요. (현재 {n_cur}회 — {remain}회 더 필요)")

    with col_right:
        n_cur = len(st.session_state.probs_history)
        cond  = n_cur >= MIN_TURNS_FOR_ANALYSIS
        st.markdown('<div class="sec-header">📊 분석 현황</div>', unsafe_allow_html=True)
        mc1, mc2 = st.columns(2)
        mc1.markdown(f'<div class="stat-card"><div class="stat-label">대화 횟수</div><div class="stat-value">{n_cur}회</div></div>', unsafe_allow_html=True)
        mc2.markdown(f'<div class="stat-card"><div class="stat-label">분석 조건</div><div class="stat-value">{"✅ 충족" if cond else f"{n_cur}/{MIN_TURNS_FOR_ANALYSIS}"}</div></div>', unsafe_allow_html=True)
        if cond:
            st.markdown('<div class="cond-met">✅ 충족 — 감정 분석 페이지에서 NLP 리포트 확인 가능</div>', unsafe_allow_html=True)
        else:
            remain = MIN_TURNS_FOR_ANALYSIS - n_cur
            st.markdown(f'<div class="cond-unmet">💬 {remain}회 더 대화하면 NLP 분석이 열립니다</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-header">🆘 위기 상담</div>', unsafe_allow_html=True)
        for num,name,desc in [("1393","자살예방상담전화","24시간"),("1577-0199","정신건강위기상담","24시간"),("129","보건복지상담","복지 연계")]:
            st.markdown(f'<div class="hotline"><span>📞</span><span class="hotline-num">{num}</span><span style="font-size:0.83rem;"><b>{name}</b> <span style="color:#7A8FA6;font-size:0.76rem;">{desc}</span></span></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 2: 감정 분석 — NLP 리포트 (v5 전체 복원 + 파스텔 UI)
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "📊 감정 분석":

    n_turns = len(st.session_state.probs_history)

    if n_turns == 0:
        st.markdown('<div class="empty-state">📊<br>1단계 일상 대화에서 대화를 시작하면<br>여기서 감정 분석 리포트를 확인할 수 있어요.</div>', unsafe_allow_html=True)
        if st.button("💬 일상 대화 시작하기", key="goto_chat_nlp", type="primary"):
            st.session_state.page="💬 일상 대화"; st.rerun()

    elif n_turns < MIN_TURNS_FOR_ANALYSIS:
        need = MIN_TURNS_FOR_ANALYSIS - n_turns
        st.markdown(f"""
        <div style="background:#EDF2FF;border-radius:14px;padding:1.5rem 1.8rem;
                    border:1px solid #BFD0EE;text-align:center;">
            <div style="font-size:1.6rem;margin-bottom:0.5rem;">⏳</div>
            <div style="font-weight:700;font-size:1rem;color:#3A5A9A;margin-bottom:0.5rem;">
                감정 분석을 위해 대화가 조금 더 필요해요
            </div>
            <div style="font-size:0.9rem;color:#5A7ABE;line-height:1.7;">
                정확한 감정 분석을 위해서는 최소 <b>{MIN_TURNS_FOR_ANALYSIS}번의 대화</b>가 필요합니다.<br>
                현재 <b>{n_turns}회</b> 진행됨 — <b>{need}회 더</b> 대화를 나눠 주세요!
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💬 대화 더 하러 가기", type="primary", use_container_width=True):
            st.session_state.page="💬 일상 대화"; st.rerun()

    else:
        st.markdown(f'<div class="sec-header">💭 전체 대화 감정 분석 결과 (총 {n_turns}회 대화 기준)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#EFF6FF;border-radius:10px;padding:0.7rem 1rem;margin-bottom:1rem;
                    font-size:0.85rem;color:#2D5A8E;border-left:4px solid #6096C8;">
            💡 전체 대화의 감정별 평균 확률입니다. 일상 포함 전체 20개 감정을 표시합니다.
        </div>
        """, unsafe_allow_html=True)

        avg_probs        = get_avg_probs(exclude_daily=False, min_pct=0.0)
        avg_probs_no_daily = get_avg_probs(exclude_daily=True, min_pct=0.0) or {}

        dc1, dc2 = st.columns([1.5, 1], gap="large")

        with dc1:
            # ── 바차트: 20개 감정 전체 ──────────────────────────────
            st.markdown('<div class="sec-header">📊 감정별 평균 확률 (20개 전체)</div>', unsafe_allow_html=True)
            if not avg_probs:
                st.info("표시할 감정 데이터가 없습니다.")
            else:
                sorted_emotions = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
                emo_names = [e[0] for e in sorted_emotions]
                emo_vals  = [e[1] for e in sorted_emotions]
                x_max = max(max(emo_vals)*1.15, 5)
                fig_bar = go.Figure(go.Bar(
                    x=emo_vals, y=emo_names, orientation="h",
                    marker_color="#6096C8",
                    text=[f"{v:.1f}%" if v>=0.1 else f"{v:.2f}%" for v in emo_vals],
                    textposition="outside",
                    hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
                ))
                fig_bar.update_layout(
                    height=max(500, len(emo_names)*26),
                    margin=dict(l=0, r=80, t=10, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(gridcolor="#EEF4FA", title="평균 확률 (%)", range=[0, x_max]),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar")

            # ── 위험도 추이 (대화턴 / 1일 / 7일 / 14일 / 30일) ────────
            st.markdown('<div class="sec-header">📈 위험도 추이</div>', unsafe_allow_html=True)
            sh  = st.session_state.score_history
            rh_cur = st.session_state.risk_history
            _has_ts = sh and isinstance(sh[0], dict) and "ts" in sh[0]

            def _risk_color(s):
                if s>=60: return P_WARM
                elif s>=35: return P_PINK
                elif s>=15: return "#C4A857"
                return "#5B9E7A"

            def _turn_fig(records, height=200):
                """대화 턴 기반 차트 (항상 사용 가능)."""
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[r["turn"] for r in records],
                    y=[r["score"] for r in records],
                    mode="lines+markers",
                    line=dict(color="#8B7BAD", width=2),
                    marker=dict(color=[_risk_color(r["score"]) for r in records],
                                size=9, line=dict(color="white", width=1.5)),
                    hovertemplate="턴 %{x}<br>위험도: %{y:.1f}<extra></extra>",
                ))
                fig.add_hrect(y0=60,y1=100,fillcolor=P_WARM, opacity=0.07,line_width=0)
                fig.add_hrect(y0=35,y1=60, fillcolor=P_PINK, opacity=0.07,line_width=0)
                fig.update_layout(
                    height=height, margin=dict(l=0,r=0,t=8,b=24),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="대화 턴", gridcolor="#EEF4FA", dtick=1),
                    yaxis=dict(title="위험도", gridcolor="#EEF4FA", range=[0,100]),
                )
                return fig

            def _datetime_fig(records, height=200):
                """datetime x축 차트 — Plotly 기본 줌으로 일/시간 단위 드릴다운 가능."""
                if not records: return None
                xs = [r["ts"] for r in records]   # datetime 객체 그대로 → Plotly가 자동 스케일
                ys = [r["score"] for r in records]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    line=dict(color="#6096C8", width=2),
                    marker=dict(color=[_risk_color(s) for s in ys],
                                size=9, line=dict(color="white", width=1.5)),
                    text=[r["text"][:20]+"…" if len(r["text"])>20 else r["text"]
                          for r in records],
                    hovertemplate="%{x|%m/%d %H:%M}<br>위험도: %{y:.1f}<br>%{text}<extra></extra>",
                ))
                fig.add_hrect(y0=60,y1=100,fillcolor=P_WARM, opacity=0.07,line_width=0)
                fig.add_hrect(y0=35,y1=60, fillcolor=P_PINK, opacity=0.07,line_width=0)
                fig.update_layout(
                    height=height, margin=dict(l=0,r=0,t=8,b=24),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="시간", gridcolor="#EEF4FA",
                               rangeslider=dict(visible=True, thickness=0.06)),  # 하단 범위 슬라이더
                    yaxis=dict(title="위험도", gridcolor="#EEF4FA", range=[0,100]),
                )
                return fig

            if rh_cur:
                from datetime import timedelta
                _now_dt = datetime.now()

                if _has_ts:
                    tab_turn, tab1, tab7, tab14, tab30 = st.tabs(["대화턴","1일","7일","14일","30일"])

                    with tab_turn:
                        st.plotly_chart(_turn_fig(rh_cur), use_container_width=True, key="turn_rh_cur")
                        st.markdown('<div style="font-size:0.79rem;color:#8A9AB5;">💡 다른 탭에서 날짜·시간 기반 추이를 확인하고 줌으로 세부 대화를 볼 수 있어요.</div>', unsafe_allow_html=True)

                    with tab1:
                        _r1 = [r for r in rh_cur if (_now_dt - r["ts"]).days < 1]
                        if _r1:
                            st.plotly_chart(_datetime_fig(_r1), use_container_width=True, key="dt_r1")
                            st.markdown('<div style="font-size:0.79rem;color:#8A9AB5;">📌 오늘 대화 기록이에요. 그래프를 드래그해 확대하면 특정 시간대를 자세히 볼 수 있어요.</div>', unsafe_allow_html=True)
                        else:
                            st.info("오늘 대화 기록이 없어요.")

                    with tab7:
                        _r7 = [r for r in rh_cur if (_now_dt - r["ts"]).days < 7]
                        if _r7:
                            st.plotly_chart(_datetime_fig(_r7), use_container_width=True, key="dt_r7")
                            st.markdown('<div style="font-size:0.79rem;color:#8A9AB5;">📌 최근 7일 기록이에요. 그래프를 드래그하거나 하단 슬라이더로 특정 날짜를 확대해 보세요.</div>', unsafe_allow_html=True)
                        else:
                            st.info("최근 7일 대화 기록이 없어요.")

                    with tab14:
                        _r14 = [r for r in rh_cur if (_now_dt - r["ts"]).days < 14]
                        if _r14:
                            st.plotly_chart(_datetime_fig(_r14), use_container_width=True, key="dt_r14")
                            st.markdown('<div style="font-size:0.79rem;color:#8A9AB5;">📌 최근 14일 기록이에요. 하단 슬라이더로 범위를 좁혀 특정 날짜를 확대해 보세요.</div>', unsafe_allow_html=True)
                        else:
                            st.info("최근 14일 대화 기록이 없어요.")

                    with tab30:
                        _r30 = [r for r in rh_cur if (_now_dt - r["ts"]).days < 30]
                        if _r30:
                            st.plotly_chart(_datetime_fig(_r30), use_container_width=True, key="dt_r30")
                            st.markdown('<div style="font-size:0.79rem;color:#8A9AB5;">📌 최근 30일 기록이에요. 하단 슬라이더로 범위를 좁혀 특정 날짜를 확대해 보세요.</div>', unsafe_allow_html=True)
                        else:
                            st.info("최근 30일 대화 기록이 없어요.")
                else:
                    # 타임스탬프 없으면 턴 기반만 표시
                    st.plotly_chart(_turn_fig(rh_cur), use_container_width=True, key="turn_rh_cur_nots")
                    st.markdown('<div style="font-size:0.79rem;color:#8A9AB5;">* 날짜별 탭은 앱을 재시작하면 표시돼요.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:0.84rem;color:#A0AEC0;">대화를 시작하면 위험도 추이가 표시돼요.</div>', unsafe_allow_html=True)

            # ── 워드클라우드 ─────────────────────────────────────────
            st.markdown('<div class="sec-header">☁️ 대화 워드클라우드</div>', unsafe_allow_html=True)
            user_msgs    = [m["content"] for m in st.session_state.messages if m["role"]=="user"]
            n_user_turns = len(user_msgs)

            # 발화별 유효 단어 수 계산 (Chen 2021 + Jin et al. 2023)
            # - valid_msgs : 게이트 조건 판단용 (3단어 이상 발화가 3번 이상인지)
            # - 워드클라우드 입력 : user_msgs 전체 (짧은 발화도 단어 자체는 유효)
            msg_word_counts = [(m, count_valid_words_kiwi(m)) for m in user_msgs]
            valid_msgs      = [m for m, cnt in msg_word_counts if cnt >= MIN_WORDS_PER_TURN]
            short_msgs      = [m for m, cnt in msg_word_counts if cnt < MIN_WORDS_PER_TURN]
            n_valid         = len(valid_msgs)
            n_short         = len(short_msgs)

            if not _WORDCLOUD_OK:
                st.warning("워드클라우드 패키지 필요: `pip install wordcloud matplotlib`")

            elif n_valid < MIN_TURNS_FOR_WORDCLOUD:
                # ── 게이트 조건 미충족: 구체적 진행 현황 안내 ───────────
                remain_valid = MIN_TURNS_FOR_WORDCLOUD - n_valid
                turn_ok   = n_user_turns >= MIN_TURNS_FOR_WORDCLOUD
                valid_ok  = n_valid >= MIN_TURNS_FOR_WORDCLOUD

                st.markdown(f"""
                <div style="background:#EFF6FF;border-radius:12px;padding:1rem 1.2rem;
                            border:1px solid #BFD8EE;border-left:4px solid #6096C8;">
                    <div style="font-weight:700;font-size:0.92rem;color:#2D5A8E;margin-bottom:0.7rem;">
                        ☁️ 워드클라우드를 만들려면 조금 더 이야기해 주세요!
                    </div>
                    <div style="font-size:0.85rem;color:#3A5A9A;line-height:2.0;">
                        {"✅" if turn_ok else "⏳"} <b>전체 대화 횟수</b>: {n_user_turns}회 / 최소 {MIN_TURNS_FOR_WORDCLOUD}회<br>
                        {"✅" if valid_ok else "⏳"} <b>3단어 이상 발화</b>: {n_valid}회 / 최소 {MIN_TURNS_FOR_WORDCLOUD}회
                        {"&nbsp;&nbsp;→ <b>" + str(remain_valid) + "회 더 필요</b>" if not valid_ok else ""}<br>
                        ℹ️ <b>짧은 발화</b> (단어 2개 이하): {n_short}회
                        &nbsp;&nbsp;→ 워드클라우드에도 포함돼요 (단어 자체는 유효)
                    </div>
                    <div style="margin-top:0.7rem;font-size:0.82rem;color:#6B8FB5;">
                        💬 &nbsp;"오늘 많이 지치고 힘들었어"처럼 3단어 이상으로 이야기해 주시면<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;워드클라우드가 열려요. 짧은 대화는 자유롭게 하셔도 돼요!
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # ── 게이트 통과: 전체 발화(user_msgs)를 워드클라우드에 투입 ──
                # 짧은 발화도 단어 자체는 유효하므로 전체 포함
                wc_bytes, uniq_count = make_wordcloud_png(user_msgs)

                if wc_bytes and uniq_count >= MIN_CUMULATIVE_WORDS:
                    # ── 정상 표시 ──────────────────────────────────────
                    st.image(wc_bytes, use_container_width=True,
                             caption=f"대화에서 자주 언급된 단어 (고유 단어 {uniq_count}개 · 명사/동사/형용사/부사 포함 · 부정어·강조어 보존)")
                    if n_short > 0:
                        st.markdown(
                            f'<div style="font-size:0.8rem;color:#8A9AB5;margin-top:0.3rem;">'
                            f'ℹ️ 짧은 발화 {n_short}개도 워드클라우드에 포함되었어요.</div>',
                            unsafe_allow_html=True
                        )

                elif wc_bytes:
                    # ── 단어 수 부족: 클라우드는 생성됐지만 고유어 적음 ──
                    st.image(wc_bytes, use_container_width=True,
                             caption=f"대화에서 자주 언급된 단어 (현재 고유 단어 {uniq_count}개)")
                    st.markdown(f"""
                    <div style="background:#FFF8EC;border-radius:10px;padding:0.8rem 1rem;
                                margin-top:0.5rem;border-left:4px solid #C4956B;font-size:0.84rem;
                                color:#7A5A30;line-height:1.8;">
                        ⚠️ <b>단어 다양성이 아직 부족해요</b><br>
                        현재 고유 단어 <b>{uniq_count}개</b> / 권장 최소 <b>{MIN_CUMULATIVE_WORDS}개</b><br>
                        비슷한 단어를 반복하기보다 다양한 감정·상황 표현을 써보시면<br>
                        더 풍부한 워드클라우드를 볼 수 있어요.
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    # ── 빈도 필터에서 모두 탈락: 단어가 반복되지 않음 ──
                    from collections import Counter as _C
                    _sample_tokens = []
                    for _m in user_msgs:   # 전체 발화 기준으로 미리보기
                        _sample_tokens.extend(kiwi_tokenize(_m))
                    _freq_preview  = _C(_sample_tokens).most_common(5)
                    _preview_str   = "  ·  ".join(f"{w}({c}회)" for w, c in _freq_preview) if _freq_preview else "없음"

                    st.markdown(f"""
                    <div style="background:#F5F8FC;border-radius:12px;padding:1rem 1.2rem;
                                border:1px solid #D0DFF0;border-left:4px solid #8B7BAD;">
                        <div style="font-weight:700;font-size:0.9rem;color:#4A3A7A;margin-bottom:0.6rem;">
                            ☁️ 아직 워드클라우드를 그리기엔 단어 반복이 부족해요
                        </div>
                        <div style="font-size:0.84rem;color:#5A5A8A;line-height:1.9;">
                            ✅ 의미 있는 발화: <b>{n_valid}회</b><br>
                            ⏳ 현재 수집된 고유 단어: <b>{uniq_count}개</b><br>
                            ⏳ 동일 단어 반복 등장 횟수: 아직 부족
                            (워드클라우드는 같은 단어가 2번 이상 나올 때 표시돼요)<br>
                            📋 현재 가장 많이 나온 단어: <span style="color:#6096C8;">{_preview_str}</span>
                        </div>
                        <div style="margin-top:0.7rem;font-size:0.82rem;color:#7A7AAA;">
                            💡 비슷한 주제로 조금 더 이야기하면 단어가 반복되며 클라우드가 표시돼요.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── 대화 이력 ────────────────────────────────────────────
            st.markdown('<div class="sec-header">📋 대화 이력</div>', unsafe_allow_html=True)
            for i, content in enumerate(user_msgs):
                st.markdown(f"""
                <div style="background:#F5F8FC;border-radius:8px;padding:0.5rem 0.9rem;
                            margin-bottom:0.3rem;font-size:0.84rem;border-left:3px solid #6096C8;">
                    <span style="color:#A0AEC0;font-size:0.74rem;">#{i+1}번째 대화</span><br>
                    {content[:80]}{"..." if len(content)>80 else ""}
                </div>
                """, unsafe_allow_html=True)

        with dc2:
            # ── 파이차트: 상위 6개 ──────────────────────────────────
            st.markdown('<div class="sec-header">🥧 감정 분포 (상위 6개)</div>', unsafe_allow_html=True)
            if avg_probs:
                sorted_emotions = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
                top6       = sorted_emotions[:6]
                others_val = sum(e[1] for e in sorted_emotions[6:])
                pie_labels = [e[0] for e in top6] + (["기타"] if others_val>0.5 else [])
                pie_vals   = [e[1] for e in top6] + ([round(others_val,1)] if others_val>0.5 else [])
                pastel_colors = ["#6096C8","#8B7BAD","#5B9E7A","#C4956B","#C07070","#5B9999","#A0B8D0"]
                fig_pie = go.Figure(go.Pie(
                    labels=pie_labels, values=pie_vals, hole=0.44,
                    marker_colors=pastel_colors[:len(pie_labels)],
                    textinfo="label+percent", textfont_size=10,
                ))
                fig_pie.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                                      paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True, key="fig_pie")

                # ── Top 3 감정 카드 ─────────────────────────────────
                st.markdown('<div class="sec-header">🏆 가장 많이 감지된 감정 Top 3</div>', unsafe_allow_html=True)
                medals = ["🥇","🥈","🥉"]
                for rank,(emo,val) in enumerate(sorted_emotions[:3],1):
                    c = P_WARM if emo in HIGH_RISK else CHART_PRIMARY_COLOR
                    st.markdown(f"""
                    <div style="background:#F5F8FC;border-radius:10px;padding:0.7rem 1rem;
                                margin-bottom:0.4rem;border-left:4px solid {c};">
                        <span style="font-size:1.1rem;">{medals[rank-1]}</span>
                        <b style="color:{c};margin-left:0.4rem;">{emo}</b>
                        <span style="float:right;font-weight:700;color:{c};">{val:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── 고위험 감정 경고 ────────────────────────────────────
            high_risk_detected = [(e,v) for e,v in avg_probs_no_daily.items() if e in HIGH_RISK and v>5.0]
            if high_risk_detected:
                st.markdown("---")
                st.markdown('<div class="sec-header">⚠️ 주의 감정 감지</div>', unsafe_allow_html=True)
                for emo,val in high_risk_detected:
                    st.markdown(f"""
                    <div style="background:#FDF6F6;border-radius:8px;padding:0.5rem 0.8rem;
                                margin-bottom:0.3rem;border-left:3px solid {P_WARM};
                                font-size:0.84rem;color:#6A3030;">
                        ⚠️ <b>{emo}</b> 감정이 평균 <b>{val:.1f}%</b> 감지되었습니다.
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="crisis-box" style="margin-top:0.5rem;">
                    마음이 힘드실 때 연락할 수 있는 곳이 있어요.<br>
                    ☎ <b>자살예방 1393</b> (24시간)
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("📝 PHQ-9 검진하기 →", key="goto_care_nlp", type="primary"):
            st.session_state.page="📝 PHQ-9"; st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 3: 맞춤 케어
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "🌱 맞춤 케어":
    a = st.session_state.last_analysis; p = st.session_state.user_profile
    if not a and not st.session_state.phq9_done:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem 2rem 1rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">📋</div>
            <div style="font-weight:800;font-size:1.15rem;color:#3A5A9A;margin-bottom:0.6rem;">
                PHQ-9 우울 자가검진을 먼저 받아주세요
            </div>
            <div style="font-size:0.9rem;color:#5A7ABE;line-height:1.8;margin-bottom:1.4rem;">
                맞춤 케어는 PHQ-9 검진 결과를 기반으로 제공돼요.<br>
                9개 문항으로 약 2~3분이면 완료돼요.
            </div>
        </div>
        """, unsafe_allow_html=True)
        col_l, col_c, col_r = st.columns([1,2,1])
        with col_c:
            if st.button("📝 PHQ-9 자가검진 하기", type="primary", use_container_width=True):
                st.session_state.page = "📝 PHQ-9"; st.rerun()
    elif not a:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem 2rem 1rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">💬</div>
            <div style="font-weight:800;font-size:1.15rem;color:#3A5A9A;margin-bottom:0.6rem;">
                PHQ-9 검진 완료! 이제 일상 대화를 시작해 주세요
            </div>
            <div style="font-size:0.9rem;color:#5A7ABE;line-height:1.8;margin-bottom:1.4rem;">
                챗봇과 대화를 나누면 감정 분석 결과와 함께<br>
                더 정확한 맞춤 케어를 제공해 드려요.
            </div>
        </div>
        """, unsafe_allow_html=True)
        col_l2, col_c2, col_r2 = st.columns([1,2,1])
        with col_c2:
            if st.button("💬 일상 대화 시작하기", key="goto_chat_2", type="primary", use_container_width=True):
                st.session_state.page="💬 일상 대화"; st.rerun()
    else:
        age_group=p.get("age_group",""); phq9_done=st.session_state.phq9_done
        phq9_total=st.session_state.phq9_total if phq9_done else None
        def get_phq9_care_level(score):
            if score is None: return None
            if score>=15: return "고위험"
            elif score>=10: return "중증"
            elif score>=5: return "경증"
            return "양호"
        cl = get_phq9_care_level(phq9_total)
        if not phq9_done:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem 2rem 1rem;">
                <div style="font-size:3rem;margin-bottom:0.8rem;">📋</div>
                <div style="font-weight:800;font-size:1.15rem;color:#3A5A9A;margin-bottom:0.6rem;">
                    PHQ-9 우울 자가검진을 먼저 받아주세요
                </div>
                <div style="font-size:0.9rem;color:#5A7ABE;line-height:1.8;margin-bottom:1.4rem;">
                    맞춤 케어는 PHQ-9 검진 결과를 기반으로 제공돼요.<br>
                    9개 문항으로 약 2~3분이면 완료돼요.
                </div>
            </div>
            """, unsafe_allow_html=True)
            col_l, col_c, col_r = st.columns([1,2,1])
            with col_c:
                if st.button("📝 PHQ-9 자가검진 시작하기", type="primary", use_container_width=True):
                    st.session_state.page = "📝 PHQ-9"; st.rerun()
        elif cl == "양호":
            st.markdown(f'<div style="background:#EAF5EF;border-radius:14px;padding:1.5rem 1.8rem;border:1px solid #B0D9C0;text-align:center;"><div style="font-size:1.8rem;margin-bottom:0.4rem;">🌟</div><div style="font-weight:800;font-size:1rem;color:#8A5060;margin-bottom:0.3rem;">현재 마음 상태가 안정적이에요! (PHQ-9: {phq9_total}점)</div><div style="font-size:0.88rem;color:#8A5060;line-height:1.7;">규칙적인 일상과 사회적 교류를 유지해 주세요 😊</div></div>', unsafe_allow_html=True)
        else:
            cc1, cc2 = st.columns([1,1], gap="large")
            with cc1:
                if cl == "경증":
                    st.markdown(f'<div class="care-section"><div class="care-section-title">🟡 경도 — 주의 요망 (PHQ-9: {phq9_total}점)</div>', unsafe_allow_html=True)
                    for ti,de in [("😴 수면","매일 같은 시간에 자고 일어나세요. 7~8시간 목표."),("🍚 식사","하루 세 끼를 규칙적으로 드세요."),("🌅 일상","기상 후 햇빛 쬐기, 짧은 산책, 좋아하는 활동 하나를 루틴으로.")]:
                        st.markdown(f'<div class="self-care-item"><b>{ti}</b><br><span style="color:#5A7A96;font-size:0.81rem;">{de}</span></div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown('<div class="care-section"><div class="care-section-title">📞 상담 기관</div>', unsafe_allow_html=True)
                    for nm,nu,de in [("정신건강복지센터","1577-0199","무료 전화 상담"),("대학 상담센터","학교 홈페이지","재학생 무료"),("직장 EAP","HR 포털","직장인 무료"),("마음이음","www.mind-i.com","24시간 채팅")]:
                        st.markdown(f'<div style="background:white;border-radius:8px;padding:0.5rem 0.85rem;margin-bottom:0.3rem;border-left:3px solid #C4A0A0;font-size:0.83rem;"><b style="color:#7A5040;">{nm}</b> — {nu}<br><span style="color:#7A8FA6;font-size:0.78rem;">{de}</span></div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                elif cl == "중증":
                    st.markdown(f'<div class="care-section" style="border-left:4px solid {P_PURPLE};"><div class="care-section-title" style="color:#3A5A9A;">🟠 중등도 위험 — 전문가 상담 권고 (PHQ-9: {phq9_total}점)</div>', unsafe_allow_html=True)
                    for st_,de in [("1. 전화 예약","1577-0199로 초기 상담 예약"),("2. 비용","초기 상담 무료"),("3. 운영","평일 09:00~18:00 기본")]:
                        st.markdown(f'<div class="self-care-item" style="border-left-color:{P_PURPLE};"><b>{st_}</b> — {de}</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(f'''<div style="background:#EDF2FF;border-radius:12px;padding:0.9rem 1.1rem;border-left:4px solid #6096C8;font-size:0.86rem;color:#2D4A7A;line-height:1.9;">
                        ⏰ <b>이 상태가 2주 이상 지속된다면</b>, 꼭 전문가와 상의해 보세요.<br>
                        📋 <b>지역 자살예방센터 무료 집단 프로그램</b> — 심리교육·집단상담·회복 워크숍 (지역 센터에 문의)<br>
                        📞 정신건강복지센터 초기 상담 예약: <b>1577-0199</b>
                    </div>''', unsafe_allow_html=True)
                elif cl == "고위험":
                    st.markdown(f'<div style="background:#FDF6F6;border-radius:14px;padding:1.2rem 1.4rem;border:1.5px solid {P_WARM};margin-bottom:0.9rem;"><div style="font-size:0.97rem;font-weight:800;color:#6A3030;margin-bottom:0.7rem;">🔴 고위험 — 도움 받으실 수 있어요 (PHQ-9: {phq9_total}점)</div><div style="font-size:0.85rem;color:#6B3535;margin-bottom:0.9rem;line-height:1.7;">PHQ-9 점수 15점 이상입니다. 아래 번호로 연락해 보세요.</div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size:0.8rem;color:#2D4A7A;margin-bottom:0.5rem;">📲 아래 버튼을 클릭하면 바로 전화가 연결됩니다.</div>', unsafe_allow_html=True)
                    for nu,nm in [("1393","자살예방상담전화"),("1577-0199","정신건강위기상담전화")]:
                        st.markdown(f'<a class="crisis-btn" href="tel:{nu}">📞 {nu} — {nm} (24시간 무료)</a>', unsafe_allow_html=True)
                    st.markdown('''<div style="font-size:0.84rem;color:#6B3535;margin-top:0.7rem;line-height:1.8;">
                        📱 <b>지금은 앱보다 사람과 직접 통화하는 것이 더 안전합니다.</b><br>
                        전화가 어렵다면 문자(#1393)나 카카오톡 채널 <b>마음이음</b>을 이용하세요.
                    </div></div>''', unsafe_allow_html=True)
                    contact=p.get("contact","")
                    st.markdown('<div class="notify-box">💌 <b>가까운 분께 힘든 상태를 알려보세요.</b><br><i style="font-size:0.83rem;color:#5A7A9E;">"나 요즘 많이 힘들어. 잠깐 얘기할 수 있어?"</i></div>', unsafe_allow_html=True)
                    if contact: st.markdown(f'<div style="background:#FEF9E8;border-radius:10px;padding:0.6rem 0.9rem;margin-top:0.4rem;font-size:0.84rem;color:#7A5A20;border-left:4px solid {P_PINK};">📱 비상 연락처: <b>{contact}</b></div>', unsafe_allow_html=True)
            with cc2:
                st.markdown('<div class="sec-header">🆘 위기 상담 핫라인</div>', unsafe_allow_html=True)
                for nu,nm,de in [("1393","자살예방상담전화","24시간"),("1577-0199","정신건강위기상담","24시간"),("129","보건복지상담","복지 연계")]:
                    st.markdown(f'<div class="hotline"><span>📞</span><span class="hotline-num">{nu}</span><span><b>{nm}</b><br><span style="font-size:0.76rem;color:#7A8FA6;">{de}</span></span></div>', unsafe_allow_html=True)
                st.markdown('<div class="sec-header" style="margin-top:1rem;">📋 생애주기별 안내</div>', unsafe_allow_html=True)
                lifecycle={"20대":"청년 불씨 프로젝트, 대학 상담센터, 마인드부스터 Green","30대":"직장인 EAP, 지역 정신건강복지센터","40대":"중년 심리지원 프로그램","50대 이상":"노인 돌봄 서비스, 복지관 프로그램"}
                st.markdown(f'<div style="background:#E8F4FF;border-radius:10px;padding:0.7rem 0.9rem;font-size:0.86rem;line-height:1.7;margin-bottom:0.7rem;border-left:3px solid #6096C8;"><b style="color:#2D5A8E;">{age_group or "맞춤"} 추천</b><br><span style="color:#4A6A8E;">{lifecycle.get(age_group,"지역 정신건강복지센터 활용을 권장합니다.")}</span></div>', unsafe_allow_html=True)
                if p.get("interests"):
                    st.markdown('<div class="sec-header">🎯 관심사 기반 활동</div>', unsafe_allow_html=True)
                    ic={"영화":"영화 동아리·시네마테라피","음악":"음악치료·합창단","독서":"독서치료·북클럽","운동":"운동치료·요가 클래스","게임":"디지털 힐링·보드게임 소모임","요리":"요리 클래스·푸드테라피","여행":"치유 여행·탐방","반려동물":"동물매개치료","그림/예술":"미술치료·공방 클래스","기타":"지역 커뮤니티 탐색"}
                    for interest in p["interests"]: st.markdown(f'<div style="background:#EAF5EF;border-radius:8px;padding:0.45rem 0.85rem;margin-bottom:0.28rem;font-size:0.84rem;border-left:3px solid {P_ROSE};"><b style="color:{P_ROSE};">🎯 {interest}</b> — {ic.get(interest,"관련 활동 탐색")}</div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.button("📝 PHQ-9 우울 자가검진 →", type="primary", key="btn_care_to_phq"): st.session_state.page="📝 PHQ-9"; st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 4: PHQ-9
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "📝 PHQ-9":
    st.markdown(f'<div class="card"><div class="card-title">PHQ-9란?</div><div style="font-size:0.88rem;line-height:1.8;color:#374151;"><b>우울 자가검사 (PHQ-9)</b>는 9개 문항으로 구성된 자기보고식 설문지입니다. <b>10점 이상이면 전문가 상담을 권장합니다.</b><br><span style="color:{P_WARM};font-size:0.82rem;">⚠️ PHQ-9는 선별 검사이며 확진 도구가 아닙니다.</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-header">📝 지난 2주간 얼마나 자주 다음 문제를 경험하셨나요?</div>', unsafe_allow_html=True)
    answers = []
    for i,q in enumerate(PHQ9_QUESTIONS):
        if i==8: st.markdown(f'<div style="background:#FDF6F6;border-radius:8px;padding:0.55rem 0.9rem;margin:0.7rem 0 0.25rem;border-left:4px solid {P_WARM};font-size:0.82rem;color:#6A3030;">⚠️ 민감한 내용이 포함된 문항입니다.</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#F5F8FC;border-radius:10px;padding:0.65rem 0.95rem;margin-bottom:0.25rem;border-left:3px solid {P_CORAL};font-size:0.88rem;font-weight:500;">Q{i+1}. {q}</div>', unsafe_allow_html=True)
        val = st.radio(f"q{i+1}",options=[0,1,2,3],format_func=lambda x:["0 — 없음","1 — 2~6일","2 — 7~12일","3 — 거의 매일"][x],index=st.session_state.phq9_answers[i],key=f"phq9_{i}",horizontal=True,label_visibility="collapsed")
        answers.append(val); st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)
    if st.button("결과 확인하기", key="phq9_submit", type="primary", use_container_width=True):
        st.session_state.phq9_answers=answers; st.session_state.phq9_total=sum(answers); st.session_state.phq9_done=True; st.rerun()
    if st.session_state.phq9_done:
        total=st.session_state.phq9_total; saved_answers=st.session_state.phq9_answers
        sl=sb=sc=si=""
        for s_min,s_max,s_label,s_bg,s_color,s_inter in PHQ9_SEVERITY:
            if s_min<=total<=s_max: sl,sb,sc,si=s_label,s_bg,s_color,s_inter; break
        ihr=total>=10
        st.markdown("---"); st.markdown('<div class="sec-header">📋 PHQ-9 검진 결과</div>', unsafe_allow_html=True)
        rc1,rc2=st.columns(2); rc1.metric("PHQ-9 총점",f"{total}점 / 27점"); rc2.metric("우울 심각도",sl)
        st.markdown(f'<div style="background:{sb};border-radius:12px;padding:0.9rem 1.2rem;margin-top:0.4rem;border-left:5px solid {sc};"><b style="color:{sc};font-size:0.97rem;">{sl} ({total}점)</b><br><span style="color:{sc}AA;font-size:0.86rem;">권장 개입: {si}</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fig_g=go.Figure(go.Indicator(mode="gauge+number",value=total,number={"suffix":"점","font":{"size":34}},gauge={"axis":{"range":[0,27]},"bar":{"color":sc},"bgcolor":"white","steps":[{"range":[0,4],"color":"#A8D8B8"},{"range":[5,9],"color":"#F0D888"},{"range":[10,14],"color":"#F0B870"},{"range":[15,19],"color":"#E08868"},{"range":[20,27],"color":"#C86060"}],"threshold":{"line":{"color":sc,"width":3},"thickness":0.75,"value":total}},title={"text":sl,"font":{"size":15}}))
        fig_g.update_layout(height=240,margin=dict(l=20,r=20,t=38,b=8),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True, key="fig_phq_gauge")
        st.markdown('<div style="font-size:0.76rem;color:#A0AEC0;text-align:center;margin-top:-0.4rem;"><span style="color:#5B9E7A;">■</span> 양호(0~4) · <span style="color:#C4A830;">■</span> 경도(5~9) · <span style="color:#D08840;">■</span> 보통(10~14) · <span style="color:#C06848;">■</span> 중증(15~19) · <span style="color:#B04040;">■</span> 위험(20~27)</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        sbl=["우울감","흥미 저하","수면","식욕","행동","피로","죄책감","집중","자해"]
        fig_b2=go.Figure(go.Bar(x=sbl,y=saved_answers,marker_color="#6096C8",text=saved_answers,textposition="outside"))
        fig_b2.update_layout(height=240,margin=dict(l=0,r=0,t=8,b=28),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",yaxis=dict(range=[0,3.5],tickvals=[0,1,2,3],ticktext=["없음","2~6일","7~12일","거의 매일"],gridcolor="#EEF4FA"),xaxis=dict(gridcolor="#EEF4FA"))
        st.plotly_chart(fig_b2, use_container_width=True, key="fig_phq_bar")
        if saved_answers[8]>=1:
            st.markdown(f'<div class="crisis-box" style="margin-top:0.9rem;">마음이 많이 힘드신가요? 혼자 감당하지 않아도 됩니다.<br>☎ <b>자살예방상담전화 1393</b> (24시간) · ☎ <b>정신건강위기상담 1577-0199</b></div>', unsafe_allow_html=True)
        elif ihr:
            st.markdown(f'<div style="background:#EDF2FF;border-radius:10px;padding:0.7rem 0.9rem;font-size:0.84rem;color:#3A5A9A;border-left:4px solid {P_BLUE};margin-top:0.5rem;">PHQ-9 10점 이상입니다. 정신건강의학과나 상담 센터 방문을 권장합니다.</div>', unsafe_allow_html=True)
        st.markdown("---")
        cg1,cg2=st.columns(2)
        with cg1:
            if st.button("🏥 공공 서비스 →",type="primary",key="btn_phq_to_service",use_container_width=True): st.session_state.page="🏥 공공 서비스"; st.rerun()
        with cg2:
            if st.button("🌱 맞춤 케어 →",key="btn_phq_to_care",use_container_width=True): st.session_state.page="🌱 맞춤 케어"; st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 5: 공공 서비스
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "🏥 공공 서비스":
    sc1,sc2=st.columns([1,1],gap="large")
    def pbtn(href,label,bg,tc="white"):
        return f'<a href="{href}" target="_blank" style="display:block;background:{bg};color:{tc};padding:0.6rem 1rem;border-radius:10px;text-decoration:none;font-size:0.87rem;font-weight:600;text-align:center;margin-bottom:0.4rem;border:1px solid rgba(0,0,0,0.06);">{label}</a>'
    with sc1:
        st.markdown('<div class="sec-header">🆘 위기 상담 핫라인</div>', unsafe_allow_html=True)
        for nu,nm,de in [("1393","자살예방상담전화","24시간 무료"),("1577-0199","정신건강위기상담전화","24시간"),("129","보건복지상담센터","복지 연계")]:
            st.markdown(f'<div class="hotline"><span>📞</span><span class="hotline-num">{nu}</span><span><b>{nm}</b><br><span style="font-size:0.79rem;color:#7A8FA6;">{de}</span></span></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="sec-header">🏥 내 지역 센터 찾기</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#E8F4FF;border-radius:12px;padding:0.9rem 1.1rem;border:1px solid #BFD8EE;margin-bottom:0.8rem;font-size:0.86rem;line-height:1.7;border-left:3px solid #6096C8;"><b style="color:#2D5A8E;">📍 시·군·구 단위 상세 검색 가능</b></div>', unsafe_allow_html=True)

        st.markdown(pbtn(SIGUNGU_SEARCH_URL,"🔍 시·군·구 정신건강복지센터 검색","#6096C8","white"), unsafe_allow_html=True)
        st.markdown(pbtn("https://map.naver.com/v5/search/%EC%A0%95%EC%8B%A0%EA%B1%B4%EA%B0%95%EB%B3%B5%EC%A7%80%EC%84%BC%ED%84%B0","📍 내 주변 센터 (네이버 지도)","#5B9E7A","white"), unsafe_allow_html=True)
    with sc2:
        st.markdown('<div class="sec-header">🌐 주요 기관 바로가기</div>', unsafe_allow_html=True)
        def pbtn2(href,label,bg,tc="white"): return f'<a href="{href}" target="_blank" style="display:block;background:{bg};color:{tc};padding:0.5rem 0.9rem;border-radius:9px;text-decoration:none;font-size:0.84rem;font-weight:600;text-align:center;margin-bottom:0.35rem;border:1px solid rgba(0,0,0,0.06);">{label}</a>'
        st.markdown(pbtn2("https://www.ncmh.go.kr","🏛️ 국립정신건강센터","#6096C8","white"), unsafe_allow_html=True)
        st.markdown(pbtn2("https://www.mentalhealth.go.kr","💻 국가정신건강정보포털","#8B7BAD","white"), unsafe_allow_html=True)
        st.markdown(pbtn2("https://www.bokjiro.go.kr","🤝 복지로 — 복지서비스 통합 검색","#5B9999","white"), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="sec-header">🌱 지역사회 지원 프로그램</div>', unsafe_allow_html=True)
        for prog in COMMUNITY_PROGRAMS:
            st.markdown(f'<div style="background:white;border-radius:12px;padding:0.85rem 1rem;margin-bottom:0.5rem;border-left:4px solid {P_PEACH};border:1px solid #E5EEF7;"><div style="font-weight:700;font-size:0.88rem;color:#2D3748;margin-bottom:0.25rem;">{prog["name"]}</div><div style="font-size:0.81rem;color:#8A7060;margin-bottom:0.4rem;line-height:1.55;">{prog["desc"]}</div><a href="{prog["link"]}" target="_blank" style="background:{P_PURPLE};color:white;padding:0.25rem 0.7rem;border-radius:6px;text-decoration:none;font-size:0.78rem;font-weight:600;">🔗 {prog["link_text"]}</a></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="sec-header">🎯 PHQ-9 결과 기반 추천</div>', unsafe_allow_html=True)
        if st.session_state.phq9_done:
            pt=st.session_state.phq9_total
            if pt>=15: st.markdown(f'<div style="background:#FDF6F6;border-radius:10px;padding:0.7rem 0.9rem;font-size:0.85rem;color:#6A3030;border-left:4px solid {P_WARM};">🔴 고위험 ({pt}점) — 자살예방상담전화(1393)에 연락해 보세요.</div>', unsafe_allow_html=True)
            elif pt>=10: st.markdown(f'<div style="background:#EDF2FF;border-radius:10px;padding:0.7rem 0.9rem;font-size:0.85rem;color:#3A5A9A;border-left:4px solid {P_PURPLE};">🟠 중등도 ({pt}점) — 지역 정신건강복지센터 방문을 권장합니다.</div>', unsafe_allow_html=True)
            elif pt>=5: st.markdown(f'<div style="background:#E8F4FF;border-radius:10px;padding:0.7rem 0.9rem;font-size:0.85rem;color:#2A5090;border-left:4px solid #6096C8;">🟡 경도 ({pt}점) — 마인드부스터 Green 등을 활용해 보세요.</div>', unsafe_allow_html=True)
            else: st.markdown(f'<div style="background:#EAF5EF;border-radius:10px;padding:0.7rem 0.9rem;font-size:0.85rem;color:#8A5060;border-left:4px solid {P_ROSE};">🟢 정상 ({pt}점) — 긍정적인 사회 참여를 유지하세요.</div>', unsafe_allow_html=True)
        else: st.markdown('<div style="background:#EDF2FF;border-radius:12px;padding:0.9rem 1.2rem;border-left:4px solid #6096C8;font-size:0.88rem;color:#3A5A9A;">📋 PHQ-9 자가검진을 완료하면 현재 상태에 맞는 서비스를 추천해 드릴 수 있어요.</div>', unsafe_allow_html=True)
    st.markdown("---")
    pb1,pb2=st.columns(2)
    with pb1:
        if st.button("📝 PHQ-9 →",key="goto_phq_1",type="primary",use_container_width=True): st.session_state.page="📝 PHQ-9"; st.rerun()
    with pb2:
        if st.button("🌱 맞춤 케어 →",key="goto_care_2",use_container_width=True): st.session_state.page="🌱 맞춤 케어"; st.rerun()


# ═══════════════════════════════════════════════════════════════
# 상담사 대시보드
# ═══════════════════════════════════════════════════════════════
elif st.session_state.page == "🩺 상담사 대시보드":
    st.markdown('<div style="background:linear-gradient(135deg,#5B82B5 0%,#7B9FD0 55%,#8B7BAD 100%);border-radius:14px;padding:0.9rem 1.4rem;color:white;margin-bottom:1.1rem;"><div style="font-size:1.05rem;font-weight:900;margin-bottom:0.2rem;">🩺 상담사 전용 대시보드</div><div style="font-size:0.82rem;opacity:0.92;">대화 단위 위험도 · 표현 패턴 · 감정 추이 분석</div></div>', unsafe_allow_html=True)
    rh=st.session_state.risk_history; p=st.session_state.user_profile
    if not rh:
        st.markdown('<div class="empty-state">🩺<br>대화 데이터가 없습니다.<br>1단계 일상 대화를 진행하면 결과를 확인할 수 있습니다.</div>', unsafe_allow_html=True)
        if st.button("💬 일상 대화로 이동",type="primary"): st.session_state.page="💬 일상 대화"; st.rerun()
    else:
        tt=len(rh); ht=sum(1 for r in rh if r["score"]>=60 or any(e in HIGH_RISK for e in r.get("top_emos",[])))
        mt=sum(1 for r in rh if 35<=r["score"]<60); av=sum(r["score"] for r in rh)/tt
        sm1,sm2,sm3,sm4=st.columns(4)
        sm1.metric("총 대화 턴",f"{tt}회"); sm2.metric("🔴 고위험 턴",f"{ht}회",f"{ht/tt*100:.0f}%")
        sm3.metric("🟠 중등도 턴",f"{mt}회"); sm4.metric("평균 위험도",f"{av:.1f}점")
        dc1,dc2=st.columns([1.4,1],gap="large")
        with dc1:
            st.markdown('<div class="sec-header">📈 위험도 타임라인</div>', unsafe_allow_html=True)
            def s2c(s):
                if s>=60: return P_WARM
                elif s>=35: return P_PINK
                elif s>=15: return "#C4A857"
                return P_ROSE

            _db_has_ts = rh and isinstance(rh[0], dict) and "ts" in rh[0]

            def _db_turn_fig(records):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[r["turn"] for r in records], y=[r["score"] for r in records],
                    mode="lines+markers",
                    line=dict(color="#8B7BAD",width=2),
                    marker=dict(color=[s2c(r["score"]) for r in records],size=10,
                                line=dict(color="white",width=1.5)),
                    hovertemplate="턴 %{x}<br>위험도: %{y:.1f}<extra></extra>",
                ))
                fig.add_hrect(y0=60,y1=100,fillcolor=P_WARM,opacity=0.06,line_width=0)
                fig.add_hrect(y0=35,y1=60, fillcolor=P_PINK,opacity=0.06,line_width=0)
                fig.update_layout(height=200,margin=dict(l=0,r=0,t=8,b=28),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="대화 턴",gridcolor="#EEF4FA",dtick=1),
                    yaxis=dict(title="위험도",gridcolor="#EEF4FA",range=[0,100]))
                return fig

            def _db_dt_fig(records):
                if not records: return None
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[r["ts"] for r in records], y=[r["score"] for r in records],
                    mode="lines+markers",
                    line=dict(color="#8B7BAD",width=2),
                    marker=dict(color=[s2c(r["score"]) for r in records],size=10,
                                line=dict(color="white",width=1.5)),
                    text=[r["text"][:20]+"…" if len(r["text"])>20 else r["text"] for r in records],
                    hovertemplate="%{x|%m/%d %H:%M}<br>위험도: %{y:.1f}<br>%{text}<extra></extra>",
                ))
                fig.add_hrect(y0=60,y1=100,fillcolor=P_WARM,opacity=0.06,line_width=0)
                fig.add_hrect(y0=35,y1=60, fillcolor=P_PINK,opacity=0.06,line_width=0)
                fig.update_layout(height=200,margin=dict(l=0,r=0,t=8,b=28),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="시간",gridcolor="#EEF4FA",
                               rangeslider=dict(visible=True,thickness=0.06)),
                    yaxis=dict(title="위험도",gridcolor="#EEF4FA",range=[0,100]))
                return fig

            if _db_has_ts:
                from datetime import timedelta as _td
                _now2 = datetime.now()
                _dbtab_turn,_dbtab1,_dbtab7,_dbtab14,_dbtab30 = st.tabs(["대화턴","1일","7일","14일","30일"])
                with _dbtab_turn:
                    st.plotly_chart(_db_turn_fig(rh), use_container_width=True, key="db_tl_turn")
                with _dbtab1:
                    _dr1=[r for r in rh if (_now2-r["ts"]).days<1]
                    f=_db_dt_fig(_dr1)
                    st.plotly_chart(f, use_container_width=True, key="db_tl_1d") if f else st.info("오늘 기록 없음")
                with _dbtab7:
                    _dr7=[r for r in rh if (_now2-r["ts"]).days<7]
                    f=_db_dt_fig(_dr7)
                    st.plotly_chart(f, use_container_width=True, key="db_tl_7d") if f else st.info("최근 7일 기록 없음")
                with _dbtab14:
                    _dr14=[r for r in rh if (_now2-r["ts"]).days<14]
                    f=_db_dt_fig(_dr14)
                    st.plotly_chart(f, use_container_width=True, key="db_tl_14d") if f else st.info("최근 14일 기록 없음")
                with _dbtab30:
                    _dr30=[r for r in rh if (_now2-r["ts"]).days<30]
                    f=_db_dt_fig(_dr30)
                    st.plotly_chart(f, use_container_width=True, key="db_tl_30d") if f else st.info("최근 30일 기록 없음")
                st.markdown('<div style="font-size:0.78rem;color:#8A9AB5;margin-top:0.2rem;">💡 그래프를 드래그하거나 하단 슬라이더로 날짜를 좁혀 세부 대화를 확인하세요.</div>',unsafe_allow_html=True)
            else:
                st.plotly_chart(_db_turn_fig(rh), use_container_width=True, key="db_tl_turn_only")
            st.markdown('<div class="sec-header">📋 턴별 위험도 상세</div>', unsafe_allow_html=True)
            for r in rh:
                sc=r["score"]
                if sc>=60: badge='<span class="risk-badge-high">🔴 고위험</span>'
                elif sc>=35: badge='<span class="risk-badge-mid">🟠 중등도</span>'
                elif sc>=15: badge='<span class="risk-badge-low">🟡 경도</span>'
                else: badge='<span class="risk-badge-safe">🟢 양호</span>'
                st.markdown(f'<div style="background:#F5F8FC;border-radius:9px;padding:0.55rem 0.9rem;margin-bottom:0.3rem;font-size:0.84rem;display:flex;align-items:center;gap:0.5rem;"><span style="font-weight:700;color:{P_PEACH};min-width:36px;">#{r["turn"]}</span>{badge}<span style="color:#5A7A96;flex:1;">{r["text"]}{"..." if len(r["text"])>=60 else ""}</span></div>', unsafe_allow_html=True)
        with dc2:
            st.markdown('<div class="sec-header">🔍 감정 발생 패턴</div>', unsafe_allow_html=True)
            ae={}
            for r in rh:
                for emo,prob in r["top_emos"]: ae[emo]=ae.get(emo,0)+prob
            if ae:
                for emo,tot in sorted(ae.items(),key=lambda x:x[1],reverse=True)[:8]:
                    ir=emo in HIGH_RISK; c=P_WARM if ir else P_CORAL
                    bw=max(int(min(tot/tt*100,100)*1.8),4)
                    tag=f'<span style="background:#F5E5E5;color:#6A3030;border-radius:5px;padding:0.12rem 0.45rem;font-size:0.76rem;font-weight:600;">⚠ 위험</span>' if ir else ""
                    st.markdown(f'<div style="background:white;border-radius:8px;padding:0.48rem 0.75rem;margin-bottom:0.3rem;border-left:3px solid {c};"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.18rem;"><span style="font-weight:600;font-size:0.84rem;color:{c};">{emo}</span>{tag}<span style="font-size:0.76rem;color:#A0AEC0;">{tot*100:.0f}% 누적</span></div><div style="background:#EEF4FA;border-radius:4px;height:4px;"><div style="background:{c};height:4px;border-radius:4px;width:{bw}%;max-width:100%;"></div></div></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-header" style="margin-top:0.9rem;">👤 내담자 정보</div>', unsafe_allow_html=True)
            if p:
                for label,val in [("닉네임",p.get("nickname","—")),("연령대",p.get("age_group","—")),("성별",p.get("gender","—")),("직업",p.get("occupation","—")),("거주지",p.get("region","—")),("관심사",", ".join(p.get("interests",[]))or"—"),("비상연락",p.get("contact","—"))]:
                    st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.83rem;padding:0.28rem 0;border-bottom:1px solid #EEF4FA;"><span style="color:#8A7060;">{label}</span><span style="font-weight:600;color:#2D3748;">{val}</span></div>', unsafe_allow_html=True)
            if st.session_state.phq9_done:
                tp=st.session_state.phq9_total
                st.markdown(f'<div style="background:#F0F6FF;border-radius:10px;padding:0.7rem 0.9rem;margin-top:0.7rem;border-left:4px solid #6096C8;font-size:0.84rem;">📋 PHQ-9: <b>{tp}점</b> {"— 🔴 전문가 의뢰" if tp>=15 else ("— 🟠 상담 권고" if tp>=10 else ("— 🟡 주의" if tp>=5 else "— 🟢 정상"))}</div>', unsafe_allow_html=True)
        st.markdown("---")
        dd1,dd2=st.columns(2)
        with dd1:
            if st.button("💬 대화 화면으로",use_container_width=True): st.session_state.page="💬 일상 대화"; st.rerun()
        with dd2:
            if st.button("🌱 맞춤 케어 보기",type="primary",use_container_width=True): st.session_state.page="🌱 맞춤 케어"; st.rerun()
