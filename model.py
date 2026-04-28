"""
model.py — AI 로직 모듈
========================
우울 점수 정의 (논문 스타일):
  - P(우울) = 1 - P(일상)
  - 모델이 출력하는 일상 확률이 낮을수록 우울 점수가 높아짐
  - "이 문장이 우울 발화일 확률"을 0~100점으로 표현
  - 근거: 오재동·오하영(2022) 논문 기반 우울 감정 탐지 모델

챗봇: GPT-4o mini (OpenAI API)
  필수 환경변수:  OPENAI_API_KEY=sk-...
  패키지 설치:    pip install openai
  키 설정 방법:
    Windows PowerShell : $env:OPENAI_API_KEY = "sk-..."
    Windows CMD        : set OPENAI_API_KEY=sk-...
    macOS/Linux        : export OPENAI_API_KEY="sk-..."
"""

import json
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ──────────────────────────────────────────────────
# 0. 상수
# ──────────────────────────────────────────────────
MODEL_DIR = "./saved_models/KLUBERT_Dataset1"

EMOTION_MAP = {
    '우울감':0,  '슬픔':1,       '외로움':2,    '분노':3,    '무기력':4,
    '감정조절이상':5, '상실감':6, '식욕저하':7,  '식욕증가':8, '불면':9,
    '초조함':10, '일상':11,      '피로':12,     '죄책감':13,  '집중력저하':14,
    '자신감저하':15, '자존감저하':16, '절망감':17, '자살충동':18, '불안':19,
}
INV_MAP = {v: k for k, v in EMOTION_MAP.items()}

# 일상 레이블 인덱스 (점수 계산 핵심)
DAILY_IDX = 11


# ──────────────────────────────────────────────────
# 1. 전처리
# ──────────────────────────────────────────────────

# 구어체 → 표준어 매핑 사전
SLANG_MAP = {
    # 자살/죽음 관련
    r"주글래|주겄다|주것다|죽겄다|주거버릴|죽어버릴래|죽어버리고|죽어버릴|죽어버려|주거|죽을래": "죽고 싶어",
    r"자살할래|자살하고싶어|자살하고싶다|자살충동|목숨끊|스스로목숨": "자살충동",
    r"사라지고싶|없어지고싶|없어져버리|사라져버리": "사라지고 싶어",
    r"살기싫|살기 싫|살고싶지않|살고 싶지 않": "살기 싫어",
    r"나왜살|나 왜 살|왜살지|왜 살지|살아서뭐해|살아서 뭐해|살아뭐해": "왜 살아야 하지",

    # 무기력/절망 관련
    r"힘들어죽겠|힘들어 죽겠|힘들어죽을|힘들어 죽을": "너무 힘들어",
    r"못하겠어|못하겠다|못해먹겠|더이상못해|더 이상 못해": "더 이상 못 하겠어",
    r"포기할래|포기하고싶|포기하고 싶": "포기하고 싶어",
    r"지쳐버|지쳐죽|완전지쳐|너무지쳐": "너무 지쳤어",
    r"아무것도하기싫|아무것도 하기 싫|아무것도안하고싶|아무 것도 하기 싫": "아무것도 하기 싫어",

    # 외로움/고립 관련
    r"혼자죽|혼자서죽|아무도없어|아무도 없어|아무도날|아무도 날": "아무도 없어 외로워",
    r"왕따|따돌림|무시당해|무시당하고있": "외로워 고립됐어",

    # 슬픔/우울 관련
    r"너무슬퍼|너무 슬퍼|슬퍼죽겠|슬퍼 죽겠": "너무 슬퍼",
    r"우울해죽겠|우울해서죽겠|너무우울|너무 우울": "너무 우울해",
    r"눈물이나|눈물 나|울고싶어|울고 싶어|울고싶다": "슬퍼서 울고 싶어",

    # 불안/초조 관련
    r"불안해죽겠|너무불안|너무 불안|미칠것같|미칠 것 같": "너무 불안해",
    r"떨려죽겠|초조해죽겠": "너무 초조해",
}

# 고위험 키워드 직접 감지
HIGH_RISK_KEYWORDS = [
    "죽고 싶", "죽을래", "주글래", "자살", "자해", "목숨",
    "사라지고 싶", "없어지고 싶", "살기 싫", "왜 살아야",
    "죽어버리", "스스로 목숨", "자살충동", "죽겠어", "죽겠다",
    "살고 싶지 않", "살기싫", "나왜살", "왜살지",
]


def spell_correct(text: str) -> str:
    """
    py-hanspell로 맞춤법 교정을 시도합니다.
    네트워크 오류나 실패 시 원본 텍스트를 반환합니다.
    """
    try:
        from hanspell import spell_checker
        result = spell_checker.check(text)
        return result.checked
    except Exception:
        return text


# ──────────────────────────────────────────────────
# 불용어 사전
# streamlit에서도 import해서 워드클라우드에 재사용
# ──────────────────────────────────────────────────

# 부정어 (제거 X) — 감정 방향성 해석에 필수
NEGATION_WORDS = {"안", "못", "없다", "없어", "아니", "아니다", "아닙니다"}

# 감정 강조어 (제거 X) — 감정 강도 판단에 필수
EMOTION_EMPHASIS = {"너무", "정말", "매우", "엄청", "진짜", "완전", "정말로", "참", "왕"}

# 일반 불용어 (제거 O)
_KO_STOPWORDS = {
    "그리고", "그런데", "하지만", "그래서", "그러면", "그래도", "근데", "그냥",
    "조금", "약간", "되게", "아주", "많이", "좀",  # "너무", "정말" 제거 (강조어)
    "오늘", "지금", "이번", "저번", "언제", "어디", "무엇", "어떻게", "어떤",
    "나는", "내가", "제가", "저는", "우리", "당신", "그는", "그녀", "그들",
    "것", "수", "때", "중", "등", "및", "또", "또는", "그것", "이것",
    "합니다", "했습니다", "있습니다", "없습니다", "됩니다",  # "없어" 제거 (부정어)
    "하는", "해서", "하고", "하면", "하면서",
    "같아", "같은", "같이", "이렇게", "그렇게",
    "뭐", "왜", "더", "덜", "잘",  # "못", "안" 제거 (부정어)
    "이", "가", "을", "를", "은", "는", "에", "도", "만", "로", "으로",
    "에서", "부터", "까지", "라고", "이라고", "라는", "이라는",
}


def remove_stopwords_with_exceptions(text: str) -> str:
    """
    부정어·감정강조어를 보존하면서 일반 불용어만 제거한다.
    워드클라우드 생성 시 감정 방향성과 강도를 유지하기 위한 함수.

    - 부정어 ("안", "못", "없어", "아니"): 제거 X → 감정 방향성 유지
    - 감정강조어 ("너무", "정말", "매우", "엄청"): 제거 X → 감정 강도 반영
    - 일반 불용어: 제거 O

    Returns:
        str: 필터링된 텍스트
    """
    tokens = text.split()
    filtered = []
    for t in tokens:
        if t in NEGATION_WORDS or t in EMOTION_EMPHASIS:
            filtered.append(t)
        elif t not in _KO_STOPWORDS:
            filtered.append(t)
    return " ".join(filtered)


def count_valid_words(text: str) -> int:
    """
    불용어 제거 후 유효한 단어 개수를 반환한다.
    워드클라우드 노출 조건 판단에 사용.

    Args:
        text: 입력 텍스트

    Returns:
        int: 유효 단어 개수
    """
    cleaned = remove_stopwords_with_exceptions(text)
    return len(cleaned.split())


def remove_stopwords(text: str) -> str:
    """공백 기준 토크나이징 후 불용어를 제거한다. (학습 전처리용)"""
    tokens = text.split()
    return " ".join(t for t in tokens if t not in _KO_STOPWORDS)


def preprocess_text(text: str) -> tuple[str, bool]:
    """
    입력 텍스트 전처리:
        1. 공백 정규화
        2. 고위험 키워드 원본에서 먼저 체크
        3. SLANG_MAP 구어체 → 표준어 변환
        4. py-hanspell 맞춤법 교정
        5. 교정 후 고위험 키워드 재체크
        6. 특수문자 정리
    """
    # 1. 공백 정규화
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. 고위험 키워드 원본에서 먼저 체크
    raw_no_space = re.sub(r'\s+', '', text)
    is_high_risk = any(
        re.sub(r'\s+', '', kw) in raw_no_space
        for kw in HIGH_RISK_KEYWORDS
    )

    # 3. SLANG_MAP 구어체 → 표준어 변환
    no_space = re.sub(r'\s+', '', text)
    for pattern, replacement in SLANG_MAP.items():
        if re.search(pattern, no_space):
            text = text + " " + replacement
            if any(kw in replacement for kw in ["죽", "자살", "사라", "살기 싫", "왜 살"]):
                is_high_risk = True

    # 3-1. 불용어 제거 (훈련 전처리와 동일하게 적용)
    text = remove_stopwords(text)

    # 4. py-hanspell 맞춤법 교정
    corrected = spell_correct(text)

    # 5. 교정 후 고위험 키워드 재체크
    if not is_high_risk:
        corrected_no_space = re.sub(r'\s+', '', corrected)
        is_high_risk = any(
            re.sub(r'\s+', '', kw) in corrected_no_space
            for kw in HIGH_RISK_KEYWORDS
        )

    # 6. 특수문자 정리
    corrected = re.sub(r'[^\w\s?!.,~]', ' ', corrected)
    corrected = re.sub(r'\s+', ' ', corrected).strip()

    return corrected, is_high_risk


# ──────────────────────────────────────────────────
# 2. 모델 로드
# ──────────────────────────────────────────────────
def load_model(model_dir: str = MODEL_DIR):
    """저장된 KLUE-BERT 모델을 로드합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_cfg_path = os.path.join(model_dir, "run_config.json")
    with open(run_cfg_path, encoding="utf-8") as f:
        run_cfg = json.load(f)

    inv_map = {int(v): k for k, v in run_cfg["emotion_map"].items()}

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    return model, tokenizer, inv_map, run_cfg, device


# ──────────────────────────────────────────────────
# 3. 우울 점수 산출
# ──────────────────────────────────────────────────
def get_depression_score(
    text: str,
    model,
    tokenizer,
    device,
    inv_map: dict,
    max_len: int = 64,
    threshold: float = 3.0,
) -> dict:
    """
    문장 하나를 받아 전처리 → 감정 분류 → 우울 점수를 계산합니다.

    우울 점수 정의:
        score = (1 - P(일상)) × 100

        - P(일상): 모델이 해당 문장을 '일상' 발화로 분류할 확률
        - 일상 확률이 높을수록(=우울하지 않을수록) 점수가 낮아짐
        - 일상 확률이 낮을수록(=우울 발화일수록) 점수가 높아짐
        - 근거: 오재동·오하영(2022) 우울 감정 탐지 논문 기반
    """
    # ── 전처리 ──────────────────────────────────────
    normalized_text, is_high_risk = preprocess_text(text)

    model.eval()
    enc = tokenizer(
        normalized_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits.squeeze(0).cpu()

    probs = F.softmax(logits, dim=0).numpy()

    # ── 우울 점수: P(우울) = 1 - P(일상) ────────────
    daily_prob = float(probs[DAILY_IDX])
    score = round((1.0 - daily_prob) * 100, 1)
    score = min(max(score, 0.0), 100.0)

    # ── 고위험 키워드 감지 시 점수 보정 ──────────────
    if is_high_risk:
        score = max(score, 65.0)
        probs[18] = max(probs[18], 0.35)

    # ── 다중 레이블 ──────────────────────────────────
    multi = [inv_map[i] for i, v in enumerate(logits.numpy()) if v > threshold]
    if is_high_risk and "자살충동" not in multi:
        multi.append("자살충동")

    # ── 위험 등급 ────────────────────────────────────
    if is_high_risk or "자살충동" in multi or probs[18] > 0.3:
        level = "🔴 고위험"
    elif score >= 60:
        level = "🔴 고위험"
    elif score >= 35:
        level = "🟠 중증"
    elif score >= 15:
        level = "🟡 경증"
    else:
        level = "🟢 양호"

    # ── 상위 3개 감정 ────────────────────────────────
    top3 = [
        (inv_map[i], round(float(probs[i]) * 100, 1))
        for i in np.argsort(probs)[::-1][:3]
    ]

    return {
        "text":  text,
        "score": score,
        "level": level,
        "top3":  top3,
        "multi": multi,
        "probs": probs,
    }


# ──────────────────────────────────────────────────
# 4. GPT-4o mini 챗봇 응답
#
# 필수 환경변수 설정 후 streamlit 실행:
#   Windows PowerShell : $env:OPENAI_API_KEY = "sk-..."
#   Windows CMD        : set OPENAI_API_KEY=sk-...
#   macOS/Linux        : export OPENAI_API_KEY="sk-..."
#
# 패키지 설치:
#   pip install openai
# ──────────────────────────────────────────────────
def get_chatbot_response(
    user_text: str,
    analysis: dict,
    conversation_history: list,
    persona_system: str = "",
) -> str:
    """
    KLUEBERT 분석 결과를 GPT-4o mini 프롬프트에 포함해서
    공감형 상담 챗봇 응답을 생성합니다.

    매개변수:
        user_text            — 사용자 입력 문장
        analysis             — get_depression_score() 반환값
        conversation_history — 이전 대화 목록 [{"role":"user","content":"..."}, ...]
        persona_system       — (선택) 페르소나 system 프롬프트
                               예: "당신은 '지우'라는 전문 심리 상담사입니다."
    반환값:
        str — GPT-4o mini 응답 텍스트
    """
    from openai import OpenAI

    client = OpenAI()  # OPENAI_API_KEY 환경변수에서 자동 로드

    top3_str  = ", ".join(f"{emo}({prob}%)" for emo, prob in analysis["top3"])
    multi_str = ", ".join(analysis["multi"]) if analysis["multi"] else "없음"
    level     = analysis["level"]
    score     = analysis["score"]

    # 페르소나가 있으면 맨 앞에 붙여 개성 반영
    persona_block = f"{persona_system}\n\n" if persona_system.strip() else ""

    system_prompt = f"""{persona_block}당신은 공감 능력이 뛰어난 심리 상담 챗봇입니다.
사용자의 말에 귀 기울이고, 따뜻하게 공감하며 대화를 이어가세요.

[현재 사용자 감정 분석 결과 — KLUEBERT 모델 출력]
- 주요 감지 감정: {top3_str}
- 복합 감정 (다중 레이블): {multi_str}
- 우울 위험 점수: {score}/100점
- 위험 등급: {level}

[응답 지침]
1. 먼저 사용자의 감정에 진심으로 공감하세요.
2. 위험 등급이 🟠 중증 이상이면 전문 상담을 조심스럽게 권유하세요.
3. 위험 등급이 🔴 고위험이거나 자살충동 감정이 감지되면 반드시 위기 상담 정보를 안내하세요:
   - 자살예방상담전화: 1393 (24시간)
   - 정신건강위기상담전화: 1577-0199 (24시간)
4. 질문은 한 번에 하나만 하세요.
5. 답변은 3~5문장 내외로 간결하게 작성하세요.
6. 절대 진단을 내리거나 약을 권유하지 마세요."""

    # OpenAI Chat Completions API 호출
    messages = (
        [{"role": "system", "content": system_prompt}]
        + conversation_history
        + [{"role": "user", "content": user_text}]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content
