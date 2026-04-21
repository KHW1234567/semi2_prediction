# 마음 온도계 — Streamlit 실행 가이드

## 📁 파일 구조

```
MedicalDA05_semi2/
├── app.py                          ← 이 파일 실행
├── model.py                       
├── saved_models/
│   └── KLUEBERT_Dataset1/         ← 모델 저장된 폴더
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       ├── run_config.json         
│       └── history.json
└── README.md
```

---

## 🚀 실행 방법


### 1단계: 모델 다운로드

https://drive.google.com/drive/u/0/folders/1I5b3VqcPctEbg7D3TPhTGMeaORIchdAn

구글 드라이브에서 KLUEBERT_Dataset1 폴더 통째로 다운로드

클론한 MedicalDA05_semi2 폴더 안에 saved_models 폴더 만들어서 다운 받은거 넣기


### 2단계: 패키지 설치

```bash
pip install streamlit openai
```

### 3단계: API 키 설정


```bash
$env:OPENAI_API_KEY = "공유드린키"
```


### 3단계: streamlit 실행

```bash
streamlit run streamlit_app_4.py
```



