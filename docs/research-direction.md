# 연구 방향: 문서 핵심 주장 보존형 이미지 게이트

## 한 줄 요약

Wikipedia 기반 멀티모달 RAG에서 모든 이미지를 처리하지 않는다. 문서의 핵심 주장을 대표하거나 텍스트에 없는 지식을 제공하는 이미지만 선별해 KG와 RAG에 투입한다.

## 문제 정의

문서 속 이미지는 모두 같은 가치가 없다.

- 핵심 대표 이미지: 문서 주제 또는 핵심 주장을 잘 보여 준다.
- 지식 기여 이미지: 차트, 지도, 도식처럼 텍스트에 없는 수치·관계·구조를 제공한다.
- 장식 이미지: 문서 이해와 답변에 거의 기여하지 않는다.

기존 멀티모달 RAG는 관련성 또는 검색 순위에 따라 이미지를 통합하는 경우가 많다. 이 연구는 **문서 수집·인덱싱 단계에서 이미지를 넣을 가치가 있는지** 판단한다.

## 연구 목표

로컬 VLM 기반 이미지 게이트가 불필요한 이미지 처리 비용과 KG 노이즈를 줄이면서도, 문서 기반 질의응답 성능과 핵심 지식 커버리지를 유지하는지 검증한다.

## 핵심 연구 질문

1. 로컬 VLM은 이미지의 핵심 주장 대표성과 추가 지식 기여도를 사람과 비슷하게 판정하는가?
2. 선별된 이미지만 KG/RAG에 넣어도 엔티티·관계 커버리지와 질의응답 품질이 유지되는가?
3. 전체 이미지 처리 대비 VLM 호출 수, 처리 시간, 인덱스 크기, 저장량을 얼마나 줄이는가?

## 연구 가설

- H1: 문서 문맥을 함께 제공한 VLM 게이트는 이미지 단독 또는 CLIP 유사도보다 대표 이미지 판정 F1이 높다.
- H2: 게이트 적용 시스템은 전체 이미지 투입 시스템과 유사한 QA 품질을 유지한다.
- H3: 게이트 적용 시스템은 처리 이미지 수와 멀티모달 인덱싱 비용을 유의미하게 줄인다.

## 범위

### 이번 여름 인턴 발표

- Wikipedia 문서 10개와 연결 이미지 수집
- Ollama VLM 기반 `keep/drop` 게이트 프로토타입
- BLIP 캡션, OCR 텍스트, 키워드 추출
- 20~30개 이미지의 단독 pilot 라벨
- 통과 이미지와 핵심 주장·엔티티를 연결한 evidence KG
- 대표 성공·실패 사례, 제거율, 처리 시간을 제시
- TechRAG, MMGraphRAG, KnoBuilder와 연구 위치 비교

### KCI 논문

- 사람 라벨 데이터셋 구축
- CLIP, BLIP-caption similarity, heuristic, VLM 게이트 비교
- KG 커버리지·RAG 답변 정확도·비용의 정량 실험
- 한국어 Wikipedia 또는 명확한 한국어 도메인으로 재현 가능한 벤치마크 제시

## 시스템 설계

```text
Wikipedia API
  └ 문서, 요약, 섹션, 이미지 URL, 캡션, 라이선스 수집
      └ 이미지 다운로드
          └ OCR + BLIP 캡션 + 키워드 추출
              └ Ollama VLM 이미지 게이트
                  └ 통과: image index + KG triple + RAG
                  └ 탈락: 원본 메타데이터만 보존, KG/RAG 제외
```

### 게이트 입력

- 이미지
- 문서 제목
- lead summary
- 이미지 캡션
- 이미지가 위치한 섹션 제목과 인접 문단

이미지 단독 판정은 금지한다. 문서 문맥 없이는 “대표성”을 판정할 수 없다.

### 게이트 출력

```json
{
  "representative": 0,
  "knowledge_contribution": 0,
  "image_type": "photo|chart|map|diagram|decorative",
  "supported_claim_ids": [],
  "decision": "keep|drop",
  "confidence": 0.0
}
```

`representative`와 `knowledge_contribution`은 분리한다. 인물 사진은 대표성이 높지만 새 지식은 적을 수 있고, 복잡한 차트는 대표성은 낮아도 중요한 지식을 제공할 수 있다.

## 평가 설계

### Gold set: 직접 구축

이 연구의 핵심 라벨인 `문서 핵심 주장 대표성`과 `추가 지식 기여도`를 동시에 제공하는 공개 데이터셋은 없다. 따라서 영어 Wikipedia 기반 gold set을 직접 구축한다. WIT와 WISMIR3는 대규모 Wikipedia 이미지-텍스트 쌍을 제공하지만, 이미지가 문서 핵심 주장을 대표하는지에 대한 라벨은 제공하지 않는다. M2RAG와 Visual-RAG는 최종 RAG 성능을 보조 평가하는 데 사용한다.

#### 규모

- Pilot: 문서 30개, 이미지 120~150개. 라벨 기준과 프롬프트를 고정한다.
- 본 실험: 문서 150~200개, 이미지 600~1,000개.
- 분할: 이미지 단위가 아닌 **문서 단위**로 train/dev/test를 나눈다. 한 문서의 이미지는 한 split에만 둔다.

처음에는 학습용 데이터가 아니라 고정 평가셋으로 사용한다. 모델 fine-tuning은 gold set과 별도 데이터가 충분할 때만 고려한다.

#### 라벨 스키마

| 항목 | 값 | 판정 기준 |
| --- | --- | --- |
| `image_type` | photo, portrait, chart, map, diagram, logo, decorative, other | 시각적 형식 |
| `representative` | 0/1 | 이미지와 캡션이 문서 lead의 핵심 주제 또는 핵심 주장 하나를 직접 보여 주는가 |
| `knowledge_contribution` | 0/1 | lead·캡션만으로 얻을 수 없는 수치, 관계, 공간 구조, 시간 변화, 시각적 구분을 제공하는가 |
| `keep_for_rag` | 0/1 | `representative` 또는 `knowledge_contribution`이 1인가 |
| `keep_for_kg` | 0/1 | 엔티티·관계·수치·구조를 추출해 KG 노드 또는 edge로 만들 근거가 있는가 |
| `supported_claim` | text / section id | 어떤 lead claim 또는 문서 섹션을 뒷받침하는가 |

`keep_for_rag`와 `keep_for_kg`를 분리한다. 대표 인물 사진은 RAG의 시각적 근거가 될 수 있지만, 새 관계를 만들지 못하면 KG에는 넣지 않는다.

#### 라벨 절차

1. 문서 제목, lead summary, 섹션 제목, 인접 문단, 이미지, 캡션을 annotation card로 제공한다.
2. Pilot에서는 한 명이 라벨링하되, 판정 기준과 사유를 함께 기록한다.
3. 본 논문 실험에서는 두 명이 독립 라벨링하고, 불일치는 제3 검토 또는 합의 회의로 해결한다.
4. 본 논문 실험에서 Cohen's kappa를 보고한다. `representative`, `knowledge_contribution`, `keep_for_kg` 각각 계산한다.
5. 애매한 사례와 최종 판정 사유를 별도 adjudication log로 보존한다.

모델 출력은 gold label을 본 뒤 수정하지 않는다. 프롬프트와 threshold는 dev set에서만 고정하고 test set은 마지막 한 번만 평가한다.

### 비교군

- No-image: 이미지 미사용
- All-image: 모든 이미지 투입
- Heuristic: lead image, 캡션 길이, 이미지 크기 기반
- CLIP: 이미지-문서 요약 유사도 기반
- BLIP-caption: 생성 캡션-문서 요약 유사도 기반
- VLM gate: 제안 방법
- Oracle: 사람 라벨 선택

### 지표

- 게이트 품질: Precision, Recall, F1, Cohen's kappa
- KG 품질: 엔티티·관계 커버리지, 노이즈 triple 비율
- RAG 품질: QA 정확도, 근거 인용 정확도
- 비용: VLM 호출 수, 이미지당 처리 시간, 인덱스 크기, 저장량

## 관련 연구와 차별점

- TechRAG는 전문 기술 문헌에서 hybrid multimodal retrieval과 agentic pipeline을 다룬다. 본 연구는 그보다 앞단인 **인덱싱 전 이미지 입장 심사**에 둔다.
- MMGraphRAG는 시각 정보를 정제해 멀티모달 KG를 구축한다. 본 연구는 KG에 넣을 이미지를 먼저 줄이는 비용·노이즈 문제에 초점을 둔다.
- KnoBuilder는 비정형 텍스트에서 agent가 KG를 구축한다. 본 연구의 1차 목표는 agentic KG 전체 구현이 아니라, KG 입력 이미지 선별이다.
- 최근 visual evidence selection 연구는 질의 조건부 이미지 유용성을 다룬다. 본 연구는 **질의 이전의 문서 조건부 대표성·지식 기여도**를 별도 과제로 정의한다. 단순 “이미지 게이트”만으로는 차별성이 약하므로, 사람 라벨과 KG/RAG/비용의 연결 평가가 필요하다.

## 내일까지 구현할 MVP

목표는 "이미지 게이트와 provenance-preserving evidence KG가 실제로 입력부터 결과까지 동작한다"는 것을 재현 가능하게 보이는 것이다. gold set, Agent, 완성형 RAG는 내일 범위에서 제외한다.

1. **수집**: 영어 Wikipedia 문서 10개를 제목 목록으로 고정하고, API에서 lead summary·섹션·이미지 URL·캡션·라이선스 메타데이터를 받는다.
2. **저장**: 원문 메타데이터는 `data/raw/`, 이미지 파일은 `data/images/`, 처리 결과는 `data/processed/`에 둔다. 결과 JSONL에는 문서 id, 이미지 id, URL, caption, section, license를 보존한다.
3. **이미지 설명**: BLIP으로 caption을 생성하고 OCR 텍스트와 키워드를 추출한다. 실패도 `error` 필드에 기록한다.
4. **게이트**: Ollama VLM에 이미지와 문서 문맥을 입력하고, 고정 JSON의 `representative`, `knowledge_contribution`, `image_type`, `decision`, `confidence`를 받는다.
5. **검증**: JSON schema 검증, 재시도, 원본 응답 보관을 구현한다. 모델 출력이 JSON이 아니면 결과로 쓰지 않는다.
6. **KG**: 통과 이미지만 `Document-Section-Claim-Image-Entity` evidence graph에 연결한다. 모든 edge에 출처·근거·confidence를 보존한다.
7. **시연**: 문서별로 원본 이미지 수, keep/drop 수, 선택 사유, 처리 시간, 실패 수와 KG edge 수를 Markdown 표 또는 간단한 HTML로 생성한다.
8. **수동 점검**: 20~30개 이미지만 본인이 라벨링하고, VLM과 다른 사례를 기록한다. 이는 pilot analysis이지 최종 gold set이 아니다.

### 내일 완료 기준

- 한 명령으로 10개 문서 수집부터 결과 표 생성까지 실행된다.
- 이미지마다 문서 문맥, VLM 판정, 원본 응답, 처리 시간, 오류 상태가 남는다.
- KG edge마다 문서와 이미지의 출처 및 근거가 남는다.
- 최소 3개 `keep`, 3개 `drop`, 1개 실패 사례를 발표에서 보여 준다.
- 결과 수치: 전체 이미지 수, keep 비율, 평균 처리 시간, 실패율.

## 발표 메시지

"멀티모달 RAG의 문제는 이미지를 어떻게 많이 넣느냐가 아니라, 어떤 이미지를 지식으로 승격할 가치가 있느냐이다. 우리는 문서 핵심 주장 보존형 이미지 게이트로 이를 검증한다."

## 읽을 논문

- [TechRAG: Evidence-Gated Multimodal Agentic RAG for Technical Literature Reasoning](https://www.alphaxiv.org/overview/2606.01613v2)
- [MMGraphRAG: Bridging Vision and Language with Interpretable Multimodal Knowledge Graphs](https://arxiv.org/abs/2507.20804)
- [KnoBuilder: An LLM-Agent for Autonomous and Personalized Knowledge Graph Construction from Unstructured Text](https://openreview.net/forum?id=teewCPCv2m)
- [Utility-Oriented Visual Evidence Selection for Multimodal RAG](https://aclanthology.org/2026.acl-long.1620/)
- [Modality Relevance is not Modality Utility](https://arxiv.org/abs/2607.05438)
- [WIT: Wikipedia-based Image Text Dataset](https://research.google/pubs/wit-wikipedia-based-image-text-dataset-for-multimodal-multilingual-machine-learning/)
- [WISMIR3: A Multi-Modal Dataset to Challenge Text-Image Retrieval Approaches](https://aclanthology.org/2024.alvr-1.1/)
- [M2RAG: Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts](https://arxiv.org/abs/2502.17297)
- [Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation](https://arxiv.org/abs/2502.16636)
