# 용어집

보고서와 코드에서 쓰는 이름의 정의. 새 이름을 만들 때는 여기에 먼저 등록.

## 데이터

- passage(지문): SQuAD 문서 한 토막. `title` + `context` 로 구성
- passage key: `title#sha1(context)[:8]`. 같은 title 아래 지문이 여러 개라 짝짓기는 이 키로 함
- gold: 데이터셋에 사람이 붙여 둔 질문-정답 쌍. 지문당 평균 4.6개, 답은 지문에서 그대로 따온 짧은 구
- QA_0: 외부 루프 보상에 쓰는 gold 질문 집합을 가리키는 제안서 표기. gold 와 같은 것
- QA_gen: 모델이 지문만 보고 스스로 쓴 질문-정답 집합. gold 를 보여주지 않음
- dev / val: dev 30지문은 파일럿 점검용, val 200지문은 최종 비교용. 서로 겹치지 않음

## 모델 동작

- self-edit(SE): 지문을 보고 모델이 쓴 학습용 텍스트(함의, 재진술). TTT 의 학습 데이터가 됨
- TTT(test-time training): 지문 하나마다 self-edit 으로 LoRA 를 잠깐 학습시켜 그 지문에 적응시키는 것. 평가가 끝나면 버림
- LoRA: 원 가중치를 두고 저랭크 행렬만 학습. TTT 는 rank 32, SFT 는 rank 64
- SFT: 채택된 (지문, self-edit) 쌍으로 모델 자체를 학습시키는 단계
- closed-book: 지문도 적응도 없이 질문에 답하게 한 상태. 모델이 이미 알던 양을 재는 바닥값
- passage-only: self-edit 대신 지문 원문으로 TTT 한 상태

## 루프

- 내부 루프(SE-RL): SEAL 의 ReST-EM. 지문마다 self-edit 을 K=5개 뽑아 각각 TTT 한 뒤 보상 질문으로 채점하고, 최고점 self-edit 만 모아 SFT. 2라운드, 라운드당 50지문
- 외부 루프: QA_gen 을 쓰는 질문 생성기를 학습시키는 단계. 지문마다 질문세트 후보 M=6개를 뽑고, 각 후보가 고른 self-edit 을 gold 로 채점해 보상 R 을 줌. R=1 인 후보만 SFT. 3회 반복, 회당 40지문
- 보상 R: 후보가 고른 self-edit 의 gold 정확도가 K개 self-edit 의 평균보다 높으면 1, 아니면 0
- n0 / nN: 외부 루프 학습 전(n=0) 질문 생성기와 N회 학습 후(n=N) 질문 생성기

## 조건

이름 규칙: `qagen_*` 는 QA_gen 을 내부 루프 보상으로 쓴 조건, `n0` 과 `nN` 은 외부 루프 학습 전과 후의 질문
생성기, 접미사 `_N` 은 외부 루프로 학습한 모델에서 출발한 조건, `_seed*` 는 같은 조건의 반복 시드.
시드 이름이 없는 `qagen_nN` 은 시드들의 지문별 평균.

| 이름 | 정의 |
| --- | --- |
| closed_book | 지문 없이 답 |
| passage_only | 지문 원문으로 TTT |
| base_se | RL 전 모델의 self-edit 으로 TTT. 주 비교 기준 |
| gpt_se | GPT 가 쓴 self-edit 으로 TTT |
| qagen_nN_seed0/1/2 | 제안 방법. 외부 루프로 학습한 질문을 내부 루프 보상으로 씀 |
| qagen_n0 | 같은 내부 루프, 질문은 학습 전 생성기가 씀 |
| qagen_randR | 같은 내부 루프, 질문은 보상을 무작위로 준 외부 루프가 학습한 생성기가 씀 |
| qa0_sup | 내부 루프 보상까지 gold 질문으로 준 상한선 |
| inner_random | self-edit 을 점수 없이 무작위로 골라 SFT. SFT 쌍 수는 qagen_n0 에 맞춤 |
| outer_only | 외부 루프만 돌린 모델, 내부 루프 없음 |
| closed_book_N / passage_only_N | 외부 루프로 학습한 모델의 closed-book, passage-only |

## 지표

- 정확도: gold 질문 정답률. 채점은 LLM judge 가 yes/no 로 판정
- pp(percentage point): 정확도 차이의 단위. 45.3% 와 50.3% 의 차이가 +5.0pp
- 95% CI: paired bootstrap 신뢰구간. 같은 지문끼리 뺀 차이를 10,000번 재표집
- title-cluster CI: 같은 문서의 지문은 독립이 아니므로 문서 단위로 재표집한 구간
- contains-match: 모델 답 안에 gold 답 문자열이 들어 있으면 정답. judge 없이 재는 보조 지표
- SQuAD F1: 모델 답과 gold 답의 토큰 겹침 F1. 정밀도와 재현율의 조화평균이라 부분 일치도 점수를 줌
- parse rate: 생성한 질문 텍스트에서 질문-답 쌍을 규칙대로 뽑아낸 비율
- gold coverage: gold 답 중 생성 답이 F1 0.5 이상으로 맞춘 비율
- answer in passage: 생성한 답이 지문에 그대로 있는 비율
- duplicate rate: 한 질문세트 안에서 중복된 질문의 비율
- margin: 후보들의 gold 정확도 최고값과 평균의 차. 보상이 얼마나 뚜렷한지를 봄
- top tie: 최고점을 공유한 후보가 둘 이상인 지문의 비율. 동점은 평균으로 처리
- reward rate: R=1 을 받은 후보의 비율
- Spearman: 같은 지문 안에서 self-edit 들을 QA_gen 점수와 gold 점수로 매겼을 때의 순위 상관
- Cohen's kappa: 두 채점 기준이 고른 최고 self-edit 이 얼마나 일치하는지. 우연 일치를 뺀 값
- self-flip: 같은 답을 같은 judge 에 다시 물었을 때 판정이 뒤집힌 비율

## 가설

- H1: qagen_nN > base_se. 주가설
- H2: qagen_n0 > base_se
- H3: qagen_nN > qagen_n0
- H3b: qagen_nN > qagen_randR
- H4: qagen_nN 이 qa0_sup 대비 -3pp 이내(비열등성)
- RQ1: QA_gen 점수가 gold 점수를 대신할 수 있는지. 판정 기준은 Spearman 신뢰구간 하한이 0보다 큰지
- 순서 고정. 앞 가설이 실패하면 뒤는 미검정으로 두고 탐색적 결과로만 보고
