# cc-insights 기술 아키텍처

## 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                      Claude Code CLI                             │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   cc-insights Plugin                     │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │    │
│  │  │   commands/   │  │   scripts/    │  │    data/    │  │    │
│  │  │  (Skill MD)   │→ │  (Python)     │→ │  (JSON)     │  │    │
│  │  └───────────────┘  └───────────────┘  └─────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↑                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ~/.claude/history.jsonl                     │    │
│  │              (Claude Code 사용 기록)                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 데이터 흐름

```
history.jsonl ──→ analyzer.py ──→ 분석 결과 출력
                      │
                      ├──→ patterns.py (패턴 감지)
                      │
                      └──→ baseline.json (비교 데이터)
```

## 핵심 컴포넌트

### 1. 데이터 소스: history.jsonl

Claude Code가 자동으로 생성하는 사용 기록 파일입니다.

**위치:** `~/.claude/history.jsonl`

**데이터 구조:**
```json
{
  "display": "프롬프트 내용",
  "pastedContents": {},
  "timestamp": 1705123456789,
  "project": "/Users/tony/my-project",
  "sessionId": "uuid-string"
}
```

**필드 설명:**
| 필드 | 타입 | 설명 |
|------|------|------|
| display | string | 사용자가 입력한 프롬프트 |
| timestamp | number | Unix timestamp (밀리초) |
| project | string | 프로젝트 경로 |
| sessionId | string | 세션 고유 ID |

### 2. 메인 분석기: analyzer.py

**클래스:** `CCInsightsAnalyzer`

**주요 메서드:**

| 메서드 | 설명 |
|--------|------|
| `load_history()` | 기간 내 히스토리 로드 |
| `load_all_history()` | 전체 히스토리 로드 |
| `analyze_vague_prompts()` | 모호한 프롬프트 분석 |
| `analyze_quality_scores()` | 품질 점수 계산 |
| `analyze_automation_candidates()` | 자동화 후보 감지 |
| `analyze_inefficiencies()` | 비효율 패턴 감지 |
| `analyze_time_patterns()` | 시간대 패턴 분석 |
| `analyze_project_patterns()` | 프로젝트별 패턴 분석 |
| `analyze_project_detail()` | 특정 프로젝트 상세 분석 |
| `analyze_overall_stats()` | 전체 사용 통계 |
| `analyze_trends()` | 주간 트렌드 분석 |
| `generate_user_profile()` | 사용자 프로필 생성 |
| `analyze_strengths_weaknesses()` | 강점/약점 분석 |
| `generate_personalized_tips()` | 개인화 팁 생성 |

**데이터 클래스:**
```python
@dataclass
class PromptEntry:
    display: str      # 프롬프트 내용
    timestamp: datetime
    project: str      # 프로젝트 경로
    session_id: str   # 세션 ID
```

### 3. 패턴 감지: patterns.py

**주요 함수:**

| 함수 | 설명 |
|------|------|
| `detect_vague_patterns()` | 모호한 표현 감지 |
| `detect_automation_candidates()` | 반복 패턴 감지 |
| `detect_inefficiency_patterns()` | 비효율 패턴 감지 |
| `calculate_prompt_quality_score()` | 품질 점수 계산 |
| `get_improvement_suggestion()` | 개선 제안 생성 |
| `analyze_session_patterns()` | 세션 패턴 분석 |

**품질 점수 계산 기준:**

```python
def calculate_prompt_quality_score(prompt: str) -> Tuple[int, List[str]]:
    score = 10
    issues = []

    # 길이 검사 (너무 짧으면 감점)
    if len(prompt) < 10:
        score -= 3
        issues.append("너무 짧음")

    # 모호한 표현 검사
    vague_patterns = ["이거", "저거", "그거", "확인", "해줘"]
    for pattern in vague_patterns:
        if pattern in prompt:
            score -= 1
            issues.append(f"모호한 표현: {pattern}")

    # 구체성 가점
    if any(word in prompt for word in ["파일", "함수", "에러", "테스트"]):
        score += 1

    return max(0, min(10, score)), issues
```

**자동화 후보 감지 로직:**

```python
def detect_automation_candidates(prompts: List[str]) -> List[dict]:
    # 유사한 프롬프트 그룹화
    # 3회 이상 반복된 패턴 추출
    # 신뢰도 점수 계산
    # skill 이름 및 설명 생성
```

### 4. Skill 정의: commands/*.md

**구조:**
```markdown
---
description: 명령어 설명
argument-hint: "[옵션]"
allowed-tools: Bash(python3:*)
---

# 명령어 이름

사용 방법 및 설명...

## 실행

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" <command>
```
```

### 5. 데이터 저장: data/

**baseline.json:**
- 초기 설정 시 생성
- 향후 비교를 위한 기준 데이터
- 자동화 후보, 프로젝트 빈도, 시간 패턴 저장

```json
{
  "generated_at": "2025-01-14T10:00:00",
  "total_prompts": 5000,
  "date_range": {
    "start": "2024-11-15",
    "end": "2025-01-14"
  },
  "top_prompts": [["커밋", 150], ["확인", 80]],
  "top_projects": [["project-a", 2000], ["project-b", 1500]],
  "automation_candidates": [...],
  "hourly_avg": {...},
  "avg_prompts_per_day": 83.3
}
```

## 분석 알고리즘

### 프롬프트 품질 점수

```
기본 점수: 10점

감점 요소:
- 10자 미만: -3점
- 모호한 표현 (이거, 저거, 확인해줘 등): -1점/개
- 지시대명사 과다: -1점

가점 요소:
- 구체적 키워드 (파일명, 함수명, 에러 메시지): +1점
- 명확한 기대 결과 명시: +1점
- 제약 조건 포함: +1점

최종 점수: 0~10점
```

### 자동화 후보 신뢰도

```
신뢰도 = (반복 횟수 / 총 프롬프트) × 패턴 일관성 × 100

패턴 일관성:
- 동일 문구: 1.0
- 유사 문구 (edit distance < 3): 0.8
- 의미적 유사 (키워드 매칭): 0.6
```

### 사용자 프로필 분류

**작업 스타일:**
| 분류 | 조건 |
|------|------|
| 얼리버드 | 오전(6-12시) 비율 > 40% |
| 나이트 아울 | 저녁(18-24시) 비율 > 35% |
| 심야형 | 심야(0-6시) 비율 > 20% |
| 균형 잡힌 스타일 | 그 외 |

**세션 스타일:**
| 분류 | 조건 |
|------|------|
| 딥다이브 | 세션당 평균 > 50 프롬프트 |
| 스프린터 | 세션당 평균 < 15 프롬프트 |
| 밸런서 | 그 외 |

## 성능 고려사항

### 메모리 사용

- 전체 히스토리 로드 시 대용량 파일 처리
- 스트리밍 방식으로 line by line 읽기
- 필요한 기간만 필터링하여 로드

### 처리 시간

| 프롬프트 수 | 예상 처리 시간 |
|-------------|----------------|
| 1,000 | < 1초 |
| 10,000 | 2-3초 |
| 50,000 | 10-15초 |

### 최적화 포인트

1. **캐싱**: baseline 데이터 저장으로 재분석 방지
2. **증분 분석**: 마지막 분석 이후 데이터만 처리 (향후 구현)
3. **병렬 처리**: 독립적인 분석 작업 병렬화 (향후 구현)
