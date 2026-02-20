---
description: Claude Code 사용 패턴 분석 및 개인화된 피드백 제공
argument-hint: "[--days N] [report]"
allowed-tools: Bash(python3:*), Read, Write
---

# cc-insights 분석

Claude Code 사용 패턴을 분석하여 개인화된 인사이트를 제공합니다.

## 사용법

### 기본 분석 (최근 7일)

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" --days 7 --format summary
```

### 기간 지정 분석

`$ARGUMENTS`에 `--days N`이 포함되어 있으면 해당 기간으로 분석:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" --days 30 --format summary
```

### 상세 리포트 저장

`$ARGUMENTS`에 `report`가 포함되어 있으면:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" --days 7 --format markdown --output "${CLAUDE_PLUGIN_ROOT}/reports/$(date +%Y-%m-%d).md"
```

## 분석 항목

### 1. 프롬프트 품질 점수 (0-10)

각 프롬프트의 품질을 평가합니다:
- **10점**: 구체적이고 명확한 프롬프트
- **5-7점**: 개선 여지가 있는 프롬프트
- **0-4점**: 모호하거나 불명확한 프롬프트

### 2. 반복 패턴 → Skill 제안

3회 이상 반복된 패턴을 감지하여 자동화 가능한 skill을 제안합니다.

예시:
- "로그 확인" 12회 → `/log-check` skill 제안
- "handoff 작성" 8회 → `/handoff` skill 제안

### 3. 비효율 패턴 감지

- `/clear` 과다 사용 → `claude --continue` 권장
- 동시 세션 관리 → `/rename` 권장

### 4. 시간/프로젝트 패턴

- 피크 시간대 분석
- 야간 작업 비율
- 프로젝트별 집중도

## 출력 예시

```
╔══════════════════════════════════════════════════════════════╗
║  cc-insights 분석 결과 (최근 7일)
╠══════════════════════════════════════════════════════════════╣
║  총 프롬프트: 312개
║  평균 품질 점수: 8.2/10
║  모호한 프롬프트: 45개 (14.4%)
║  자동화 후보: 4개 패턴 감지
╚══════════════════════════════════════════════════════════════╝

[Skill 제안]
  → /commit: 23회 반복 (변경사항 커밋)
  → /log-check: 12회 반복 (로그 확인 및 에러 분석)

[개선 필요 프롬프트 예시]
  • "로그 확인" → "pm2 logs에서 최근 ERROR 로그 확인하고 원인 분석"
  • "확인해줘" → 무엇을 확인할지 명시
```

## baseline 비교

`/cc-insights:setup`으로 생성한 baseline이 있으면 비교 정보를 제공합니다:
- "이번 주 '로그 확인' 12회 (평균 대비 +40%)"
- "모호한 프롬프트 비율 14% → 10% 개선"

## 관련 명령어

| 명령어 | 설명 |
|--------|------|
| `/cc-insights:summary` | 빠른 현황 확인 (매일) |
| `/cc-insights:tips` | 개인화된 팁 |
| `/cc-insights:trends` | 주간 트렌드 |
| `/cc-insights:skills` | skill 목록 및 생성 |

## 참고

- 분석은 `~/.claude/history.jsonl` 파일을 사용합니다.
- 한국어와 영어 프롬프트 모두 분석됩니다.
