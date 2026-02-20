# cc-insights

Claude Code 사용 패턴을 분석하여 개인화된 인사이트를 제공하는 플러그인입니다.

## 개요

cc-insights는 `~/.claude/history.jsonl` 파일을 분석하여 다음과 같은 인사이트를 제공합니다:

- **프롬프트 품질 분석**: 모호하거나 비효율적인 프롬프트 감지
- **자동화 추천**: 반복 패턴을 감지하여 skill 자동화 제안
- **사용 통계**: 프로젝트별, 시간대별 활동 패턴 분석
- **개인화 팁**: 사용자 스타일에 맞는 맞춤형 개선 제안
- **트렌드 추적**: 주간/월간 사용 패턴 변화 모니터링

## 빠른 시작

```bash
# 초기 설정 (baseline 생성)
/cc-insights:setup

# 빠른 현황 확인
/cc-insights:summary

# 전체 분석
/cc-insights:analyze

# 개인화된 팁
/cc-insights:tips
```

## 주요 기능

| 명령어 | 설명 | 용도 |
|--------|------|------|
| `/cc-insights:setup` | 초기 설정 및 baseline 생성 | 최초 1회 |
| `/cc-insights:summary` | 빠른 현황 요약 | 매일 |
| `/cc-insights:analyze` | 상세 분석 리포트 | 주 1회 |
| `/cc-insights:tips` | 개인화된 개선 팁 | 필요 시 |
| `/cc-insights:trends` | 주간 트렌드 분석 | 주 1회 |
| `/cc-insights:skills` | skill 자동화 추천 | 필요 시 |
| `/cc-insights:projects` | 프로젝트별 분석 | 필요 시 |
| `/cc-insights:stats` | 전체 사용 통계 | 월 1회 |

## 분석 항목

### 1. 프롬프트 품질 점수 (0-10점)

각 프롬프트를 다음 기준으로 평가합니다:

- **길이**: 너무 짧거나 모호한 프롬프트 감지
- **명확성**: 지시대명사(이거, 저거) 과다 사용 감지
- **구체성**: 컨텍스트, 기대 결과, 제약 조건 포함 여부

### 2. 자동화 후보 감지

3회 이상 반복된 패턴을 감지하여 skill로 자동화할 수 있는 항목을 제안합니다:

- `/commit` - 커밋 관련 작업
- `/handoff` - 핸드오프 문서 작성
- `/log-check` - 로그 확인 및 분석

### 3. 비효율 패턴 감지

- `/clear` 과다 사용 → `claude --continue` 권장
- 동시 세션 과다 → `/rename`으로 세션 구분 권장

### 4. 시간/프로젝트 패턴

- 피크 작업 시간대 분석
- 야간 작업 비율 및 권장사항
- 프로젝트별 집중도 분석

## 디렉토리 구조

```
cc-insights/
├── .claude-plugin/     # 플러그인 설정
├── commands/           # skill 정의 파일
│   ├── cc-insights:analyze.md
│   ├── cc-insights:projects.md
│   ├── cc-insights:setup.md
│   ├── cc-insights:skills.md
│   ├── cc-insights:stats.md
│   ├── cc-insights:summary.md
│   ├── cc-insights:tips.md
│   └── cc-insights:trends.md
├── scripts/            # Python 분석 스크립트
│   ├── analyzer.py     # 메인 분석기
│   ├── patterns.py     # 패턴 감지 로직
│   └── skill_generator.py
├── data/               # baseline 및 캐시 데이터
├── reports/            # 저장된 리포트
└── docs/               # 문서
```

## 요구사항

- Python 3.8+
- Claude Code CLI
- `~/.claude/history.jsonl` 파일 (Claude Code 사용 시 자동 생성)

## 관련 문서

- [명령어 상세](./COMMANDS.md)
- [기술 아키텍처](./ARCHITECTURE.md)
- [향후 개발 계획](./ROADMAP.md)
