---
description: 트랜스크립트 기반 도구 사용 패턴, 파일 활동, 에러, 워크플로우 분석
argument-hint: "[--days N] [--transcript-sub tools|files|errors|efficiency|workflows]"
allowed-tools: Bash(python3:*)
---

# cc-insights 트랜스크립트 분석

`~/.claude/transcripts/` 의 전체 대화 기록을 분석하여 도구 사용 패턴, 파일 활동, 에러 패턴, 워크플로우 패턴 등 인사이트를 제공합니다.

## 실행

### 기본 분석 (최근 7일 종합 요약)

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" transcript --days 7
```

### 기간 지정 분석

`$ARGUMENTS`에 `--days N`이 포함되어 있으면 해당 기간으로 분석:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" transcript --days 30
```

### 서브커맨드

`$ARGUMENTS`에 서브커맨드가 포함되어 있으면 해당 분석만 실행:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" transcript --transcript-sub tools
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" transcript --transcript-sub files
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" transcript --transcript-sub errors
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" transcript --transcript-sub efficiency
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" transcript --transcript-sub workflows
```

## 분석 항목

### 1. 도구 사용 분포 (tools)
- 도구별 사용 횟수, 세션당 평균, 전체 비율
- bash, read, edit, write, glob, grep, task 등 모든 도구

### 2. 파일 활동 (files)
- 가장 많이 읽기/수정/생성된 파일 Top N
- 고유 파일 수 통계

### 3. 에러 패턴 (errors)
- 총 에러 수 및 에러율
- 도구별 에러 분포
- 에러 유형 (exit code, ENOENT, 권한 등)

### 4. 세션 효율성 (efficiency)
- 평균/중앙값 세션 시간
- 도구 밀도 (회/분)
- 세션별 에러율

### 5. 워크플로우 패턴 (workflows)
- 도구 호출 체인 패턴 (2-3gram)
- 예: read → edit → bash (코드 수정 후 실행)

## 관련 명령어

| 명령어 | 설명 |
|--------|------|
| `/cc-insights:analyze` | 프롬프트 패턴 분석 |
| `/cc-insights:summary` | 빠른 현황 요약 |
| `/cc-insights:transcript` | 트랜스크립트 분석 (이 명령) |
