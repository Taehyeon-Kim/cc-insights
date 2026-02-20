---
description: Skill 자동화 추천 및 생성
argument-hint: "[list|generate <name>]"
allowed-tools: Bash(python3:*), Read, Write
---

# cc-insights Skill 관리

반복되는 작업 패턴을 분석하여 자동화할 skill을 추천합니다.

## 사용법

### Skill 목록 확인

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/analyzer.py" skills --days 30
```

### Skill 코드 생성

`$ARGUMENTS`에 `generate <name>`이 포함되어 있으면:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/skill_generator.py" <name>
```

## 추천 기준

- **3회 이상** 반복된 패턴만 추천
- 신뢰도 = 반복 횟수에 비례 (최대 95%)

## 감지 가능한 패턴

| 패턴 | Skill 이름 | 설명 |
|------|-----------|------|
| 로그 확인 | `/log-check` | 로그 확인 및 에러 분석 |
| 핸드오프 작성 | `/handoff` | 핸드오프 문서 자동 작성 |
| 배포 진행 | `/deploy` | 배포 및 상태 확인 |
| 상태 확인 | `/status-check` | 프로젝트 상태 종합 확인 |
| 커밋 | `/commit` | 변경사항 커밋 |
| 테스트 실행 | `/run-tests` | 테스트 실행 및 결과 분석 |

## 출력 예시

```
╔══════════════════════════════════════════════════════════════╗
║  🤖 Skill 자동화 추천                                         ║
╚══════════════════════════════════════════════════════════════╝

  1. /commit - 변경사항 커밋
     반복 횟수: 23회
     신뢰도: [█████████░] 90%
     샘플:
       • "커밋해줘"
       • "변경사항 커밋"

  2. /log-check - 로그 확인 및 에러 분석
     반복 횟수: 15회
     신뢰도: [████████░░] 80%
     샘플:
       • "로그 확인"
       • "pm2 로그 체크"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  skill 생성: /cc-insights:skills generate <skill_name>
```

## Skill 생성 후

생성된 skill 파일은 `${CLAUDE_PLUGIN_ROOT}/generated_skills/` 디렉토리에 저장됩니다.

이 파일을 프로젝트의 `.claude/commands/` 디렉토리로 복사하여 사용하세요.
