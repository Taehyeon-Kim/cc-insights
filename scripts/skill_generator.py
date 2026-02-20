#!/usr/bin/env python3
"""
cc-insights Skill 자동 생성기
반복 패턴 기반으로 skill 코드를 생성합니다.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

GENERATED_SKILLS_PATH = Path(__file__).parent.parent / "generated_skills"

# Skill 템플릿 (한국어)
SKILL_TEMPLATES = {
    "log-check": """---
description: 로그 확인 및 에러 분석
allowed-tools: Bash(pm2:*), Bash(tail:*), Bash(grep:*), Read
---

# 로그 확인

프로젝트 로그를 확인하고 에러를 분석합니다.

## 실행 단계

1. pm2 또는 docker 로그 확인
2. ERROR 레벨 필터링
3. 최근 발생 시간순 정렬
4. 원인 분석 및 해결책 제안

## 기본 명령

```bash
pm2 logs --lines 100 | grep -i error
```

$ARGUMENTS가 있으면 해당 프로젝트/서비스 로그를 확인합니다.
""",

    "handoff": """---
description: 핸드오프 문서 자동 작성
allowed-tools: Read, Write, Bash(git:*)
---

# 핸드오프 문서 작성

현재 작업 상태를 정리하여 핸드오프 문서를 작성합니다.

## 포함 내용

1. **오늘 완료한 작업**
   - git log에서 최근 커밋 요약

2. **진행 중인 작업**
   - 현재 브랜치 상태
   - 미완료 TODO 항목

3. **다음 단계**
   - 다음 세션에서 해야 할 작업

4. **주의사항**
   - 알려진 이슈
   - 환경 설정 관련

## 저장 위치

프로젝트 루트의 `HANDOFF.md` 또는 지정된 위치에 저장합니다.
""",

    "commit": """---
description: 변경사항 커밋
allowed-tools: Bash(git:*)
---

# 변경사항 커밋

현재 변경사항을 검토하고 커밋합니다.

## 실행 단계

1. `git status`로 변경사항 확인
2. `git diff`로 변경 내용 검토
3. conventional commits 형식으로 커밋 메시지 생성
4. 커밋 실행

## 커밋 메시지 형식

```
<type>(<scope>): <description>

[optional body]

Co-Authored-By: Claude <noreply@anthropic.com>
```

$ARGUMENTS가 있으면 커밋 메시지 힌트로 사용합니다.
""",

    "deploy": """---
description: 배포 및 상태 확인
allowed-tools: Bash(pm2:*), Bash(docker:*), Bash(git:*)
---

# 배포

프로젝트를 배포하고 상태를 확인합니다.

## 실행 단계

1. 현재 브랜치 확인
2. 테스트 실행 (있는 경우)
3. 배포 스크립트 실행
4. 헬스체크 확인
5. 로그 모니터링

## 배포 대상

$ARGUMENTS로 환경을 지정합니다:
- `staging`: 스테이징 환경
- `production`: 프로덕션 환경 (주의!)

기본값은 staging입니다.
""",

    "status-check": """---
description: 프로젝트 상태 종합 확인
allowed-tools: Bash(git:*), Bash(pm2:*), Bash(docker:*), Read
---

# 상태 확인

프로젝트의 전반적인 상태를 확인합니다.

## 확인 항목

1. **Git 상태**
   - 현재 브랜치
   - 커밋되지 않은 변경사항
   - 원격과 동기화 상태

2. **실행 상태**
   - PM2 프로세스 상태
   - Docker 컨테이너 상태

3. **최근 로그**
   - 에러 로그 유무
   - 경고 메시지

4. **리소스**
   - 디스크 사용량 (선택)
""",

    "run-tests": """---
description: 테스트 실행 및 결과 분석
allowed-tools: Bash(npm:*), Bash(pytest:*), Bash(cargo:*)
---

# 테스트 실행

프로젝트 테스트를 실행하고 결과를 분석합니다.

## 실행 단계

1. 테스트 프레임워크 감지 (package.json, pytest.ini 등)
2. 테스트 실행
3. 결과 요약
4. 실패 시 원인 분석

## 지원 프레임워크

- npm test / jest / vitest
- pytest
- cargo test
- go test

$ARGUMENTS로 특정 테스트 파일이나 패턴을 지정할 수 있습니다.
""",

    "build": """---
description: 프로젝트 빌드
allowed-tools: Bash(npm:*), Bash(cargo:*), Bash(make:*)
---

# 프로젝트 빌드

프로젝트를 빌드합니다.

## 실행 단계

1. 빌드 시스템 감지
2. 의존성 확인
3. 빌드 실행
4. 빌드 결과 확인

## 지원 빌드 시스템

- npm run build
- cargo build
- make build

$ARGUMENTS로 빌드 옵션을 지정할 수 있습니다.
""",

    "restart-server": """---
description: 서버 재시작
allowed-tools: Bash(pm2:*), Bash(docker:*), Bash(systemctl:*)
---

# 서버 재시작

서버를 안전하게 재시작합니다.

## 실행 단계

1. 현재 상태 확인
2. 그레이스풀 종료
3. 재시작
4. 상태 확인
5. 로그 모니터링

$ARGUMENTS로 특정 서비스를 지정할 수 있습니다.
""",
}


def generate_skill(skill_name: str, output_dir: Path = None) -> str:
    """Skill 파일 생성"""
    if skill_name not in SKILL_TEMPLATES:
        available = ", ".join(SKILL_TEMPLATES.keys())
        return f"[오류] 알 수 없는 skill: {skill_name}\n사용 가능: {available}"

    template = SKILL_TEMPLATES[skill_name]

    # 생성 정보 추가
    template += f"""
---
*cc-insights에서 자동 생성됨 ({datetime.now().strftime('%Y-%m-%d %H:%M')})*
"""

    if output_dir:
        # 출력 경로를 generated_skills/ 내부로 제한
        safe_dir = GENERATED_SKILLS_PATH if output_dir is None else output_dir
        resolved = Path(safe_dir).resolve()
        allowed = GENERATED_SKILLS_PATH.resolve()
        if not str(resolved).startswith(str(allowed)):
            return f"[보안] 출력 경로는 generated_skills/ 디렉토리 내에만 허용됩니다: {allowed}"

        output_path = resolved / f"{skill_name}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, template.encode("utf-8"))
        finally:
            os.close(fd)
        return f"[cc-insights] Skill 생성 완료: {output_path}"

    return template


def list_available_skills() -> str:
    """사용 가능한 skill 목록"""
    result = "[cc-insights] 사용 가능한 Skill 템플릿:\n\n"
    for name, template in SKILL_TEMPLATES.items():
        # description 추출
        lines = template.split('\n')
        desc = ""
        for line in lines:
            if line.startswith('description:'):
                desc = line.replace('description:', '').strip()
                break
        result += f"  /{name}: {desc}\n"
    return result


def main():
    """CLI 엔트리포인트"""
    if len(sys.argv) < 2:
        print(list_available_skills())
        return

    skill_name = sys.argv[1]

    if skill_name == "--list":
        print(list_available_skills())
        return

    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else GENERATED_SKILLS_PATH

    result = generate_skill(skill_name, output_dir)
    print(result)


if __name__ == "__main__":
    main()
