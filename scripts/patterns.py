#!/usr/bin/env python3
"""
cc-insights 패턴 감지 모듈
한국어 특화 프롬프트 패턴 분석
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter


@dataclass
class PatternMatch:
    """패턴 매칭 결과"""
    pattern_name: str
    prompt: str
    category: str  # vague, repetitive, inefficient
    issue: str
    suggestion: str
    confidence: float  # 0.0 - 1.0


# 한국어 모호한 요청 패턴
KOREAN_VAGUE_PATTERNS: List[Tuple[str, str, str]] = [
    # (regex, issue, suggestion_template)
    (r'^(이거|저거|그거|이것|저것|그것)\s',
     '지시대명사로 시작',
     '무엇을 가리키는지 구체적으로 명시해주세요'),

    (r'^(확인|체크|검토)(\s*(해줘|해봐|좀))?\.?$',
     '대상 미지정',
     '무엇을 확인할지 명시: "pm2 logs에서 ERROR 확인"'),

    (r'^(ㅇㅇ|응|네|넵|예)\s*(해줘|진행|배포|해봐)?',
     '확인 응답만',
     '어떤 작업을 진행할지 명시해주세요'),

    (r'^계속\s*(진행|해줘)?\.?$',
     '컨텍스트 의존적',
     '무엇을 계속할지 명시: "테스트 실행 계속 진행"'),

    (r'^(커밋|배포|테스트|빌드)(\s*(해줘|해봐))?\.?$',
     '단일 키워드',
     '옵션이나 범위 명시: "staging에 배포하고 헬스체크"'),

    (r'^(로그|상태|서버)\s*(확인|체크)?\.?$',
     '대상 불명확',
     '어떤 로그/상태인지 명시: "pm2 logs laytonalpha --lines 100"'),

    (r'^(고쳐|수정|변경)(\s*(해줘|해봐))?\.?$',
     '대상과 방법 미지정',
     '무엇을 어떻게 수정할지 명시'),

    (r'^(다음|이슈|문제)\s*(해결)?\.?$',
     '대상 불명확',
     '어떤 이슈인지 구체적으로 명시'),

    (r'^핸드오프\s*(작성|업데이트)?\.?$',
     '내용 미지정',
     '주요 변경사항과 컨텍스트 포함 권장'),

    (r'^A로\s*진행\.?$',
     '선택지 참조',
     '선택 내용을 명시: "Redis 캐싱 방식으로 진행"'),
]

# 영어 모호한 패턴 (혼용 대비)
ENGLISH_VAGUE_PATTERNS: List[Tuple[str, str, str]] = [
    (r'^(fix|update|change)\s+(it|this|that)\.?$',
     'Vague reference',
     'Specify what to fix and how'),

    (r'^(check|verify|test)\.?$',
     'Missing target',
     'Specify what to check'),

    (r'^continue\.?$',
     'Context dependent',
     'Specify what to continue'),
]

# 반복 가능한 작업 패턴 (skill 후보)
AUTOMATION_PATTERNS: List[Tuple[str, str, str]] = [
    # (regex, skill_name, description)
    (r'(로그|logs?)\s*(확인|체크|check)', 'log-check', '로그 확인 및 에러 분석'),
    (r'(핸드오프|handoff)\s*(작성|업데이트|write)', 'handoff', '핸드오프 문서 자동 작성'),
    (r'(배포|deploy)\s*(진행|하고|and)', 'deploy', '배포 및 상태 확인'),
    (r'(상태|status)\s*(확인|check)', 'status-check', '프로젝트 상태 종합 확인'),
    (r'(테스트|test)\s*(실행|run)', 'run-tests', '테스트 실행 및 결과 분석'),
    (r'(커밋|commit)', 'commit', '변경사항 커밋'),
    (r'(빌드|build)', 'build', '프로젝트 빌드'),
    (r'서버\s*(재시작|restart)', 'restart-server', '서버 재시작'),
]

# 비효율 패턴
INEFFICIENCY_PATTERNS: List[Tuple[str, str, str]] = [
    (r'^/clear\s*$',
     '/clear 사용',
     '`claude --continue` 사용 시 컨텍스트 유지 가능'),
]


def detect_vague_patterns(prompt: str) -> Optional[PatternMatch]:
    """모호한 프롬프트 패턴 감지"""
    prompt_lower = prompt.strip().lower()

    # 슬래시 커맨드는 제외
    if prompt.startswith('/'):
        return None

    # 너무 짧은 프롬프트
    if len(prompt.strip()) <= 5:
        return PatternMatch(
            pattern_name='too_short',
            prompt=prompt,
            category='vague',
            issue='너무 짧음 (5자 이하)',
            suggestion='구체적인 컨텍스트와 원하는 결과 명시',
            confidence=0.9
        )

    # 한국어 패턴 체크
    for pattern, issue, suggestion in KOREAN_VAGUE_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return PatternMatch(
                pattern_name=pattern[:20],
                prompt=prompt,
                category='vague',
                issue=issue,
                suggestion=suggestion,
                confidence=0.85
            )

    # 영어 패턴 체크
    for pattern, issue, suggestion in ENGLISH_VAGUE_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return PatternMatch(
                pattern_name=pattern[:20],
                prompt=prompt,
                category='vague',
                issue=issue,
                suggestion=suggestion,
                confidence=0.80
            )

    return None


def detect_automation_candidates(prompts: List[str]) -> List[dict]:
    """자동화 가능한 반복 패턴 감지"""
    from collections import defaultdict

    pattern_matches = defaultdict(list)

    for prompt in prompts:
        prompt_lower = prompt.lower()

        for pattern, skill_name, description in AUTOMATION_PATTERNS:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                pattern_matches[skill_name].append(prompt)
                break

    # 3회 이상 반복된 것만 추천
    candidates = []
    for skill_name, matched_prompts in pattern_matches.items():
        if len(matched_prompts) >= 3:
            description = next(
                d for p, s, d in AUTOMATION_PATTERNS if s == skill_name
            )
            candidates.append({
                'skill_name': skill_name,
                'description': description,
                'count': len(matched_prompts),
                'samples': matched_prompts[:5],
                'confidence': min(0.95, 0.6 + len(matched_prompts) * 0.05)
            })

    return sorted(candidates, key=lambda x: x['count'], reverse=True)


def detect_inefficiency_patterns(prompts: List[str]) -> List[dict]:
    """비효율 패턴 감지"""
    inefficiencies = []

    for pattern, issue, suggestion in INEFFICIENCY_PATTERNS:
        matches = [p for p in prompts if re.search(pattern, p, re.IGNORECASE)]
        if matches:
            inefficiencies.append({
                'pattern': issue,
                'count': len(matches),
                'suggestion': suggestion,
                'samples': matches[:3]
            })

    return inefficiencies


def calculate_prompt_quality_score(prompt: str) -> Tuple[int, List[str]]:
    """프롬프트 품질 점수 계산 (0-10)"""
    score = 10
    issues = []

    # 길이 체크
    if len(prompt) < 10:
        score -= 3
        issues.append('너무 짧음')
    elif len(prompt) < 20:
        score -= 1
        issues.append('다소 짧음')

    # 지시대명사 사용
    if re.search(r'(이거|저거|그거|이것|저것|그것|it|this|that)\s', prompt, re.IGNORECASE):
        score -= 2
        issues.append('모호한 지시대명사')

    # 파일 경로 포함 여부 (좋은 징후)
    if re.search(r'[/\\][\w.-]+\.(py|js|ts|md|json|yaml|tsx|jsx)', prompt):
        score += 1

    # 구체적 명령어 포함 (좋은 징후)
    if re.search(r'(pm2|npm|git|docker|python|node)\s+\w+', prompt):
        score += 1

    # 에러 메시지 포함 (좋은 징후)
    if re.search(r'(error|에러|오류|exception|failed|실패)', prompt, re.IGNORECASE):
        score += 0.5

    # 모호한 동사만 사용
    vague_verbs = re.search(r'^(확인|체크|고쳐|수정|해줘|해봐|check|fix|update)\s*$', prompt, re.IGNORECASE)
    if vague_verbs:
        score -= 3
        issues.append('대상 없는 동사만 사용')

    return max(0, min(10, int(score))), issues


def get_improvement_suggestion(prompt: str, project_context: Optional[str] = None) -> str:
    """프롬프트 개선 제안 생성"""
    prompt_lower = prompt.lower()

    suggestions = {
        '로그': f'"pm2 logs {project_context or "<project>"} --lines 100에서 ERROR 레벨 로그 확인하고 원인 분석"',
        '확인': '"<대상> 상태를 확인하고 문제가 있으면 원인과 해결책 제시"',
        '배포': '"staging 환경에 배포하고 헬스체크 결과 확인"',
        '커밋': '"현재 변경사항을 커밋하고 메시지는 <내용> 포함"',
        '테스트': '"<모듈> 테스트를 실행하고 실패하면 원인 분석"',
        '핸드오프': '"오늘 작업 내용과 다음 단계를 핸드오프 문서에 정리"',
    }

    for keyword, suggestion in suggestions.items():
        if keyword in prompt_lower:
            return suggestion

    return '"무엇을(What), 어디서(Where), 왜(Why), 어떻게(How)를 포함해서 작성"'


def analyze_session_patterns(prompts_with_sessions: List[dict]) -> dict:
    """세션 관리 패턴 분석"""
    from collections import defaultdict

    sessions_per_project = defaultdict(set)
    clear_count = 0

    for entry in prompts_with_sessions:
        project = entry.get('project', 'unknown')
        session_id = entry.get('session_id', '')
        prompt = entry.get('prompt', '')

        if project:
            sessions_per_project[project].add(session_id)

        if prompt.strip().startswith('/clear'):
            clear_count += 1

    # 동시 세션이 많은 프로젝트 찾기
    multi_session_projects = {
        proj: len(sessions)
        for proj, sessions in sessions_per_project.items()
        if len(sessions) > 2
    }

    return {
        'multi_session_projects': multi_session_projects,
        'clear_count': clear_count,
        'total_projects': len(sessions_per_project),
        'avg_sessions_per_project': sum(len(s) for s in sessions_per_project.values()) / max(1, len(sessions_per_project))
    }


# ========== 새로운 분석 함수들 ==========

def detect_workflow_patterns(prompts: List[str]) -> List[dict]:
    """워크플로우 패턴 감지 - 연속적인 작업 흐름 파악"""
    workflows = []

    # 일반적인 워크플로우 시퀀스 정의
    WORKFLOW_SEQUENCES = [
        {
            'name': 'git-workflow',
            'description': 'Git 작업 흐름',
            'patterns': [r'(status|diff|add|commit|push|pull)', r'(커밋|푸시|풀|상태)'],
            'steps': ['변경사항 확인', '스테이징', '커밋', '푸시']
        },
        {
            'name': 'test-debug',
            'description': '테스트 및 디버깅',
            'patterns': [r'(test|테스트)', r'(error|에러|버그|bug)', r'(fix|수정|고쳐)'],
            'steps': ['테스트 실행', '에러 확인', '수정']
        },
        {
            'name': 'deploy-check',
            'description': '배포 및 확인',
            'patterns': [r'(deploy|배포)', r'(status|상태|확인|check)', r'(log|로그)'],
            'steps': ['배포', '상태 확인', '로그 확인']
        }
    ]

    for workflow in WORKFLOW_SEQUENCES:
        matches = 0
        for pattern_group in workflow['patterns']:
            if any(re.search(pattern_group, p, re.IGNORECASE) for p in prompts):
                matches += 1

        if matches >= 2:  # 최소 2개 패턴 매칭
            workflows.append({
                'name': workflow['name'],
                'description': workflow['description'],
                'confidence': round(matches / len(workflow['patterns']), 2),
                'suggested_skill': f"/{workflow['name']}"
            })

    return workflows


def detect_learning_patterns(prompts: List[str]) -> dict:
    """학습 패턴 감지 - 사용자의 질문/학습 스타일 분석"""
    learning_indicators = {
        'questions': [],  # 질문 형태
        'explanations': [],  # 설명 요청
        'examples': [],  # 예시 요청
        'debugging': []  # 디버깅 질문
    }

    QUESTION_PATTERNS = [
        (r'(왜|why|어떻게|how|무엇|what|언제|when)', 'questions'),
        (r'(설명|explain|알려줘|tell me)', 'explanations'),
        (r'(예시|example|샘플|sample)', 'examples'),
        (r'(에러|error|버그|bug|안 되|doesn\'t work|실패|fail)', 'debugging')
    ]

    for prompt in prompts:
        for pattern, category in QUESTION_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                learning_indicators[category].append(prompt)
                break

    total = len(prompts)
    return {
        'question_ratio': round(len(learning_indicators['questions']) / max(1, total) * 100, 1),
        'explanation_ratio': round(len(learning_indicators['explanations']) / max(1, total) * 100, 1),
        'example_requests': len(learning_indicators['examples']),
        'debugging_questions': len(learning_indicators['debugging']),
        'learning_style': _determine_learning_style(learning_indicators, total)
    }


def _determine_learning_style(indicators: dict, total: int) -> str:
    """학습 스타일 판단"""
    question_ratio = len(indicators['questions']) / max(1, total)
    example_ratio = len(indicators['examples']) / max(1, total)
    debug_ratio = len(indicators['debugging']) / max(1, total)

    if debug_ratio > 0.3:
        return "🔧 실습형 (직접 해보면서 배움)"
    elif question_ratio > 0.2:
        return "🤔 탐구형 (이유와 원리 중시)"
    elif example_ratio > 0.1:
        return "📖 예시형 (구체적 예시로 이해)"
    else:
        return "⚡ 실행형 (바로 실행하고 결과 확인)"


def calculate_efficiency_score(prompts: List[str], sessions: dict) -> dict:
    """효율성 점수 계산"""
    score = 100
    issues = []

    # 1. /clear 과다 사용 (-5점씩)
    clear_count = sum(1 for p in prompts if p.strip() == '/clear')
    clear_penalty = min(20, clear_count * 5)
    if clear_penalty > 0:
        score -= clear_penalty
        issues.append(f"/clear {clear_count}회 사용")

    # 2. 너무 짧은 프롬프트 비율 (-최대 15점)
    short_prompts = sum(1 for p in prompts if len(p) < 15 and not p.startswith('/'))
    short_ratio = short_prompts / max(1, len(prompts))
    if short_ratio > 0.3:
        penalty = min(15, int(short_ratio * 30))
        score -= penalty
        issues.append(f"짧은 프롬프트 {round(short_ratio * 100)}%")

    # 3. 동시 세션 과다 (-최대 10점)
    if len(sessions) > 5:
        penalty = min(10, (len(sessions) - 5) * 2)
        score -= penalty
        issues.append(f"동시 세션 {len(sessions)}개")

    # 4. 모호한 프롬프트 비율 (-최대 15점)
    vague_count = sum(1 for p in prompts if detect_vague_patterns(p) is not None)
    vague_ratio = vague_count / max(1, len(prompts))
    if vague_ratio > 0.2:
        penalty = min(15, int(vague_ratio * 40))
        score -= penalty
        issues.append(f"모호한 프롬프트 {round(vague_ratio * 100)}%")

    # 등급 계산
    if score >= 90:
        grade = "S"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    else:
        grade = "D"

    return {
        'score': max(0, score),
        'grade': grade,
        'issues': issues,
        'recommendations': _get_efficiency_recommendations(issues)
    }


def _get_efficiency_recommendations(issues: List[str]) -> List[str]:
    """효율성 개선 권장사항"""
    recommendations = []

    for issue in issues:
        if '/clear' in issue:
            recommendations.append("→ `claude --continue`로 컨텍스트 유지")
        elif '짧은 프롬프트' in issue:
            recommendations.append("→ 프롬프트에 컨텍스트와 기대 결과 추가")
        elif '동시 세션' in issue:
            recommendations.append("→ `/rename`으로 세션 정리")
        elif '모호한' in issue:
            recommendations.append("→ 지시대명사 대신 구체적 대상 명시")

    return recommendations


# 추가 패턴 정의
COMPLEXITY_PATTERNS = [
    (r'(복잡한|complex|advanced|고급)', 'high'),
    (r'(간단한|simple|basic|기본)', 'low'),
    (r'(중간|moderate|일반)', 'medium')
]

DOMAIN_PATTERNS = [
    (r'(api|endpoint|rest|graphql)', 'backend'),
    (r'(component|ui|css|style|design)', 'frontend'),
    (r'(db|database|query|sql)', 'database'),
    (r'(deploy|ci|cd|docker|k8s)', 'devops'),
    (r'(test|spec|coverage)', 'testing')
]


def detect_domain_focus(prompts: List[str]) -> dict:
    """작업 도메인 분석"""
    domain_counts = Counter()

    for prompt in prompts:
        prompt_lower = prompt.lower()
        for pattern, domain in DOMAIN_PATTERNS:
            if re.search(pattern, prompt_lower):
                domain_counts[domain] += 1
                break

    total = sum(domain_counts.values())
    return {
        'primary_domain': domain_counts.most_common(1)[0][0] if domain_counts else 'general',
        'domain_distribution': dict(domain_counts),
        'focus_ratio': round(domain_counts.most_common(1)[0][1] / max(1, total) * 100, 1) if domain_counts else 0
    }
