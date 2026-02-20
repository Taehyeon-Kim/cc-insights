#!/usr/bin/env python3
"""
cc-insights 메인 분석기
history.jsonl을 분석하여 개인화된 인사이트 제공
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any, Tuple
import statistics

# 같은 디렉토리의 patterns 모듈 import
sys.path.insert(0, str(Path(__file__).parent))
from patterns import (
    detect_vague_patterns,
    detect_automation_candidates,
    detect_inefficiency_patterns,
    calculate_prompt_quality_score,
    get_improvement_suggestion,
    analyze_session_patterns,
    detect_workflow_patterns,
    detect_learning_patterns,
    calculate_efficiency_score,
)

# 설정
HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"
PLUGIN_DATA_PATH = Path(__file__).parent.parent / "data"
BASELINE_PATH = PLUGIN_DATA_PATH / "baseline.json"
REPORTS_PATH = Path(__file__).parent.parent / "reports"


@dataclass
class PromptEntry:
    """프롬프트 엔트리"""
    display: str
    timestamp: datetime
    project: str
    session_id: str

    @classmethod
    def from_json(cls, data: dict) -> "PromptEntry":
        return cls(
            display=data.get("display", ""),
            timestamp=datetime.fromtimestamp(data.get("timestamp", 0) / 1000),
            project=data.get("project", ""),
            session_id=data.get("sessionId", "")
        )


class CCInsightsAnalyzer:
    """cc-insights 메인 분석기"""

    def __init__(self, days: int = 7):
        self.days = days
        self.entries: List[PromptEntry] = []
        self.cutoff = datetime.now() - timedelta(days=days)
        self.baseline: Optional[dict] = None

    def load_history(self, since: Optional[datetime] = None, limit: Optional[int] = None) -> int:
        """히스토리 로드"""
        if not HISTORY_PATH.exists():
            print(f"[오류] history.jsonl을 찾을 수 없습니다: {HISTORY_PATH}", file=sys.stderr)
            return 0

        cutoff = since or self.cutoff
        count = 0

        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    entry = PromptEntry.from_json(data)
                    if entry.timestamp >= cutoff:
                        self.entries.append(entry)
                        count += 1
                        if limit and count >= limit:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue

        return count

    def load_all_history(self) -> int:
        """전체 히스토리 로드 (baseline 생성용)"""
        if not HISTORY_PATH.exists():
            return 0

        count = 0
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    entry = PromptEntry.from_json(data)
                    self.entries.append(entry)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue

        return count

    def load_baseline(self) -> bool:
        """baseline 데이터 로드"""
        if BASELINE_PATH.exists():
            with open(BASELINE_PATH, "r", encoding="utf-8") as f:
                self.baseline = json.load(f)
            return True
        return False

    def save_baseline(self, data: dict) -> None:
        """baseline 데이터 저장"""
        PLUGIN_DATA_PATH.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def analyze_vague_prompts(self) -> List[dict]:
        """모호한 프롬프트 분석"""
        results = []

        for entry in self.entries:
            match = detect_vague_patterns(entry.display)
            if match:
                results.append({
                    "prompt": entry.display,
                    "issue": match.issue,
                    "suggestion": match.suggestion,
                    "confidence": match.confidence,
                    "timestamp": entry.timestamp.isoformat(),
                    "project": Path(entry.project).name if entry.project else "unknown"
                })

        return results

    def analyze_quality_scores(self) -> dict:
        """프롬프트 품질 점수 분석"""
        scores = []
        low_quality = []  # 0-4점

        for entry in self.entries:
            if entry.display.startswith('/'):
                continue  # 슬래시 커맨드 제외

            score, issues = calculate_prompt_quality_score(entry.display)
            scores.append(score)

            if score <= 4:
                low_quality.append({
                    "prompt": entry.display,
                    "score": score,
                    "issues": issues,
                    "improvement": get_improvement_suggestion(
                        entry.display,
                        Path(entry.project).name if entry.project else None
                    )
                })

        if not scores:
            return {"avg_score": 0, "distribution": {}, "low_quality": []}

        return {
            "avg_score": round(sum(scores) / len(scores), 1),
            "distribution": dict(Counter(scores)),
            "total_prompts": len(scores),
            "low_quality_count": len(low_quality),
            "low_quality": low_quality[:10]  # 상위 10개만
        }

    def analyze_automation_candidates(self) -> List[dict]:
        """자동화 후보 분석"""
        prompts = [e.display for e in self.entries]
        return detect_automation_candidates(prompts)

    def analyze_inefficiencies(self) -> List[dict]:
        """비효율 패턴 분석"""
        prompts = [e.display for e in self.entries]
        return detect_inefficiency_patterns(prompts)

    def analyze_session_management(self) -> dict:
        """세션 관리 패턴 분석"""
        data = [
            {"project": e.project, "session_id": e.session_id, "prompt": e.display}
            for e in self.entries
        ]
        return analyze_session_patterns(data)

    def analyze_time_patterns(self) -> dict:
        """시간 패턴 분석"""
        hourly = Counter()
        daily = Counter()
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']

        for entry in self.entries:
            hourly[entry.timestamp.hour] += 1
            daily[weekday_names[entry.timestamp.weekday()]] += 1

        peak_hours = hourly.most_common(3)
        peak_days = daily.most_common(3)

        # 야간 작업 비율 (22-06시)
        night_count = sum(hourly.get(h, 0) for h in list(range(22, 24)) + list(range(0, 6)))
        total = sum(hourly.values())
        night_ratio = round(night_count / max(1, total) * 100, 1)

        return {
            "hourly_distribution": dict(hourly),
            "daily_distribution": dict(daily),
            "peak_hours": peak_hours,
            "peak_days": peak_days,
            "night_work_ratio": night_ratio,
            "recommendation": self._get_time_recommendation(peak_hours, night_ratio)
        }

    def analyze_project_patterns(self) -> dict:
        """프로젝트별 패턴 분석"""
        project_stats = defaultdict(lambda: {"count": 0, "sessions": set()})

        for entry in self.entries:
            proj_name = Path(entry.project).name if entry.project else "unknown"
            project_stats[proj_name]["count"] += 1
            project_stats[proj_name]["sessions"].add(entry.session_id)

        # 세션 수 계산
        result = {}
        for proj, stats in project_stats.items():
            result[proj] = {
                "count": stats["count"],
                "sessions": len(stats["sessions"])
            }

        return dict(sorted(result.items(), key=lambda x: x[1]["count"], reverse=True))

    def analyze_project_detail(self, project_name: str) -> dict:
        """특정 프로젝트 상세 분석"""
        # 해당 프로젝트 엔트리만 필터링
        project_entries = [
            e for e in self.entries
            if (Path(e.project).name if e.project else "unknown") == project_name
        ]

        if not project_entries:
            return {"error": f"프로젝트 '{project_name}'을(를) 찾을 수 없습니다."}

        # 품질 점수 분석
        scores = []
        low_quality = []
        for entry in project_entries:
            if entry.display.startswith('/'):
                continue
            score, issues = calculate_prompt_quality_score(entry.display)
            scores.append(score)
            if score <= 4:
                low_quality.append({
                    "prompt": entry.display,
                    "score": score,
                    "issues": issues
                })

        avg_score = round(sum(scores) / len(scores), 1) if scores else 0

        # 시간 패턴
        hourly = Counter(e.timestamp.hour for e in project_entries)
        peak_hours = hourly.most_common(3)

        # 세션 분석
        sessions = defaultdict(int)
        for e in project_entries:
            sessions[e.session_id] += 1
        avg_prompts_per_session = statistics.mean(sessions.values()) if sessions else 0

        # 자동화 후보
        prompts = [e.display for e in project_entries]
        automation = detect_automation_candidates(prompts)

        # 모호한 프롬프트
        vague_count = sum(1 for e in project_entries if detect_vague_patterns(e.display))

        # 날짜 범위
        start_date = min(e.timestamp for e in project_entries)
        end_date = max(e.timestamp for e in project_entries)

        return {
            "project_name": project_name,
            "total_prompts": len(project_entries),
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
                "days": (end_date - start_date).days + 1
            },
            "quality": {
                "avg_score": avg_score,
                "low_quality_count": len(low_quality),
                "low_quality_ratio": round(len(low_quality) / max(1, len(scores)) * 100, 1)
            },
            "vague": {
                "count": vague_count,
                "ratio": round(vague_count / max(1, len(project_entries)) * 100, 1)
            },
            "sessions": {
                "total": len(sessions),
                "avg_prompts_per_session": round(avg_prompts_per_session, 1)
            },
            "time_patterns": {
                "peak_hours": peak_hours,
                "hourly_distribution": dict(hourly)
            },
            "automation_candidates": automation[:3],
            "low_quality_samples": low_quality[:5]
        }

    def generate_projects_output(self) -> str:
        """프로젝트 목록 및 통계 출력"""
        projects = self.analyze_project_patterns()
        total = len(self.entries)

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║  📁 프로젝트별 분석 (최근 {self.days}일)                        ║
╚══════════════════════════════════════════════════════════════╝

  총 {len(projects)}개 프로젝트, {total:,}개 프롬프트

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  프로젝트별 활동량
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        for i, (proj, stats) in enumerate(list(projects.items())[:15], 1):
            ratio = stats['count'] / max(1, total) * 100
            bar_len = min(20, int(ratio))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            output += f"  {i:2}. {proj[:25]:<25} [{bar}] {stats['count']:>4}개 ({ratio:>4.1f}%)\n"
            output += f"      └─ 세션: {stats['sessions']}개\n"

        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  💡 특정 프로젝트 상세 분석:
     python3 analyzer.py projects --project <프로젝트명>
"""
        return output

    def analyze_overall_stats(self) -> dict:
        """전체 사용 통계 분석 (전체 히스토리 기반)"""
        if not self.entries:
            return {}

        # 첫 사용 시간
        first_use = min(e.timestamp for e in self.entries)
        last_use = max(e.timestamp for e in self.entries)
        total_days = (last_use - first_use).days + 1

        # 프로젝트 통계
        projects = set()
        for e in self.entries:
            proj_name = Path(e.project).name if e.project else "unknown"
            if proj_name != "unknown":
                projects.add(proj_name)

        # 세션 통계
        sessions = set(e.session_id for e in self.entries)

        # 프롬프트 통계
        total_prompts = len(self.entries)
        non_command_prompts = [e for e in self.entries if not e.display.startswith('/')]

        # 토큰 추정 (한글 1자 ≈ 1.5토큰, 영문 1단어 ≈ 1.3토큰으로 대략 추정)
        total_chars = sum(len(e.display) for e in non_command_prompts)
        # 간단히 문자 수 / 4 로 입력 토큰 추정 (보수적 추정)
        estimated_input_tokens = total_chars // 4
        # 출력 토큰은 입력의 약 5-10배로 추정 (코드 생성 등 고려)
        estimated_output_tokens = estimated_input_tokens * 7
        estimated_total_tokens = estimated_input_tokens + estimated_output_tokens

        # 월별 통계
        monthly_stats = defaultdict(lambda: {"prompts": 0, "sessions": set(), "projects": set()})
        for e in self.entries:
            month_key = e.timestamp.strftime("%Y-%m")
            monthly_stats[month_key]["prompts"] += 1
            monthly_stats[month_key]["sessions"].add(e.session_id)
            proj_name = Path(e.project).name if e.project else "unknown"
            if proj_name != "unknown":
                monthly_stats[month_key]["projects"].add(proj_name)

        # 월별 데이터 정리
        monthly_data = []
        for month, stats in sorted(monthly_stats.items()):
            monthly_data.append({
                "month": month,
                "prompts": stats["prompts"],
                "sessions": len(stats["sessions"]),
                "projects": len(stats["projects"])
            })

        # 시간대별 활동
        hourly = Counter(e.timestamp.hour for e in self.entries)
        peak_hour = hourly.most_common(1)[0] if hourly else (0, 0)

        # 요일별 활동
        weekday_names = ['월', '화', '수', '목', '금', '토', '일']
        daily = Counter(weekday_names[e.timestamp.weekday()] for e in self.entries)
        peak_day = daily.most_common(1)[0] if daily else ("", 0)

        # 평균 프롬프트 길이
        avg_prompt_length = statistics.mean(len(e.display) for e in non_command_prompts) if non_command_prompts else 0

        return {
            "first_use": first_use,
            "last_use": last_use,
            "total_days": total_days,
            "total_prompts": total_prompts,
            "total_projects": len(projects),
            "total_sessions": len(sessions),
            "avg_prompts_per_day": round(total_prompts / max(1, total_days), 1),
            "avg_sessions_per_day": round(len(sessions) / max(1, total_days), 2),
            "avg_prompt_length": round(avg_prompt_length, 1),
            "tokens": {
                "estimated_input": estimated_input_tokens,
                "estimated_output": estimated_output_tokens,
                "estimated_total": estimated_total_tokens,
                "note": "프롬프트 길이 기반 추정치 (실제와 다를 수 있음)"
            },
            "peak_hour": peak_hour,
            "peak_day": peak_day,
            "monthly_data": monthly_data[-12:],  # 최근 12개월
            "projects_list": sorted(projects)
        }

    def generate_stats_output(self) -> str:
        """전체 사용 통계 출력"""
        stats = self.analyze_overall_stats()

        if not stats:
            return "\n  ❌ 분석할 데이터가 없습니다.\n"

        # 토큰 포맷팅
        def format_tokens(n):
            if n >= 1_000_000_000:
                return f"{n / 1_000_000_000:.1f}B"
            elif n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n / 1_000:.1f}K"
            return str(n)

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 Claude Code 전체 사용 통계                                ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🗓️  사용 기간
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 첫 사용: {stats['first_use'].strftime("%Y-%m-%d %H:%M")}
  • 마지막 사용: {stats['last_use'].strftime("%Y-%m-%d %H:%M")}
  • 총 사용 기간: {stats['total_days']}일

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📈 활동 통계
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 총 프롬프트: {stats['total_prompts']:,}개
  • 총 프로젝트: {stats['total_projects']}개
  • 총 세션: {stats['total_sessions']:,}개
  • 하루 평균 프롬프트: {stats['avg_prompts_per_day']}개
  • 하루 평균 세션: {stats['avg_sessions_per_day']}개
  • 평균 프롬프트 길이: {stats['avg_prompt_length']}자

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔢 토큰 사용량 (추정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 입력 토큰: ~{format_tokens(stats['tokens']['estimated_input'])}
  • 출력 토큰: ~{format_tokens(stats['tokens']['estimated_output'])}
  • 총 토큰: ~{format_tokens(stats['tokens']['estimated_total'])}
  ⚠️  {stats['tokens']['note']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏰ 작업 패턴
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 가장 활발한 시간: {stats['peak_hour'][0]}시 ({stats['peak_hour'][1]:,}회)
  • 가장 활발한 요일: {stats['peak_day'][0]}요일 ({stats['peak_day'][1]:,}회)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📅 월별 활동
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        max_prompts = max((m['prompts'] for m in stats['monthly_data']), default=1)
        for month in stats['monthly_data']:
            bar_len = int(month['prompts'] / max_prompts * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            output += f"  {month['month']} [{bar}] {month['prompts']:>5}개 | {month['sessions']:>3}세션 | {month['projects']:>2}프로젝트\n"

        output += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📁 진행한 프로젝트 ({stats['total_projects']}개)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        # 프로젝트 목록 (최대 20개)
        projects_to_show = stats['projects_list'][:20]
        for i, proj in enumerate(projects_to_show, 1):
            output += f"  {i:2}. {proj}\n"

        if len(stats['projects_list']) > 20:
            output += f"  ... 외 {len(stats['projects_list']) - 20}개\n"

        return output

    def generate_project_detail_output(self, project_name: str) -> str:
        """특정 프로젝트 상세 분석 출력"""
        detail = self.analyze_project_detail(project_name)

        if "error" in detail:
            return f"\n  ❌ {detail['error']}\n"

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║  📁 프로젝트 상세 분석: {detail['project_name'][:30]:<30}  ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 기본 정보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 총 프롬프트: {detail['total_prompts']:,}개
  • 기간: {detail['date_range']['start']} ~ {detail['date_range']['end']} ({detail['date_range']['days']}일)
  • 세션 수: {detail['sessions']['total']}개
  • 세션당 평균: {detail['sessions']['avg_prompts_per_session']}개 프롬프트

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⭐ 품질 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 평균 품질 점수: {detail['quality']['avg_score']}/10
  • 저품질 프롬프트: {detail['quality']['low_quality_count']}개 ({detail['quality']['low_quality_ratio']}%)
  • 모호한 프롬프트: {detail['vague']['count']}개 ({detail['vague']['ratio']}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏰ 작업 시간대
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 피크 시간: {', '.join(f"{h[0]}시 ({h[1]}회)" for h in detail['time_patterns']['peak_hours'])}
"""

        if detail['automation_candidates']:
            output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🤖 Skill 자동화 후보
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            for skill in detail['automation_candidates']:
                output += f"\n  • /{skill['skill_name']}: {skill['count']}회 ({skill['description']})"

        if detail['low_quality_samples']:
            output += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📝 개선 필요 프롬프트 예시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            for sample in detail['low_quality_samples'][:3]:
                short = sample['prompt'][:50] + '...' if len(sample['prompt']) > 50 else sample['prompt']
                output += f"\n  • \"{short}\" (점수: {sample['score']}/10)"

        return output

    def _get_time_recommendation(self, peak_hours, night_ratio) -> str:
        """시간 패턴 기반 추천"""
        recommendations = []

        if peak_hours:
            hour = peak_hours[0][0]
            if 9 <= hour <= 12:
                recommendations.append("오전에 집중도가 높습니다. 복잡한 작업은 오전에 배치하세요.")
            elif 14 <= hour <= 18:
                recommendations.append("오후에 활동량이 많습니다.")
            elif 19 <= hour <= 23:
                recommendations.append("저녁/야간 작업이 많습니다.")

        if night_ratio > 20:
            recommendations.append(f"야간 작업 {night_ratio}% → handoff 자동화 skill 권장")

        return " ".join(recommendations) if recommendations else "균형 잡힌 작업 패턴입니다."

    def generate_baseline(self) -> dict:
        """전체 히스토리 기반 baseline 생성"""
        total = len(self.entries)
        if total == 0:
            return {}

        # 프롬프트 빈도
        prompt_freq = Counter(e.display.lower().strip() for e in self.entries)

        # 프로젝트 빈도
        project_freq = Counter(
            Path(e.project).name if e.project else "unknown"
            for e in self.entries
        )

        # 자동화 후보
        automation = self.analyze_automation_candidates()

        # 시간 패턴
        hourly = Counter(e.timestamp.hour for e in self.entries)

        baseline = {
            "generated_at": datetime.now().isoformat(),
            "total_prompts": total,
            "date_range": {
                "start": min(e.timestamp for e in self.entries).isoformat(),
                "end": max(e.timestamp for e in self.entries).isoformat()
            },
            "top_prompts": prompt_freq.most_common(20),
            "top_projects": project_freq.most_common(10),
            "automation_candidates": automation,
            "hourly_avg": {h: c / max(1, total) for h, c in hourly.items()},
            "avg_prompts_per_day": round(total / max(1, (max(e.timestamp for e in self.entries) - min(e.timestamp for e in self.entries)).days + 1), 1)
        }

        return baseline

    def compare_with_baseline(self, current_stats: dict) -> dict:
        """baseline과 현재 통계 비교"""
        if not self.baseline:
            return {"has_baseline": False}

        comparisons = {}

        # 자동화 후보 비교
        baseline_automation = {c["skill_name"]: c["count"] for c in self.baseline.get("automation_candidates", [])}
        current_automation = {c["skill_name"]: c["count"] for c in current_stats.get("automation_candidates", [])}

        for skill, count in current_automation.items():
            baseline_count = baseline_automation.get(skill, 0)
            if baseline_count > 0:
                change = round((count - baseline_count) / baseline_count * 100, 1)
                comparisons[skill] = {
                    "current": count,
                    "baseline_avg": baseline_count,
                    "change_percent": change
                }

        return {
            "has_baseline": True,
            "baseline_date": self.baseline.get("generated_at"),
            "comparisons": comparisons
        }

    # ========== 새로운 분석 메서드 ==========

    def generate_user_profile(self) -> dict:
        """사용자 프로필 생성 - 개발 스타일 및 특성 분석"""
        if not self.entries:
            return {}

        # 시간대별 분포
        hourly = Counter(e.timestamp.hour for e in self.entries)
        total_hours = sum(hourly.values())

        # 개발 스타일 판단
        morning_ratio = sum(hourly.get(h, 0) for h in range(6, 12)) / max(1, total_hours)
        afternoon_ratio = sum(hourly.get(h, 0) for h in range(12, 18)) / max(1, total_hours)
        evening_ratio = sum(hourly.get(h, 0) for h in range(18, 24)) / max(1, total_hours)
        night_ratio = sum(hourly.get(h, 0) for h in list(range(0, 6))) / max(1, total_hours)

        if morning_ratio > 0.4:
            work_style = "🌅 얼리버드 (오전 집중형)"
        elif evening_ratio > 0.35:
            work_style = "🌙 나이트 아울 (저녁 집중형)"
        elif night_ratio > 0.2:
            work_style = "🦉 심야형 개발자"
        else:
            work_style = "☀️ 균형 잡힌 스타일"

        # 세션당 평균 프롬프트 수
        sessions = defaultdict(int)
        for e in self.entries:
            sessions[e.session_id] += 1
        avg_prompts_per_session = statistics.mean(sessions.values()) if sessions else 0

        if avg_prompts_per_session > 50:
            session_style = "🔥 딥다이브 (긴 세션 선호)"
        elif avg_prompts_per_session < 15:
            session_style = "⚡ 스프린터 (짧은 세션 선호)"
        else:
            session_style = "🎯 밸런서 (적절한 세션 길이)"

        # 프로젝트 패턴
        projects = Counter(Path(e.project).name if e.project else "unknown" for e in self.entries)
        top_projects = projects.most_common(5)

        if len(projects) == 1:
            project_style = "🎯 집중형 (단일 프로젝트)"
        elif len(projects) <= 3:
            project_style = "📁 포커스형 (소수 프로젝트)"
        else:
            project_style = "🌐 멀티태스커 (다중 프로젝트)"

        # 프롬프트 스타일 분석
        avg_prompt_length = statistics.mean(len(e.display) for e in self.entries if not e.display.startswith('/'))

        if avg_prompt_length > 100:
            prompt_style = "📝 상세파 (긴 프롬프트)"
        elif avg_prompt_length < 30:
            prompt_style = "💨 간결파 (짧은 프롬프트)"
        else:
            prompt_style = "✍️ 적정파 (적절한 길이)"

        return {
            "work_style": work_style,
            "session_style": session_style,
            "project_style": project_style,
            "prompt_style": prompt_style,
            "metrics": {
                "avg_prompts_per_session": round(avg_prompts_per_session, 1),
                "avg_prompt_length": round(avg_prompt_length, 1),
                "total_sessions": len(sessions),
                "total_projects": len(projects),
                "morning_ratio": round(morning_ratio * 100, 1),
                "evening_ratio": round(evening_ratio * 100, 1),
            },
            "top_projects": top_projects
        }

    def analyze_strengths_weaknesses(self) -> dict:
        """강점/약점 분석"""
        quality = self.analyze_quality_scores()
        automation = self.analyze_automation_candidates()
        vague = self.analyze_vague_prompts()
        inefficiencies = self.analyze_inefficiencies()

        strengths = []
        weaknesses = []

        # 품질 점수 기반
        avg_score = quality.get('avg_score', 0)
        if avg_score >= 8:
            strengths.append({
                "area": "프롬프트 품질",
                "detail": f"평균 {avg_score}/10점으로 매우 우수",
                "emoji": "✨"
            })
        elif avg_score <= 5:
            weaknesses.append({
                "area": "프롬프트 품질",
                "detail": f"평균 {avg_score}/10점 - 개선 필요",
                "suggestion": "구체적인 컨텍스트와 기대 결과를 명시하세요",
                "emoji": "📝"
            })

        # 모호한 프롬프트 비율
        total = len(self.entries)
        vague_ratio = len(vague) / max(1, total)
        if vague_ratio < 0.1:
            strengths.append({
                "area": "명확한 커뮤니케이션",
                "detail": f"모호한 프롬프트 {round(vague_ratio * 100, 1)}%로 매우 낮음",
                "emoji": "🎯"
            })
        elif vague_ratio > 0.2:
            weaknesses.append({
                "area": "프롬프트 명확성",
                "detail": f"모호한 프롬프트 {round(vague_ratio * 100, 1)}%",
                "suggestion": "지시대명사(이거, 저거) 대신 구체적 대상 명시",
                "emoji": "🔍"
            })

        # 자동화 패턴
        if automation:
            strengths.append({
                "area": "일관된 워크플로우",
                "detail": f"{len(automation)}개의 자동화 가능한 패턴 발견",
                "emoji": "🔄"
            })

        # 비효율 패턴
        clear_count = sum(1 for e in self.entries if e.display.strip() == '/clear')
        if clear_count > 5:
            weaknesses.append({
                "area": "/clear 과다 사용",
                "detail": f"{clear_count}회 사용",
                "suggestion": "`claude --continue` 사용으로 컨텍스트 유지",
                "emoji": "⚠️"
            })

        # 프롬프트 길이 분석
        non_command_prompts = [e for e in self.entries if not e.display.startswith('/')]
        if non_command_prompts:
            lengths = [len(e.display) for e in non_command_prompts]
            short_prompts = sum(1 for l in lengths if l < 15)
            if short_prompts / len(lengths) > 0.3:
                weaknesses.append({
                    "area": "프롬프트 길이",
                    "detail": f"15자 미만 프롬프트 {round(short_prompts / len(lengths) * 100)}%",
                    "suggestion": "컨텍스트, 예상 결과, 제약 조건 포함 권장",
                    "emoji": "📏"
                })

        return {
            "strengths": strengths[:5],  # 최대 5개
            "weaknesses": weaknesses[:5],
            "overall_grade": self._calculate_overall_grade(strengths, weaknesses)
        }

    def _calculate_overall_grade(self, strengths: list, weaknesses: list) -> str:
        """종합 등급 계산"""
        score = len(strengths) * 2 - len(weaknesses)
        if score >= 6:
            return "S"
        elif score >= 4:
            return "A"
        elif score >= 2:
            return "B"
        elif score >= 0:
            return "C"
        else:
            return "D"

    def generate_action_items(self) -> List[dict]:
        """구체적인 액션 아이템 생성"""
        actions = []

        # 분석 수행
        vague = self.analyze_vague_prompts()
        automation = self.analyze_automation_candidates()
        quality = self.analyze_quality_scores()

        # 1. Skill 생성 제안
        if automation:
            top_skill = automation[0]
            actions.append({
                "priority": 1,
                "category": "자동화",
                "title": f"/{top_skill['skill_name']} skill 생성",
                "detail": f"{top_skill['count']}회 반복된 '{top_skill['description']}' 패턴",
                "command": f"/cc-insights:skills generate {top_skill['skill_name']}",
                "impact": "높음"
            })

        # 2. 모호한 프롬프트 개선
        if vague:
            most_common_issue = Counter(v['issue'] for v in vague).most_common(1)
            if most_common_issue:
                issue, count = most_common_issue[0]
                actions.append({
                    "priority": 2,
                    "category": "품질 개선",
                    "title": f"'{issue}' 패턴 개선",
                    "detail": f"{count}회 발견 - 구체적인 대상과 기대 결과 명시 필요",
                    "command": None,
                    "impact": "중간"
                })

        # 3. 효율성 개선
        clear_count = sum(1 for e in self.entries if e.display.strip() == '/clear')
        if clear_count > 3:
            actions.append({
                "priority": 3,
                "category": "효율성",
                "title": "`claude --continue` 활용",
                "detail": f"/clear를 {clear_count}회 사용 중 - 컨텍스트 유지로 효율 증가",
                "command": "claude --continue",
                "impact": "중간"
            })

        # 4. 리포트 저장 권장
        if len(self.entries) > 50:
            actions.append({
                "priority": 4,
                "category": "기록",
                "title": "주간 리포트 저장",
                "detail": "분석 기록을 남겨 성장을 추적하세요",
                "command": "/cc-insights:analyze report",
                "impact": "낮음"
            })

        return sorted(actions, key=lambda x: x['priority'])

    def analyze_trends(self, periods: int = 4) -> dict:
        """시간에 따른 트렌드 분석 (주 단위)"""
        if not self.entries:
            return {}

        now = datetime.now()
        trends = []

        for i in range(periods):
            start = now - timedelta(days=(i + 1) * 7)
            end = now - timedelta(days=i * 7)

            period_entries = [e for e in self.entries if start <= e.timestamp < end]

            if period_entries:
                # 해당 기간 분석
                scores = []
                for e in period_entries:
                    if not e.display.startswith('/'):
                        score, _ = calculate_prompt_quality_score(e.display)
                        scores.append(score)

                avg_score = statistics.mean(scores) if scores else 0
                vague_count = sum(1 for e in period_entries if detect_vague_patterns(e.display))

                trends.append({
                    "period": f"{i + 1}주 전" if i > 0 else "이번 주",
                    "start_date": start.strftime("%m/%d"),
                    "end_date": end.strftime("%m/%d"),
                    "prompt_count": len(period_entries),
                    "avg_quality": round(avg_score, 1),
                    "vague_ratio": round(vague_count / max(1, len(period_entries)) * 100, 1)
                })

        # 트렌드 방향 계산
        if len(trends) >= 2:
            quality_trend = trends[0]['avg_quality'] - trends[1]['avg_quality']
            vague_trend = trends[1]['vague_ratio'] - trends[0]['vague_ratio']
            volume_trend = trends[0]['prompt_count'] - trends[1]['prompt_count']
        else:
            quality_trend = vague_trend = volume_trend = 0

        return {
            "weekly_data": trends,
            "quality_direction": "📈 상승" if quality_trend > 0.3 else "📉 하락" if quality_trend < -0.3 else "➡️ 유지",
            "vague_direction": "📈 개선" if vague_trend > 2 else "📉 악화" if vague_trend < -2 else "➡️ 유지",
            "volume_direction": "📈 증가" if volume_trend > 10 else "📉 감소" if volume_trend < -10 else "➡️ 유지"
        }

    def generate_personalized_tips(self) -> List[dict]:
        """개인화된 팁 생성"""
        tips = []
        profile = self.generate_user_profile()
        sw = self.analyze_strengths_weaknesses()

        # 작업 스타일 기반 팁
        if "나이트 아울" in profile.get("work_style", "") or "심야형" in profile.get("work_style", ""):
            tips.append({
                "category": "작업 패턴",
                "tip": "야간 작업 후에는 `/handoff`로 컨텍스트 정리를 권장합니다",
                "reason": "다음 날 작업 재개 시 시간 절약",
                "priority": "중간"
            })

        # 세션 스타일 기반 팁
        if "딥다이브" in profile.get("session_style", ""):
            tips.append({
                "category": "세션 관리",
                "tip": "긴 세션 중간에 `/compact`로 컨텍스트 압축을 권장합니다",
                "reason": "토큰 절약 및 응답 품질 유지",
                "priority": "높음"
            })

        # 프롬프트 스타일 기반 팁
        if "간결파" in profile.get("prompt_style", ""):
            tips.append({
                "category": "프롬프트 품질",
                "tip": "프롬프트에 '기대 결과'와 '제약 조건'을 추가해보세요",
                "reason": "더 정확한 응답을 받을 수 있습니다",
                "priority": "높음"
            })

        # 약점 기반 팁
        for weakness in sw.get("weaknesses", []):
            if "suggestion" in weakness:
                tips.append({
                    "category": weakness["area"],
                    "tip": weakness["suggestion"],
                    "reason": weakness["detail"],
                    "priority": "높음"
                })

        # 일반 팁
        tips.append({
            "category": "효율성",
            "tip": "자주 쓰는 명령어는 `skill`로 만들어 재사용하세요",
            "reason": "반복 입력 시간 절약",
            "priority": "낮음"
        })

        return tips[:7]  # 최대 7개

    def generate_onboarding_report(self) -> str:
        """온보딩(초기 설정) 상세 리포트"""
        profile = self.generate_user_profile()
        sw = self.analyze_strengths_weaknesses()
        actions = self.generate_action_items()
        automation = self.analyze_automation_candidates()
        quality = self.analyze_quality_scores()
        time_patterns = self.analyze_time_patterns()
        projects = self.analyze_project_patterns()

        total = len(self.entries)

        # 날짜 범위
        if self.entries:
            start_date = min(e.timestamp for e in self.entries).strftime("%Y-%m-%d")
            end_date = max(e.timestamp for e in self.entries).strftime("%Y-%m-%d")
            days_active = (max(e.timestamp for e in self.entries) - min(e.timestamp for e in self.entries)).days + 1
        else:
            start_date = end_date = "N/A"
            days_active = 0

        report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🔍 cc-insights 온보딩 분석 완료                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  📊 분석 기간: {start_date} ~ {end_date} ({days_active}일)
║  📝 총 프롬프트: {total:,}개
║  📈 하루 평균: {round(total / max(1, days_active), 1)}개
║  ⭐ 종합 등급: {sw.get('overall_grade', 'N/A')}
╚══════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 당신의 개발 프로필
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 작업 스타일: {profile.get('work_style', 'N/A')}
  • 세션 스타일: {profile.get('session_style', 'N/A')}
  • 프로젝트 스타일: {profile.get('project_style', 'N/A')}
  • 프롬프트 스타일: {profile.get('prompt_style', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💪 강점
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for s in sw.get('strengths', []):
            report += f"\n  {s['emoji']} {s['area']}: {s['detail']}"

        if not sw.get('strengths'):
            report += "\n  (아직 충분한 데이터가 없습니다)"

        report += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 개선 포인트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for w in sw.get('weaknesses', []):
            report += f"\n  {w['emoji']} {w['area']}: {w['detail']}"
            report += f"\n     💡 제안: {w.get('suggestion', '')}"

        if not sw.get('weaknesses'):
            report += "\n  ✅ 특별한 개선 포인트가 없습니다! 훌륭해요."

        report += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Skill 자동화 추천
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        if automation:
            for i, a in enumerate(automation[:5], 1):
                confidence_bar = "█" * int(a['confidence'] * 10) + "░" * (10 - int(a['confidence'] * 10))
                report += f"\n  {i}. /{a['skill_name']} ({a['count']}회)"
                report += f"\n     {a['description']}"
                report += f"\n     신뢰도: [{confidence_bar}] {round(a['confidence'] * 100)}%"
        else:
            report += "\n  (반복 패턴이 감지되지 않았습니다)"

        report += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 지금 바로 할 수 있는 것들
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for action in actions[:3]:
            report += f"\n  [{action['priority']}] {action['title']}"
            report += f"\n      {action['detail']}"
            if action.get('command'):
                report += f"\n      → 실행: {action['command']}"

        report += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 주요 프로젝트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        top_projects = list(projects.items())[:5]
        for proj, stats in top_projects:
            bar_len = min(20, int(stats['count'] / max(1, total) * 100))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            report += f"\n  {proj}: [{bar}] {stats['count']}개"

        report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ 작업 시간 패턴
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 피크 시간: {', '.join(f"{h[0]}시 ({h[1]}회)" for h in time_patterns.get('peak_hours', [])[:3])}
  • 야간 작업: {time_patterns.get('night_work_ratio', 0)}%
  • {time_patterns.get('recommendation', '')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 다음 단계
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. /cc-insights:summary  - 빠른 현황 확인 (매일)
  2. /cc-insights:tips     - 개인화된 팁 확인
  3. /cc-insights:trends   - 주간 트렌드 확인
  4. /cc-insights:skills   - skill 목록 및 생성

✨ baseline이 저장되었습니다. 앞으로의 분석은 이 기준과 비교됩니다.
"""
        return report

    def generate_report(self, format: str = "markdown") -> str:
        """분석 리포트 생성"""
        vague = self.analyze_vague_prompts()
        quality = self.analyze_quality_scores()
        automation = self.analyze_automation_candidates()
        inefficiencies = self.analyze_inefficiencies()
        sessions = self.analyze_session_management()
        time_patterns = self.analyze_time_patterns()
        projects = self.analyze_project_patterns()

        if format == "json":
            return json.dumps({
                "vague_prompts": vague[:20],
                "quality_scores": quality,
                "automation_candidates": automation,
                "inefficiencies": inefficiencies,
                "session_management": sessions,
                "time_patterns": time_patterns,
                "project_patterns": projects
            }, ensure_ascii=False, indent=2, default=str)

        # Markdown 리포트
        return self._format_markdown_report(
            vague, quality, automation, inefficiencies, sessions, time_patterns, projects
        )

    def _format_markdown_report(self, vague, quality, automation, inefficiencies, sessions, time_patterns, projects) -> str:
        """마크다운 리포트 포맷"""
        now = datetime.now()
        total = len(self.entries)

        report = f"""# cc-insights 분석 리포트

**생성일시**: {now.strftime('%Y-%m-%d %H:%M')}
**분석 기간**: 최근 {self.days}일
**총 프롬프트**: {total}개

---

## 1. 프롬프트 품질 분석

**평균 점수**: {quality.get('avg_score', 0)}/10
**저품질 프롬프트**: {quality.get('low_quality_count', 0)}개 ({round(quality.get('low_quality_count', 0) / max(1, total) * 100, 1)}%)

### 개선이 필요한 프롬프트 (상위 5개)

| 프롬프트 | 점수 | 문제점 | 개선안 |
|----------|------|--------|--------|
"""
        for item in quality.get('low_quality', [])[:5]:
            prompt_short = item['prompt'][:30] + '...' if len(item['prompt']) > 30 else item['prompt']
            issues = ', '.join(item['issues']) if item['issues'] else '-'
            report += f"| `{prompt_short}` | {item['score']}/10 | {issues} | {item['improvement'][:40]}... |\n"

        report += f"""
## 2. 반복 패턴 → Skill 제안

"""
        if automation:
            for candidate in automation[:5]:
                report += f"""### `/{candidate['skill_name']}` ({candidate['count']}회)
- **설명**: {candidate['description']}
- **신뢰도**: {round(candidate['confidence'] * 100)}%
- **샘플**:
"""
                for sample in candidate['samples'][:3]:
                    sample_short = sample[:50] + '...' if len(sample) > 50 else sample
                    report += f"  - `{sample_short}`\n"
                report += "\n"
        else:
            report += "*반복 패턴이 감지되지 않았습니다.*\n\n"

        report += """## 3. 비효율 패턴

"""
        if inefficiencies:
            for ineff in inefficiencies:
                report += f"""### {ineff['pattern']} ({ineff['count']}회)
- **개선안**: {ineff['suggestion']}
"""
        else:
            report += "*비효율 패턴이 감지되지 않았습니다.*\n"

        if sessions.get('multi_session_projects'):
            report += f"""
## 4. 세션 관리

### 동시 세션이 많은 프로젝트
"""
            for proj, count in sessions['multi_session_projects'].items():
                proj_name = Path(proj).name if proj else proj
                report += f"- **{proj_name}**: {count}개 세션 → `/rename`으로 구분 권장\n"

        if sessions.get('clear_count', 0) > 3:
            report += f"""
### /clear 사용 빈도
- {sessions['clear_count']}회 → `claude --continue` 사용 시 컨텍스트 유지 가능
"""

        report += f"""
## 5. 시간 패턴

**피크 시간**: {', '.join(f"{h[0]}시 ({h[1]}회)" for h in time_patterns.get('peak_hours', [])[:3])}
**피크 요일**: {', '.join(f"{d[0]} ({d[1]}회)" for d in time_patterns.get('peak_days', [])[:3])}
**야간 작업**: {time_patterns.get('night_work_ratio', 0)}%

**권장사항**: {time_patterns.get('recommendation', '-')}

## 6. 프로젝트별 활동

| 프로젝트 | 프롬프트 수 | 세션 수 |
|----------|-------------|---------|
"""
        for proj, stats in list(projects.items())[:10]:
            report += f"| {proj} | {stats['count']} | {stats['sessions']} |\n"

        report += """
---

*이 리포트는 `~/.claude/history.jsonl` 분석 결과입니다.*
*cc-insights Plugin으로 생성됨*
"""
        return report

    def generate_quick_summary(self) -> str:
        """빠른 요약 (매일 확인용)"""
        total = len(self.entries)
        quality = self.analyze_quality_scores()
        vague = self.analyze_vague_prompts()
        automation = self.analyze_automation_candidates()

        # 오늘 프롬프트
        today = datetime.now().date()
        today_count = sum(1 for e in self.entries if e.timestamp.date() == today)

        # baseline 비교
        baseline_info = ""
        if self.baseline:
            baseline_avg = self.baseline.get('avg_prompts_per_day', 0)
            if baseline_avg > 0:
                change = round((today_count - baseline_avg) / baseline_avg * 100, 1)
                arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                baseline_info = f" ({arrow} 평균 대비 {change:+}%)"

        summary = f"""
┌─────────────────────────────────────────────────────────────┐
│  📊 cc-insights 빠른 요약 (최근 {self.days}일)               │
├─────────────────────────────────────────────────────────────┤
│  오늘: {today_count}개{baseline_info}
│  총 프롬프트: {total}개 | 품질: {quality.get('avg_score', 0)}/10
│  모호한 프롬프트: {len(vague)}개 ({round(len(vague) / max(1, total) * 100, 1)}%)
└─────────────────────────────────────────────────────────────┘
"""

        # Top 액션
        if automation:
            summary += f"\n💡 추천: /{automation[0]['skill_name']} skill 생성 ({automation[0]['count']}회 반복)"

        return summary

    def generate_tips_output(self) -> str:
        """개인화된 팁 출력"""
        tips = self.generate_personalized_tips()

        output = """
╔══════════════════════════════════════════════════════════════╗
║  💡 개인화된 팁                                               ║
╚══════════════════════════════════════════════════════════════╝
"""

        priority_emoji = {"높음": "🔴", "중간": "🟡", "낮음": "🟢"}

        for i, tip in enumerate(tips, 1):
            emoji = priority_emoji.get(tip['priority'], "⚪")
            output += f"""
  {emoji} [{tip['category']}]
     {tip['tip']}
     └─ 이유: {tip['reason']}
"""

        output += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        output += "\n  /cc-insights:analyze 로 전체 분석을 확인하세요."

        return output

    def generate_trends_output(self, weeks: int = 4) -> str:
        """트렌드 분석 출력"""
        # 더 많은 데이터 로드
        self.cutoff = datetime.now() - timedelta(days=weeks * 7)
        self.entries = []
        self.load_history()

        trends = self.analyze_trends(periods=weeks)

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║  📈 최근 {weeks}주 트렌드                                      ║
╚══════════════════════════════════════════════════════════════╝

  품질 점수: {trends.get('quality_direction', 'N/A')}
  모호한 프롬프트: {trends.get('vague_direction', 'N/A')}
  사용량: {trends.get('volume_direction', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  주간 세부 데이터
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        for week in trends.get('weekly_data', []):
            bar_len = min(20, week['prompt_count'] // 10)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            output += f"""
  {week['period']} ({week['start_date']} ~ {week['end_date']})
    프롬프트: [{bar}] {week['prompt_count']}개
    품질: {week['avg_quality']}/10 | 모호: {week['vague_ratio']}%
"""

        return output

    def generate_skills_output(self) -> str:
        """skill 추천 목록 출력"""
        automation = self.analyze_automation_candidates()

        output = """
╔══════════════════════════════════════════════════════════════╗
║  🤖 Skill 자동화 추천                                         ║
╚══════════════════════════════════════════════════════════════╝
"""

        if not automation:
            output += "\n  반복 패턴이 감지되지 않았습니다."
            output += "\n  (3회 이상 반복된 패턴이 있어야 추천됩니다)"
            return output

        for i, skill in enumerate(automation, 1):
            confidence_bar = "█" * int(skill['confidence'] * 10) + "░" * (10 - int(skill['confidence'] * 10))
            output += f"""
  {i}. /{skill['skill_name']} - {skill['description']}
     반복 횟수: {skill['count']}회
     신뢰도: [{confidence_bar}] {round(skill['confidence'] * 100)}%
     샘플:
"""
            for sample in skill['samples'][:2]:
                short = sample[:50] + '...' if len(sample) > 50 else sample
                output += f"       • \"{short}\"\n"

        output += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  skill 생성: /cc-insights:skills generate <skill_name>
"""

        return output

    def generate_profile_output(self) -> str:
        """사용자 프로필 출력"""
        profile = self.generate_user_profile()
        sw = self.analyze_strengths_weaknesses()

        output = f"""
╔══════════════════════════════════════════════════════════════╗
║  👤 나의 개발 프로필                                          ║
╠══════════════════════════════════════════════════════════════╣
║  종합 등급: {sw.get('overall_grade', 'N/A')}
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  개발 스타일
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 작업 시간: {profile.get('work_style', 'N/A')}
  • 세션 패턴: {profile.get('session_style', 'N/A')}
  • 프로젝트: {profile.get('project_style', 'N/A')}
  • 프롬프트: {profile.get('prompt_style', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  세부 지표
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 세션당 평균 프롬프트: {profile.get('metrics', {}).get('avg_prompts_per_session', 0)}개
  • 평균 프롬프트 길이: {profile.get('metrics', {}).get('avg_prompt_length', 0)}자
  • 총 세션 수: {profile.get('metrics', {}).get('total_sessions', 0)}개
  • 프로젝트 수: {profile.get('metrics', {}).get('total_projects', 0)}개

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  💪 강점
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for s in sw.get('strengths', []):
            output += f"\n  {s['emoji']} {s['area']}: {s['detail']}"

        if not sw.get('strengths'):
            output += "\n  (강점 분석을 위한 데이터가 부족합니다)"

        output += """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📌 개선 필요
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        for w in sw.get('weaknesses', []):
            output += f"\n  {w['emoji']} {w['area']}: {w['detail']}"

        if not sw.get('weaknesses'):
            output += "\n  ✅ 훌륭합니다! 특별한 개선점이 없습니다."

        return output

    def generate_summary(self) -> str:
        """간단 요약 (CLI 출력용)"""
        total = len(self.entries)
        vague = self.analyze_vague_prompts()
        automation = self.analyze_automation_candidates()
        quality = self.analyze_quality_scores()

        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  cc-insights 분석 결과 (최근 {self.days}일)
╠══════════════════════════════════════════════════════════════╣
║  총 프롬프트: {total}개
║  평균 품질 점수: {quality.get('avg_score', 0)}/10
║  모호한 프롬프트: {len(vague)}개 ({round(len(vague) / max(1, total) * 100, 1)}%)
║  자동화 후보: {len(automation)}개 패턴 감지
╚══════════════════════════════════════════════════════════════╝
"""

        if automation:
            summary += "\n[Skill 제안]\n"
            for c in automation[:3]:
                summary += f"  → /{c['skill_name']}: {c['count']}회 반복 ({c['description']})\n"

        if vague:
            summary += "\n[개선 필요 프롬프트 예시]\n"
            seen = set()
            for v in vague[:3]:
                if v['prompt'] not in seen:
                    seen.add(v['prompt'])
                    short = v['prompt'][:40] + '...' if len(v['prompt']) > 40 else v['prompt']
                    summary += f"  • \"{short}\" → {v['suggestion'][:50]}\n"

        return summary


def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(description="cc-insights: Claude Code 사용 패턴 분석")
    parser.add_argument("command", nargs="?", default="analyze",
                        choices=["analyze", "summary", "tips", "trends", "skills", "profile", "projects", "stats"],
                        help="실행할 명령")
    parser.add_argument("--days", type=int, default=7, help="분석 기간 (일)")
    parser.add_argument("--format", choices=["markdown", "json", "summary"], default="summary")
    parser.add_argument("--output", type=str, help="출력 파일 경로")
    parser.add_argument("--setup", action="store_true", help="초기 설정 (baseline 생성)")
    parser.add_argument("--weeks", type=int, default=4, help="트렌드 분석 주 수")
    parser.add_argument("--project", type=str, help="특정 프로젝트 상세 분석")

    args = parser.parse_args()

    analyzer = CCInsightsAnalyzer(days=args.days)

    # 초기 설정 (온보딩)
    if args.setup:
        print("[cc-insights] 🔍 전체 히스토리 스캔 중...")
        count = analyzer.load_all_history()
        print(f"[cc-insights] ✅ {count:,}개 프롬프트 로드 완료")

        baseline = analyzer.generate_baseline()
        analyzer.save_baseline(baseline)
        print(f"[cc-insights] 💾 baseline 저장 완료")

        # 강화된 온보딩 리포트 출력
        print(analyzer.generate_onboarding_report())
        return

    # 데이터 로드
    count = analyzer.load_history()
    if count == 0:
        print("[오류] 분석할 데이터가 없습니다. 먼저 Claude Code를 사용해주세요.", file=sys.stderr)
        sys.exit(1)

    # baseline 로드
    analyzer.load_baseline()

    # 명령별 처리
    if args.command == "summary":
        # 빠른 요약
        print(analyzer.generate_quick_summary())

    elif args.command == "tips":
        # 개인화된 팁
        print(analyzer.generate_tips_output())

    elif args.command == "trends":
        # 트렌드 분석
        print(analyzer.generate_trends_output(weeks=args.weeks))

    elif args.command == "skills":
        # skill 목록
        print(analyzer.generate_skills_output())

    elif args.command == "profile":
        # 사용자 프로필
        print(analyzer.generate_profile_output())

    elif args.command == "projects":
        # 프로젝트별 분석
        if args.project:
            print(analyzer.generate_project_detail_output(args.project))
        else:
            print(analyzer.generate_projects_output())

    elif args.command == "stats":
        # 전체 사용 통계 (전체 히스토리 로드)
        print("[cc-insights] 전체 히스토리 로드 중...")
        analyzer.entries = []
        count = analyzer.load_all_history()
        print(f"[cc-insights] {count:,}개 프롬프트 분석 중...")
        print(analyzer.generate_stats_output())

    else:  # analyze (기본)
        print(f"[cc-insights] {count}개 프롬프트 분석 중...")

        if args.format == "summary":
            print(analyzer.generate_summary())
        else:
            report = analyzer.generate_report(format=args.format)

            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"[cc-insights] 리포트 저장: {output_path}")
            else:
                print(report)


if __name__ == "__main__":
    main()
