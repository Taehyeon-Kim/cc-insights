#!/usr/bin/env python3
"""
cc-insights 트랜스크립트 파서
~/.claude/transcripts/ JSONL 파일을 파싱하여 도구 사용 패턴, 파일 활동,
에러 패턴, 워크플로우 패턴 등을 분석.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 설정
TRANSCRIPTS_DIR = Path.home() / ".claude" / "transcripts"
PLUGIN_DATA_PATH = Path(__file__).parent.parent / "data"
TRANSCRIPT_INDEX_PATH = PLUGIN_DATA_PATH / "transcript_index.json"

HOME = str(Path.home())

# analyzer.py의 _SENSITIVE_PATTERNS 재사용
_SENSITIVE_PATTERNS = [
    re.compile(
        r"(?:api[_-]?key|token|secret|password|passwd|credential|auth)[=:\s]+\S+",
        re.IGNORECASE,
    ),
    re.compile(r"(?:sk|pk|ghp|gho|glpat|xox[bpas])-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{20,}\.eyJ[A-Za-z0-9_\-]{20,}"),  # JWT
    re.compile(r"https?://[^@\s]*:[^@\s]*@"),  # URL with credentials
]


# ============================================================
# 데이터클래스
# ============================================================


@dataclass
class ToolCall:
    tool_name: str
    timestamp: datetime
    duration_ms: float
    tool_input: dict
    success: bool
    error_info: Optional[str]


@dataclass
class TranscriptSession:
    session_id: str
    file_path: str
    file_size: int
    mtime: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    user_message_count: int = 0
    assistant_message_count: int = 0
    tool_call_count: int = 0
    tool_counts: Dict[str, int] = field(default_factory=dict)
    files_read: List[str] = field(default_factory=list)
    files_edited: List[str] = field(default_factory=list)
    files_written: List[str] = field(default_factory=list)
    bash_commands: List[str] = field(default_factory=list)
    error_count: int = 0
    errors: List[dict] = field(default_factory=list)
    inferred_project: Optional[str] = None


# ============================================================
# 유틸리티
# ============================================================


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """ISO 8601 타임스탬프 파싱"""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _sanitize_path(path: str) -> str:
    """절대 경로 → ~ 상대 경로로 변환"""
    if path.startswith(HOME):
        return "~" + path[len(HOME) :]
    return path


def _sanitize_command(cmd: str) -> str:
    """민감 정보 마스킹"""
    sanitized = cmd
    for pattern in _SENSITIVE_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    # 첫 줄만 (멀티라인 명령은 축약)
    first_line = sanitized.split("\n")[0]
    if len(first_line) > 120:
        first_line = first_line[:120] + "..."
    return first_line


def _get_file_path(tool_input: dict) -> Optional[str]:
    """tool_input에서 파일 경로 추출 (camelCase와 snake_case 모두 지원)"""
    return (
        tool_input.get("filePath")
        or tool_input.get("file_path")
        or tool_input.get("path")
    )


def _safe_write(path: Path, content: str, mode: int = 0o600) -> None:
    """안전한 파일 쓰기 (제한된 권한)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.write(fd, content.encode("utf-8"))
    finally:
        os.close(fd)


# ============================================================
# 핵심 파싱
# ============================================================


def _detect_error(tool_name: str, tool_output: dict) -> Tuple[bool, Optional[str]]:
    """도구 결과에서 에러 감지"""
    if not tool_output:
        return False, None

    # bash: exit code != 0
    if tool_name == "bash":
        exit_code = tool_output.get("exit")
        if exit_code is not None and exit_code != 0:
            output_text = tool_output.get("output", "")[:200]
            return True, f"exit={exit_code}: {output_text}"

    # 일반 에러 문자열 감지
    output_str = str(tool_output.get("output", "") or tool_output.get("preview", ""))
    if output_str:
        error_patterns = ["Error:", "Traceback", "ENOENT", "Permission denied", "fatal:"]
        for ep in error_patterns:
            if ep in output_str[:500]:
                return True, output_str[:200]

    return False, None


def _pair_tool_calls(records: list) -> List[ToolCall]:
    """tool_use → tool_result 순차 매칭. FIFO 큐로 병렬 호출 처리."""
    pending: Dict[str, list] = defaultdict(list)  # tool_name → [tool_use records]
    paired: List[ToolCall] = []

    for record in records:
        rtype = record.get("type")

        if rtype == "tool_use":
            pending[record.get("tool_name", "")].append(record)

        elif rtype == "tool_result":
            tool_name = record.get("tool_name", "")
            queue = pending.get(tool_name, [])

            if queue:
                use_record = queue.pop(0)
                use_ts = _parse_timestamp(use_record.get("timestamp", ""))
                result_ts = _parse_timestamp(record.get("timestamp", ""))

                duration_ms = 0.0
                if use_ts and result_ts:
                    duration_ms = (result_ts - use_ts).total_seconds() * 1000

                tool_output = record.get("tool_output", {})
                is_error, error_info = _detect_error(tool_name, tool_output)

                # tool_input 요약본 (경로, 커맨드 등 핵심만)
                raw_input = use_record.get("tool_input", {})
                summarized_input = _summarize_tool_input(tool_name, raw_input)

                paired.append(
                    ToolCall(
                        tool_name=tool_name,
                        timestamp=use_ts or datetime.now(timezone.utc),
                        duration_ms=duration_ms,
                        tool_input=summarized_input,
                        success=not is_error,
                        error_info=error_info,
                    )
                )

    # 매칭되지 않은 tool_use 처리 (결과 없는 호출)
    for tool_name, remaining in pending.items():
        for use_record in remaining:
            use_ts = _parse_timestamp(use_record.get("timestamp", ""))
            raw_input = use_record.get("tool_input", {})
            paired.append(
                ToolCall(
                    tool_name=tool_name,
                    timestamp=use_ts or datetime.now(timezone.utc),
                    duration_ms=0.0,
                    tool_input=_summarize_tool_input(tool_name, raw_input),
                    success=True,
                    error_info=None,
                )
            )

    return paired


def _summarize_tool_input(tool_name: str, tool_input: dict) -> dict:
    """tool_input을 핵심 필드만 남긴 요약본으로 변환"""
    if tool_name == "bash":
        cmd = tool_input.get("command", "")
        return {"command": _sanitize_command(cmd)}
    elif tool_name in ("read", "write", "edit"):
        fp = _get_file_path(tool_input)
        return {"filePath": _sanitize_path(fp) if fp else ""}
    elif tool_name == "glob":
        return {
            "pattern": tool_input.get("pattern", ""),
            "path": _sanitize_path(tool_input.get("path", "")) if tool_input.get("path") else "",
        }
    elif tool_name == "grep":
        return {
            "pattern": tool_input.get("pattern", ""),
            "path": _sanitize_path(tool_input.get("path", "")) if tool_input.get("path") else "",
        }
    elif tool_name == "task":
        return {
            "subagent_type": tool_input.get("subagent_type", ""),
            "description": tool_input.get("description", ""),
        }
    else:
        # 기타 도구: 키만 보존
        return {k: "..." for k in list(tool_input.keys())[:5]}


def _infer_project(tool_calls: List[ToolCall]) -> Optional[str]:
    """read/edit/write의 filePath에서 프로젝트 루트 추론"""
    paths = []
    for tc in tool_calls:
        if tc.tool_name in ("read", "write", "edit"):
            fp = tc.tool_input.get("filePath", "")
            # ~로 시작하는 경로 복원
            if fp.startswith("~"):
                fp = HOME + fp[1:]
            if fp and os.path.isabs(fp):
                paths.append(fp)

    if not paths:
        return None

    try:
        common = os.path.commonpath(paths)
    except ValueError:
        return None

    # 홈 디렉토리 자체이면 프로젝트 추론 불가
    if common == HOME or common == "/":
        return None

    # .claude 내부 경로도 제외
    if "/.claude/" in common:
        return None

    # 프로젝트명: 마지막 의미 있는 디렉토리
    parts = Path(common).parts
    # ~/Documents/develop/project_name → project_name
    return parts[-1] if parts else None


def parse_transcript(file_path: Path) -> TranscriptSession:
    """단일 트랜스크립트 파일을 스트리밍 파싱"""
    session_id = file_path.stem  # ses_XXXXX
    stat = file_path.stat()

    records = []
    user_count = 0
    assistant_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
                rtype = record.get("type", "")
                if rtype == "user":
                    user_count += 1
                elif rtype == "assistant":
                    assistant_count += 1
            except json.JSONDecodeError:
                continue

    # 타임스탬프 추출
    timestamps = []
    for r in records:
        ts = _parse_timestamp(r.get("timestamp", ""))
        if ts:
            timestamps.append(ts)

    start_time = min(timestamps) if timestamps else None
    end_time = max(timestamps) if timestamps else None
    duration = (end_time - start_time).total_seconds() if start_time and end_time else 0.0

    # 도구 호출 페어링
    tool_calls = _pair_tool_calls(records)

    # 도구별 카운트
    tool_counts = Counter(tc.tool_name for tc in tool_calls)

    # 파일 활동 추출
    files_read = []
    files_edited = []
    files_written = []
    bash_commands = []

    for tc in tool_calls:
        fp = tc.tool_input.get("filePath", "")
        if tc.tool_name == "read" and fp:
            files_read.append(fp)
        elif tc.tool_name == "edit" and fp:
            files_edited.append(fp)
        elif tc.tool_name == "write" and fp:
            files_written.append(fp)
        elif tc.tool_name == "bash":
            cmd = tc.tool_input.get("command", "")
            if cmd:
                bash_commands.append(cmd)

    # 에러 수집
    errors = []
    for tc in tool_calls:
        if not tc.success and tc.error_info:
            errors.append(
                {
                    "tool_name": tc.tool_name,
                    "error_info": tc.error_info[:200],
                    "timestamp": tc.timestamp.isoformat() if tc.timestamp else "",
                }
            )

    # 프로젝트 추론
    inferred_project = _infer_project(tool_calls)

    return TranscriptSession(
        session_id=session_id,
        file_path=str(file_path),
        file_size=stat.st_size,
        mtime=stat.st_mtime,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration,
        user_message_count=user_count,
        assistant_message_count=assistant_count,
        tool_call_count=len(tool_calls),
        tool_counts=dict(tool_counts),
        files_read=list(set(files_read)),
        files_edited=list(set(files_edited)),
        files_written=list(set(files_written)),
        bash_commands=bash_commands,
        error_count=len(errors),
        errors=errors,
        inferred_project=inferred_project,
    )


# ============================================================
# 인덱스/캐시
# ============================================================


def _session_to_index_entry(session: TranscriptSession) -> dict:
    """세션을 인덱스 엔트리로 변환 (인덱스에는 요약 정보만)"""
    return {
        "session_id": session.session_id,
        "file_path": session.file_path,
        "file_size": session.file_size,
        "mtime": session.mtime,
        "start_time": session.start_time.isoformat() if session.start_time else None,
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "duration_seconds": session.duration_seconds,
        "user_message_count": session.user_message_count,
        "tool_call_count": session.tool_call_count,
        "tool_counts": session.tool_counts,
        "files_read": session.files_read,
        "files_edited": session.files_edited,
        "files_written": session.files_written,
        "bash_commands": session.bash_commands[:50],  # 최대 50개
        "error_count": session.error_count,
        "errors": session.errors[:20],  # 최대 20개
        "inferred_project": session.inferred_project,
    }


def load_index() -> Optional[dict]:
    """캐시된 인덱스 로드"""
    if not TRANSCRIPT_INDEX_PATH.exists():
        return None
    try:
        with open(TRANSCRIPT_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_index(index: dict) -> None:
    """인덱스 저장 (0600 권한)"""
    content = json.dumps(index, ensure_ascii=False, indent=2, default=str)
    _safe_write(TRANSCRIPT_INDEX_PATH, content)


def build_index(
    transcripts_dir: Path = TRANSCRIPTS_DIR,
    since: Optional[datetime] = None,
    progress: bool = True,
) -> dict:
    """트랜스크립트 인덱스 빌드. since 지정 시 해당 날짜 이후 파일만 풀 파싱."""
    existing_index = load_index()
    existing_sessions: Dict[str, dict] = {}
    last_generated = None

    if existing_index:
        last_generated_str = existing_index.get("generated_at")
        if last_generated_str:
            last_generated = _parse_timestamp(last_generated_str)
        for s in existing_index.get("sessions", []):
            existing_sessions[s["session_id"]] = s

    # 트랜스크립트 파일 목록
    if not transcripts_dir.exists():
        print(f"[오류] 트랜스크립트 디렉토리를 찾을 수 없습니다: {transcripts_dir}", file=sys.stderr)
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "total_sessions": 0, "sessions": []}

    files = sorted(transcripts_dir.glob("*.jsonl"))
    total = len(files)
    sessions = []
    parsed_count = 0
    cached_count = 0

    for i, fp in enumerate(files):
        session_id = fp.stem
        stat = fp.stat()

        # since 필터
        if since:
            file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if file_mtime < since:
                # since 이전 파일은 기존 인덱스에서 가져오기
                if session_id in existing_sessions:
                    sessions.append(existing_sessions[session_id])
                    cached_count += 1
                continue

        # 증분 업데이트: mtime이 같으면 캐시 사용
        if session_id in existing_sessions:
            cached_entry = existing_sessions[session_id]
            if abs(cached_entry.get("mtime", 0) - stat.st_mtime) < 1:
                sessions.append(cached_entry)
                cached_count += 1
                continue

        # 풀 파싱 필요
        if progress and (parsed_count % 50 == 0 or parsed_count == 0):
            print(f"  파싱 중... {i + 1}/{total} (캐시: {cached_count}, 신규: {parsed_count})", end="\r", file=sys.stderr)

        try:
            session = parse_transcript(fp)
            sessions.append(_session_to_index_entry(session))
            parsed_count += 1
        except Exception as e:
            print(f"\n  [경고] {fp.name} 파싱 실패: {e}", file=sys.stderr)
            continue

    if progress:
        print(f"  완료: {total}개 세션 (캐시: {cached_count}, 신규: {parsed_count})    ", file=sys.stderr)

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_sessions": len(sessions),
        "sessions": sessions,
    }

    save_index(index)
    return index


# ============================================================
# 분석 함수
# ============================================================


def _filter_sessions_by_days(sessions: list, days: int) -> list:
    """최근 N일 내의 세션만 필터링"""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for s in sessions:
        start_str = s.get("start_time")
        if start_str:
            start = _parse_timestamp(start_str)
            if start and start >= cutoff:
                filtered.append(s)
    return filtered


def analyze_tool_distribution(sessions: list, top_n: int = 15) -> dict:
    """도구별 사용 횟수, 세션당 평균, 전체 비율"""
    total_counts: Counter = Counter()
    total_calls = 0

    for s in sessions:
        for tool, count in s.get("tool_counts", {}).items():
            total_counts[tool] += count
            total_calls += count

    num_sessions = len(sessions)
    distribution = []

    for tool, count in total_counts.most_common(top_n):
        distribution.append(
            {
                "tool": tool,
                "count": count,
                "ratio": round(count / max(1, total_calls) * 100, 1),
                "avg_per_session": round(count / max(1, num_sessions), 1),
            }
        )

    return {
        "total_calls": total_calls,
        "total_sessions": num_sessions,
        "distribution": distribution,
    }


def analyze_file_activity(sessions: list, top_n: int = 20) -> dict:
    """가장 많이 수정된 파일, read/edit/write별 Top N"""
    read_counter: Counter = Counter()
    edit_counter: Counter = Counter()
    write_counter: Counter = Counter()

    for s in sessions:
        for f in s.get("files_read", []):
            read_counter[f] += 1
        for f in s.get("files_edited", []):
            edit_counter[f] += 1
        for f in s.get("files_written", []):
            write_counter[f] += 1

    return {
        "top_read": read_counter.most_common(top_n),
        "top_edited": edit_counter.most_common(top_n),
        "top_written": write_counter.most_common(top_n),
        "unique_files_read": len(read_counter),
        "unique_files_edited": len(edit_counter),
        "unique_files_written": len(write_counter),
    }


def analyze_error_patterns(sessions: list) -> dict:
    """에러 횟수, 에러율, 도구별 에러, 흔한 에러 유형"""
    total_errors = 0
    total_calls = 0
    tool_errors: Counter = Counter()
    error_types: Counter = Counter()

    for s in sessions:
        total_errors += s.get("error_count", 0)
        total_calls += s.get("tool_call_count", 0)

        for err in s.get("errors", []):
            tool_errors[err.get("tool_name", "unknown")] += 1
            # 에러 유형 추출
            info = err.get("error_info", "")
            if info.startswith("exit="):
                # exit=1: ... → "exit=1"
                exit_part = info.split(":")[0]
                error_types[f"bash ({exit_part})"] += 1
            elif "ENOENT" in info:
                error_types["파일 없음 (ENOENT)"] += 1
            elif "Permission denied" in info:
                error_types["권한 거부"] += 1
            elif "Traceback" in info:
                error_types["Python 예외"] += 1
            else:
                error_types["기타"] += 1

    error_rate = round(total_errors / max(1, total_calls) * 100, 1)

    return {
        "total_errors": total_errors,
        "total_calls": total_calls,
        "error_rate": error_rate,
        "by_tool": tool_errors.most_common(10),
        "by_type": error_types.most_common(10),
    }


def analyze_session_efficiency(sessions: list) -> dict:
    """평균 세션 시간, 도구 밀도(회/분), 에러율"""
    durations = []
    densities = []
    error_rates = []

    for s in sessions:
        dur = s.get("duration_seconds", 0)
        calls = s.get("tool_call_count", 0)
        errors = s.get("error_count", 0)

        if dur > 60:  # 1분 이상 세션만
            durations.append(dur)
            densities.append(calls / (dur / 60))  # 회/분
            if calls > 0:
                error_rates.append(errors / calls * 100)

    if not durations:
        return {"avg_duration_min": 0, "avg_density": 0, "avg_error_rate": 0}

    import statistics as stats

    return {
        "avg_duration_min": round(stats.mean(durations) / 60, 1),
        "median_duration_min": round(stats.median(durations) / 60, 1),
        "avg_density": round(stats.mean(densities), 1),
        "avg_error_rate": round(stats.mean(error_rates), 1) if error_rates else 0,
        "total_sessions_analyzed": len(durations),
    }


def analyze_workflow_patterns(sessions: list, min_freq: int = 3) -> list:
    """도구 호출 체인 패턴 (2-4gram)"""
    ngram_counts: Dict[int, Counter] = {2: Counter(), 3: Counter(), 4: Counter()}

    for s in sessions:
        # 세션의 도구 호출 순서 재구성 (tool_counts에서는 순서를 알 수 없으므로
        # bash_commands + files 활동으로 간접 추론하거나, 인덱스의 순서 정보 사용)
        # 간단 접근: tool_counts 키 순서 사용 (빈도 기반 패턴)
        # 실제로는 전체 파싱이 필요하지만, 인덱스 기반으로는 제한적
        # → 최근 세션만 풀 파싱하여 패턴 추출
        pass

    # 풀 파싱 기반 패턴 추출 (sessions에 tool_sequence가 있는 경우)
    # 여기서는 간단 버전: 파일에서 직접 추출
    return _extract_workflow_patterns_from_files(sessions, min_freq)


def _extract_workflow_patterns_from_files(sessions: list, min_freq: int) -> list:
    """세션 파일에서 직접 워크플로우 패턴 추출 (최근 세션만)"""
    ngram_counts: Dict[int, Counter] = {2: Counter(), 3: Counter()}
    max_sessions = 50  # 최근 50개 세션만

    recent = sorted(sessions, key=lambda s: s.get("start_time", ""), reverse=True)[:max_sessions]

    for s in recent:
        fp = Path(s.get("file_path", ""))
        if not fp.exists():
            continue

        tool_sequence = []
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("type") == "tool_use":
                            tool_sequence.append(record.get("tool_name", ""))
                    except json.JSONDecodeError:
                        continue
        except IOError:
            continue

        # n-gram 생성
        for n in (2, 3):
            for i in range(len(tool_sequence) - n + 1):
                gram = " → ".join(tool_sequence[i : i + n])
                ngram_counts[n][gram] += 1

    # 결과 정리
    patterns = []
    # 패턴 설명 매핑
    pattern_labels = {
        "read → edit": "파일 수정",
        "read → edit → bash": "코드 수정 후 실행",
        "glob → read": "파일 탐색 후 읽기",
        "glob → read → read": "파일 탐색",
        "bash → bash": "연속 명령 실행",
        "bash → bash → bash": "연속 명령",
        "read → read": "다중 파일 읽기",
        "read → read → read": "다중 파일 탐색",
        "grep → read": "검색 후 읽기",
        "grep → read → read": "검색 후 다중 읽기",
        "edit → bash": "수정 후 실행",
        "edit → edit": "연속 수정",
        "task → task": "연속 에이전트 위임",
    }

    for n in (2, 3):
        for gram, count in ngram_counts[n].most_common(20):
            if count >= min_freq:
                label = pattern_labels.get(gram, "")
                patterns.append({"pattern": gram, "count": count, "n": n, "label": label})

    # 빈도순 정렬
    patterns.sort(key=lambda x: x["count"], reverse=True)
    return patterns[:15]


def analyze_project_activity(sessions: list) -> dict:
    """프로젝트별 세션 수, 도구 사용, 에러 수"""
    project_stats: Dict[str, dict] = defaultdict(
        lambda: {"sessions": 0, "tool_calls": 0, "errors": 0, "duration_seconds": 0}
    )

    for s in sessions:
        proj = s.get("inferred_project") or "unknown"
        project_stats[proj]["sessions"] += 1
        project_stats[proj]["tool_calls"] += s.get("tool_call_count", 0)
        project_stats[proj]["errors"] += s.get("error_count", 0)
        project_stats[proj]["duration_seconds"] += s.get("duration_seconds", 0)

    # 세션 수 기준 정렬
    sorted_projects = sorted(project_stats.items(), key=lambda x: x[1]["sessions"], reverse=True)
    return dict(sorted_projects)


# ============================================================
# 출력 포맷팅
# ============================================================


def format_transcript_summary(
    sessions: list,
    days: int,
    subcmd: str = "summary",
) -> str:
    """종합 요약 포맷팅 — 기존 cc-insights 한국어 스타일"""
    num_sessions = len(sessions)
    total_calls = sum(s.get("tool_call_count", 0) for s in sessions)

    if subcmd == "tools":
        return _format_tools(sessions, days)
    elif subcmd == "files":
        return _format_files(sessions, days)
    elif subcmd == "errors":
        return _format_errors(sessions, days)
    elif subcmd == "efficiency":
        return _format_efficiency(sessions, days)
    elif subcmd == "workflows":
        return _format_workflows(sessions, days)
    else:  # summary
        return _format_full_summary(sessions, days)


def _format_full_summary(sessions: list, days: int) -> str:
    """종합 요약"""
    num_sessions = len(sessions)
    total_calls = sum(s.get("tool_call_count", 0) for s in sessions)

    tools = analyze_tool_distribution(sessions)
    files = analyze_file_activity(sessions)
    errors = analyze_error_patterns(sessions)
    efficiency = analyze_session_efficiency(sessions)
    workflows = analyze_workflow_patterns(sessions)
    projects = analyze_project_activity(sessions)

    output = f"""
cc-insights 트랜스크립트 분석 (최근 {days}일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  분석 대상: {num_sessions}개 세션 / {total_calls:,}개 도구 호출
"""

    # 도구 사용 분포 (Top 10)
    output += "\n  도구 사용 분포\n"
    max_count = tools["distribution"][0]["count"] if tools["distribution"] else 1
    for i, t in enumerate(tools["distribution"][:10], 1):
        bar_len = int(t["count"] / max_count * 20)
        bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
        output += f"  {i:2}. {t['tool']:<18} [{bar}] {t['count']:>5,}\ud68c ({t['ratio']}%)\n"

    # 자주 수정된 파일 (Top 5)
    if files["top_edited"]:
        output += "\n  자주 수정된 파일 (Top 5)\n"
        for i, (f, count) in enumerate(files["top_edited"][:5], 1):
            output += f"  {i}. {f:<45} {count:>2}\ud68c (edit)\n"

    # 에러 패턴
    output += f"\n  에러 패턴\n"
    output += f"  총 에러: {errors['total_errors']}개 (에러율: {errors['error_rate']}%)\n"
    for etype, count in errors["by_type"][:5]:
        output += f"  - {etype}: {count}건\n"

    # 워크플로우 패턴
    if workflows:
        output += "\n  워크플로우 패턴\n"
        for i, w in enumerate(workflows[:5], 1):
            label = f" {w['label']}" if w["label"] else ""
            output += f"  {i}. {w['pattern']:<35} ({w['count']:>2}\ud68c){label}\n"

    # 효율성
    output += f"\n  세션 효율성\n"
    output += f"  평균 세션: {efficiency['avg_duration_min']}분 | "
    output += f"도구 밀도: {efficiency['avg_density']}\ud68c/분 | "
    output += f"에러율: {efficiency['avg_error_rate']}%\n"

    # 프로젝트 (Top 5)
    proj_items = list(projects.items())[:5]
    if proj_items:
        output += "\n  프로젝트별 활동\n"
        for proj, stats in proj_items:
            output += f"  - {proj}: {stats['sessions']}세션 / {stats['tool_calls']:,}\ud68c\ud638\ucd9c / 에러 {stats['errors']}건\n"

    return output


def _format_tools(sessions: list, days: int) -> str:
    """도구 분포 상세"""
    tools = analyze_tool_distribution(sessions, top_n=20)
    num_sessions = len(sessions)

    output = f"""
cc-insights 도구 사용 분석 (최근 {days}일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {num_sessions}개 세션 / {tools['total_calls']:,}개 도구 호출

  도구별 상세
"""
    max_count = tools["distribution"][0]["count"] if tools["distribution"] else 1
    for i, t in enumerate(tools["distribution"], 1):
        bar_len = int(t["count"] / max_count * 25)
        bar = "\u2588" * bar_len + "\u2591" * (25 - bar_len)
        output += f"  {i:2}. {t['tool']:<18} [{bar}] {t['count']:>5,}\ud68c ({t['ratio']}%) | 세션당 {t['avg_per_session']}\ud68c\n"

    return output


def _format_files(sessions: list, days: int) -> str:
    """파일 활동 상세"""
    files = analyze_file_activity(sessions)

    output = f"""
cc-insights 파일 활동 분석 (최근 {days}일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  읽기: {files['unique_files_read']}개 고유 파일
  수정: {files['unique_files_edited']}개 고유 파일
  생성: {files['unique_files_written']}개 고유 파일
"""

    if files["top_edited"]:
        output += "\n  자주 수정된 파일 (edit)\n"
        for i, (f, count) in enumerate(files["top_edited"][:15], 1):
            output += f"  {i:2}. {f:<50} {count:>2}\ud68c\n"

    if files["top_read"]:
        output += "\n  자주 읽은 파일 (read)\n"
        for i, (f, count) in enumerate(files["top_read"][:10], 1):
            output += f"  {i:2}. {f:<50} {count:>2}\ud68c\n"

    if files["top_written"]:
        output += "\n  생성된 파일 (write)\n"
        for i, (f, count) in enumerate(files["top_written"][:10], 1):
            output += f"  {i:2}. {f:<50} {count:>2}\ud68c\n"

    return output


def _format_errors(sessions: list, days: int) -> str:
    """에러 패턴 상세"""
    errors = analyze_error_patterns(sessions)

    output = f"""
cc-insights 에러 분석 (최근 {days}일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  총 에러: {errors['total_errors']}개 / {errors['total_calls']:,}개 호출
  에러율: {errors['error_rate']}%

  도구별 에러
"""
    for tool, count in errors["by_tool"]:
        output += f"  - {tool}: {count}건\n"

    output += "\n  에러 유형\n"
    for etype, count in errors["by_type"]:
        output += f"  - {etype}: {count}건\n"

    return output


def _format_efficiency(sessions: list, days: int) -> str:
    """세션 효율성 상세"""
    eff = analyze_session_efficiency(sessions)

    output = f"""
cc-insights 효율성 분석 (최근 {days}일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  분석 세션: {eff['total_sessions_analyzed']}개 (1분 이상 세션)

  평균 세션 시간: {eff['avg_duration_min']}분
  중앙값 세션 시간: {eff['median_duration_min']}분
  평균 도구 밀도: {eff['avg_density']}\ud68c/분
  평균 에러율: {eff['avg_error_rate']}%
"""
    return output


def _format_workflows(sessions: list, days: int) -> str:
    """워크플로우 패턴 상세"""
    workflows = analyze_workflow_patterns(sessions)

    output = f"""
cc-insights 워크플로우 분석 (최근 {days}일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    if not workflows:
        output += "  워크플로우 패턴이 감지되지 않았습니다.\n"
        return output

    output += "  도구 호출 체인 패턴 (빈도순)\n\n"
    for i, w in enumerate(workflows, 1):
        label = f"  {w['label']}" if w["label"] else ""
        output += f"  {i:2}. {w['pattern']:<35} ({w['count']:>3}\ud68c){label}\n"

    return output


# ============================================================
# CLI 엔트리포인트 (독립 실행용)
# ============================================================


def run_transcript_analysis(days: int = 7, subcmd: str = "summary") -> None:
    """트랜스크립트 분석 실행 (analyzer.py에서 호출)"""
    print(f"[cc-insights] 트랜스크립트 인덱스 빌드 중...", file=sys.stderr)

    since = datetime.now(timezone.utc) - timedelta(days=days + 1)  # 약간의 여유
    index = build_index(since=since)

    sessions = _filter_sessions_by_days(index.get("sessions", []), days)

    if not sessions:
        print(f"\n  최근 {days}일 내의 트랜스크립트가 없습니다.\n")
        return

    output = format_transcript_summary(sessions, days, subcmd)
    print(output)


def main():
    """독립 실행용 CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="cc-insights 트랜스크립트 파서")
    parser.add_argument("--build-index", action="store_true", help="인덱스 빌드")
    parser.add_argument("--parse", type=str, help="단일 세션 파싱 (파일 경로)")
    parser.add_argument("--days", type=int, default=7, help="분석 기간 (일)")
    parser.add_argument(
        "--sub",
        choices=["summary", "tools", "files", "errors", "efficiency", "workflows"],
        default="summary",
        help="서브커맨드",
    )

    args = parser.parse_args()

    if args.build_index:
        print("[cc-insights] 전체 인덱스 빌드 중...")
        index = build_index()
        print(f"[cc-insights] 인덱스 저장 완료: {TRANSCRIPT_INDEX_PATH}")
        print(f"  총 세션: {index['total_sessions']}개")
        return

    if args.parse:
        fp = Path(args.parse)
        if not fp.exists():
            print(f"[오류] 파일을 찾을 수 없습니다: {fp}", file=sys.stderr)
            sys.exit(1)
        session = parse_transcript(fp)
        print(json.dumps(asdict(session), ensure_ascii=False, indent=2, default=str))
        return

    # 기본: 분석 실행
    run_transcript_analysis(days=args.days, subcmd=args.sub)


if __name__ == "__main__":
    main()
