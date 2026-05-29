import json
import shlex
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from pier.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from pier.agents.network import allowlist_from_urls
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.name import AgentName
from pier.models.agent.network import NetworkAllowlist
from pier.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from pier.models.trial.paths import EnvironmentPaths
from pier.utils.trajectory_metrics import (
    extra_with_context_metrics,
    peak_context_tokens_from_steps,
    populate_context_from_final_metrics,
)
from pier.utils.trajectory_utils import format_trajectory_json


class Pi(BaseInstalledAgent):
    """
    The Pi installed agent runs the official Pi coding-agent CLI in print mode.

    Assumptions based on Pi's public docs/source:
    - Install uses ``npm install -g --ignore-scripts @earendil-works/pi-coding-agent``
      with an optional npm version suffix from Pier's ``version`` field.
    - Invocation uses ``pi --mode json --no-session`` so Pi runs once, prints a
      JSON event stream, and does not persist a session.
    - Pier sets ``PI_CODING_AGENT_DIR`` to an isolated directory inside the task
      environment, writes any generated ``models.json`` there, and disables
      startup update/package network checks with ``PI_OFFLINE=1``.
    - If ``base_url`` is set, or ``OPENAI_BASE_URL`` is present for back-compat,
      Pier writes a minimal provider entry to Pi's ``models.json``. This is the
      same interface for any OpenAI-compatible provider name, including local
      servers such as Gemma served at ``/v1``.

    Pi does not emit Pier ATIF directly. This adapter captures stdout JSONL and
    stderr logs, then converts Pi's documented JSON event stream into a minimal
    ATIF trajectory with assistant messages, tool calls, observations, and usage
    metrics when present.
    """

    SUPPORTS_ATIF: bool = True

    _OUTPUT_FILENAME = "pi.jsonl"
    _STDERR_FILENAME = "pi.stderr"
    _REMOTE_PI_AGENT_DIR = PurePosixPath("/tmp/pi-agent")
    _REMOTE_INSTRUCTION_PATH = PurePosixPath("/tmp/pi-instruction.md")

    CLI_FLAGS = [
        CliFlag(
            "thinking",
            cli="--thinking",
            type="enum",
            choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        ),
    ]

    def __init__(
        self,
        *args,
        command_model_name: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        api: str = "openai-completions",
        compat: dict[str, Any] | None = None,
        context_window: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        self._command_model_name = command_model_name
        self._provider = provider
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._api = api
        self._compat = (
            dict(compat)
            if compat is not None
            else {
                "supportsUsageInStreaming": False,
                "maxTokensField": "max_tokens",
            }
        )
        self._context_window = context_window
        self._max_tokens = max_tokens

        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return AgentName.PI.value

    def get_version_command(self) -> str | None:
        return "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; pi --version"

    def install_spec(self) -> AgentInstallSpec:
        version_spec = f"@{self._version}" if self._version else "@latest"
        root_run = (
            "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
            "  apk add --no-cache bash curl nodejs npm;"
            " elif command -v apt-get &>/dev/null; then"
            "  apt-get update && apt-get install -y curl;"
            " elif command -v yum &>/dev/null; then"
            "  yum install -y curl;"
            " else"
            '  echo "Warning: No known package manager found, assuming curl is available" >&2;'
            " fi"
        )
        agent_run = (
            "set -euo pipefail; "
            "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then"
            f"  npm install -g --ignore-scripts @earendil-works/pi-coding-agent{version_spec};"
            " else"
            "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash &&"
            '  export NVM_DIR="$HOME/.nvm" &&'
            '  \\. "$NVM_DIR/nvm.sh" || true &&'
            "  command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } &&"
            "  nvm install 22 && nvm alias default 22 && npm -v &&"
            f"  npm install -g --ignore-scripts @earendil-works/pi-coding-agent{version_spec};"
            " fi && "
            "pi --version"
        )
        symlink_run = (
            "for bin in node npm pi; do"
            '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
            '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
            '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
            "  fi;"
            " done"
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=root_run,
                ),
                InstallStep(user="agent", run=agent_run),
                InstallStep(user="root", run=symlink_run),
            ],
            verification_command=self.get_version_command(),
        )

    def _split_model(self) -> tuple[str | None, str | None]:
        model_name = self._command_model_name or self.model_name
        if not model_name:
            return None, None
        if "/" not in model_name:
            return self._provider or "openai", model_name
        provider, model_id = model_name.split("/", 1)
        return self._provider or provider, model_id

    def _base_url_value(self) -> str | None:
        return self._base_url or self._get_env("OPENAI_BASE_URL")

    def _run_model_name(self) -> str:
        model_name = self._command_model_name or self.model_name
        if not model_name:
            raise ValueError("Model name is required")
        if "/" in model_name or not self._base_url_value():
            return model_name

        provider, model_id = self._split_model()
        return f"{provider}/{model_id}"

    def _generated_models_config(self) -> dict[str, Any]:
        provider, model_id = self._split_model()
        base_url = self._base_url_value()
        if not provider or not model_id or not base_url:
            return {}

        model: dict[str, Any] = {
            "id": model_id,
            "name": model_id,
            "api": self._api,
            "reasoning": False,
            "input": ["text"],
            "compat": self._compat,
        }
        if self._context_window is not None:
            model["contextWindow"] = self._context_window
        if self._max_tokens is not None:
            model["maxTokens"] = self._max_tokens

        return {
            "providers": {
                provider: {
                    "baseUrl": base_url,
                    "apiKey": f"${self._api_key_env}",
                    "api": self._api,
                    "models": [model],
                }
            }
        }

    def _build_models_config_command(self) -> str | None:
        config = self._generated_models_config()
        if not config:
            return None
        escaped = shlex.quote(json.dumps(config, indent=2))
        return (
            'mkdir -p "$PI_CODING_AGENT_DIR" && '
            f'printf "%s\\n" {escaped} > "$PI_CODING_AGENT_DIR/models.json"'
        )

    def _build_settings_command(self) -> str:
        settings = {
            "enableInstallTelemetry": False,
            "quietStartup": True,
            "theme": "dark",
        }
        escaped = shlex.quote(json.dumps(settings, indent=2))
        return (
            'mkdir -p "$PI_CODING_AGENT_DIR" && '
            f'printf "%s\\n" {escaped} > "$PI_CODING_AGENT_DIR/settings.json"'
        )

    def _build_process_env(self) -> dict[str, str]:
        env = self.build_process_env()
        if value := self._get_env(self._api_key_env):
            env[self._api_key_env] = value
        if value := self._get_env("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = value

        env["PI_CODING_AGENT_DIR"] = self._REMOTE_PI_AGENT_DIR.as_posix()
        env["PI_OFFLINE"] = "1"
        env["PI_SKIP_VERSION_CHECK"] = "1"
        env["PI_TELEMETRY"] = "0"
        return env

    def network_allowlist(self) -> NetworkAllowlist:
        return allowlist_from_urls([self._base_url_value()])

    @staticmethod
    def _millis_to_iso(timestamp_ms: int | float | None) -> str | None:
        if timestamp_ms is None:
            return None
        try:
            return datetime.fromtimestamp(
                timestamp_ms / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, ValueError, OverflowError):
            return None

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    @classmethod
    def _extract_text_and_reasoning(cls, content: Any) -> tuple[str, str | None]:
        if isinstance(content, str):
            return content, None

        if not isinstance(content, list):
            return cls._stringify(content), None

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                text_parts.append(cls._stringify(part))
                continue
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append(cls._stringify(part.get("text")))
            elif ptype == "thinking":
                reasoning_parts.append(cls._stringify(part.get("thinking")))

        text = "\n".join(part for part in text_parts if part).strip()
        reasoning = "\n".join(part for part in reasoning_parts if part).strip()
        return text, reasoning or None

    @staticmethod
    def _extract_tool_calls(message: dict[str, Any] | None) -> list[ToolCall]:
        if not isinstance(message, dict):
            return []
        content = message.get("content")
        if not isinstance(content, list):
            return []

        tool_calls: list[ToolCall] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "toolCall":
                continue
            arguments = part.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            tool_calls.append(
                ToolCall(
                    tool_call_id=part.get("id") or "",
                    function_name=part.get("name") or "",
                    arguments=arguments,
                )
            )
        return tool_calls

    @classmethod
    def _format_tool_result(cls, result: Any) -> str:
        if not isinstance(result, dict):
            return cls._stringify(result)

        if "content" in result:
            content = result.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(cls._stringify(item.get("text")))
                    else:
                        parts.append(cls._stringify(item))
                return "\n".join(part for part in parts if part)

        return cls._stringify(result)

    @classmethod
    def _extract_observation(
        cls,
        tool_calls: list[ToolCall],
        tool_results: list[Any],
        execution_results: dict[str, Any],
    ) -> Observation | None:
        results: list[ObservationResult] = []

        for raw in tool_results:
            if not isinstance(raw, dict):
                continue
            call_id = raw.get("toolCallId") or raw.get("toolCallID") or raw.get("id")
            results.append(
                ObservationResult(
                    source_call_id=call_id,
                    content=cls._format_tool_result(raw),
                )
            )

        existing = {result.source_call_id for result in results}
        for tool_call in tool_calls:
            call_id = tool_call.tool_call_id
            if call_id in existing or call_id not in execution_results:
                continue
            results.append(
                ObservationResult(
                    source_call_id=call_id or None,
                    content=cls._format_tool_result(execution_results[call_id]),
                )
            )

        return Observation(results=results) if results else None

    @staticmethod
    def _metrics_from_usage(usage: Any) -> Metrics | None:
        if not isinstance(usage, dict):
            return None

        input_tokens = usage.get("input")
        output_tokens = usage.get("output")
        cached_tokens = usage.get("cacheRead")
        cache_write = usage.get("cacheWrite")
        total_tokens = usage.get("totalTokens")
        cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}
        total_cost = cost.get("total") if isinstance(cost, dict) else None

        if not any(
            value is not None
            for value in (input_tokens, output_tokens, cached_tokens, total_cost)
        ):
            return None

        prompt_tokens = (input_tokens or 0) + (cached_tokens or 0)
        extra = {
            "cache_write_tokens": cache_write,
            "total_tokens": total_tokens,
        }
        if isinstance(cost, dict) and cost:
            extra["cost_breakdown"] = cost

        return Metrics(
            prompt_tokens=prompt_tokens or None,
            completion_tokens=output_tokens or None,
            cached_tokens=cached_tokens or None,
            cost_usd=total_cost,
            extra={k: v for k, v in extra.items() if v is not None} or None,
        )

    def _parse_stdout(self) -> list[dict[str, Any]]:
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _convert_events_to_trajectory(
        self, events: list[dict[str, Any]]
    ) -> Trajectory | None:
        if not events:
            return None

        session_id = "unknown"
        for event in events:
            if event.get("type") == "session" and event.get("id"):
                session_id = str(event["id"])
                break

        steps: list[Step] = []
        current_execution_results: dict[str, Any] = {}

        for event in events:
            event_type = event.get("type")

            if event_type == "tool_execution_end":
                call_id = event.get("toolCallId")
                if isinstance(call_id, str):
                    current_execution_results[call_id] = event.get("result")
                continue

            if event_type != "turn_end":
                continue

            message = event.get("message")
            if not isinstance(message, dict):
                continue

            text, reasoning = self._extract_text_and_reasoning(message.get("content"))
            tool_calls = self._extract_tool_calls(message)
            observation = self._extract_observation(
                tool_calls,
                event.get("toolResults")
                if isinstance(event.get("toolResults"), list)
                else [],
                current_execution_results,
            )
            current_execution_results = {}

            usage = message.get("usage")
            metrics = self._metrics_from_usage(usage)
            timestamp = self._millis_to_iso(message.get("timestamp"))
            model_name = None
            provider = message.get("provider")
            model = message.get("model")
            if isinstance(provider, str) and isinstance(model, str):
                model_name = f"{provider}/{model}"
            elif isinstance(model, str):
                model_name = model

            if not text and not tool_calls and observation is None:
                continue

            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=timestamp,
                    source="agent",
                    message=text,
                    reasoning_content=reasoning,
                    model_name=model_name or self.model_name,
                    tool_calls=tool_calls or None,
                    observation=observation,
                    metrics=metrics,
                    llm_call_count=1,
                )
            )

        if not steps:
            return None

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_cost = 0.0
        saw_prompt = saw_completion = saw_cached = saw_cost = False
        for step in steps:
            if not step.metrics:
                continue
            if step.metrics.prompt_tokens is not None:
                total_prompt_tokens += step.metrics.prompt_tokens
                saw_prompt = True
            if step.metrics.completion_tokens is not None:
                total_completion_tokens += step.metrics.completion_tokens
                saw_completion = True
            if step.metrics.cached_tokens is not None:
                total_cached_tokens += step.metrics.cached_tokens
                saw_cached = True
            if step.metrics.cost_usd is not None:
                total_cost += step.metrics.cost_usd
                saw_cost = True

        summarization_count = sum(
            1 for event in events if event.get("type") == "compaction_end"
        )
        final_metrics = FinalMetrics(
            total_prompt_tokens=total_prompt_tokens if saw_prompt else None,
            total_completion_tokens=total_completion_tokens if saw_completion else None,
            total_cached_tokens=total_cached_tokens if saw_cached else None,
            total_cost_usd=total_cost if saw_cost else None,
            total_steps=len(steps),
            extra=extra_with_context_metrics(
                {"compacted": True} if summarization_count else None,
                peak_context_tokens=peak_context_tokens_from_steps(steps),
                summarization_count=summarization_count,
            ),
        )

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._parse_stdout()
        if not events:
            stderr_path = self.logs_dir / self._STDERR_FILENAME
            if stderr_path.exists():
                context.metadata = {
                    **(context.metadata or {}),
                    "pi_stderr_log": str(stderr_path),
                }
            return

        try:
            trajectory = self._convert_events_to_trajectory(events)
        except Exception:
            self.logger.exception("Failed to convert Pi events to trajectory")
            return

        if not trajectory:
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            trajectory_path.write_text(
                format_trajectory_json(trajectory.to_json_dict()),
                encoding="utf-8",
            )
            self.logger.debug(f"Wrote Pi trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(
                f"Failed to write trajectory file {trajectory_path}: {exc}"
            )

        if trajectory.final_metrics:
            populate_context_from_final_metrics(context, trajectory.final_metrics)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name:
            raise ValueError("Model name is required")

        env = self._build_process_env()
        setup_commands = [
            f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}",
            self._build_settings_command(),
        ]
        if models_command := self._build_models_config_command():
            setup_commands.append(models_command)

        escaped_instruction = shlex.quote(instruction)
        setup_commands.append(
            f'printf "%s" {escaped_instruction} > {self._REMOTE_INSTRUCTION_PATH.as_posix()}'
        )

        await self.exec_as_agent(
            environment,
            command="\n".join(setup_commands),
            env=env,
        )

        model = shlex.quote(self._run_model_name())
        cli_flags = self.build_cli_flags()
        cli_flags_arg = f"{cli_flags} " if cli_flags else ""

        command = (
            "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
            "pi --mode json --no-session "
            "--no-extensions --no-skills --no-prompt-templates --no-themes "
            "--no-context-files "
            f"--model {model} "
            f"{cli_flags_arg}"
            f"< {self._REMOTE_INSTRUCTION_PATH.as_posix()} "
            f"2> {EnvironmentPaths.agent_dir / self._STDERR_FILENAME} "
            f"| tee {EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME}"
        )

        await self.exec_as_agent(
            environment,
            command=command,
            env=env,
        )
