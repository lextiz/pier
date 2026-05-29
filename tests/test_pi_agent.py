import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pier.agents.factory import AgentFactory
from pier.agents.installed.pi import Pi
from pier.models.agent.context import AgentContext
from pier.models.agent.name import AgentName
from pier.models.trial.config import AgentConfig


class ExecResult:
    def __init__(self, return_code: int = 0, stdout: str = "", stderr: str = ""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class RecordingEnvironment:
    default_user = "agent"
    agent_install_spec = None

    def __init__(self):
        self.calls = []

    def agent_process_env(self, env):
        return env or {}

    async def exec(
        self,
        command,
        user=None,
        env=None,
        cwd=None,
        timeout_sec=None,
    ):
        self.calls.append(
            {
                "command": command,
                "user": user,
                "env": env or {},
                "cwd": cwd,
                "timeout_sec": timeout_sec,
            }
        )
        return ExecResult()


class PiAgentTests(unittest.TestCase):
    def test_factory_creation_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = AgentFactory.create_agent_from_config(
                AgentConfig(
                    name="pi",
                    model_name="openai/gemma4-26b-a4b-it-q6.gguf",
                    env={
                        "OPENAI_API_KEY": "sk-local",
                        "OPENAI_BASE_URL": "http://localhost:8080/v1",
                    },
                    kwargs={"thinking": "off"},
                ),
                logs_dir=Path(tmp),
            )

        self.assertIsInstance(agent, Pi)
        self.assertEqual(agent.name(), AgentName.PI.value)
        self.assertEqual(agent.model_name, "openai/gemma4-26b-a4b-it-q6.gguf")

    def test_install_spec_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = Pi(
                logs_dir=Path(tmp), model_name="openai/gpt-5.4", version="0.77.0"
            )
            spec = agent.install_spec()

        self.assertEqual(spec.agent_name, "pi")
        self.assertEqual(spec.version, "0.77.0")
        self.assertTrue(any(step.user == "root" for step in spec.steps))
        self.assertTrue(
            any(
                "npm install -g --ignore-scripts "
                "@earendil-works/pi-coding-agent@0.77.0" in step.run
                for step in spec.steps
            )
        )
        self.assertEqual(spec.verification_command, agent.get_version_command())

    def test_command_construction_and_env_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            agent = Pi(
                logs_dir=logs_dir,
                model_name="openai/gemma4-26b-a4b-it-q6.gguf",
                extra_env={
                    "OPENAI_API_KEY": "sk-local",
                    "OPENAI_BASE_URL": "http://localhost:8080/v1",
                },
                thinking="off",
            )
            environment = RecordingEnvironment()

            asyncio.run(agent.run("Fix the bug safely", environment, AgentContext()))

        self.assertGreaterEqual(len(environment.calls), 2)
        setup = environment.calls[0]
        run = environment.calls[1]

        self.assertEqual(setup["env"]["OPENAI_API_KEY"], "sk-local")
        self.assertEqual(setup["env"]["OPENAI_BASE_URL"], "http://localhost:8080/v1")
        self.assertEqual(setup["env"]["PI_CODING_AGENT_DIR"], "/tmp/pi-agent")
        self.assertEqual(setup["env"]["PI_OFFLINE"], "1")
        self.assertIn("models.json", setup["command"])
        self.assertIn('"baseUrl": "http://localhost:8080/v1"', setup["command"])
        self.assertIn('"id": "gemma4-26b-a4b-it-q6.gguf"', setup["command"])

        self.assertIn("pi --mode json --no-session", run["command"])
        self.assertIn(
            "--no-extensions --no-skills --no-prompt-templates", run["command"]
        )
        self.assertIn("--no-context-files", run["command"])
        self.assertIn("--model openai/gemma4-26b-a4b-it-q6.gguf", run["command"])
        self.assertIn("--thinking off", run["command"])
        self.assertIn("/logs/agent/pi.jsonl", run["command"])
        self.assertIn("/logs/agent/pi.stderr", run["command"])
        self.assertEqual(run["env"]["OPENAI_API_KEY"], "sk-local")
        self.assertEqual(run["env"]["PI_SKIP_VERSION_CHECK"], "1")

    def test_network_allowlist_extracts_model_base_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = Pi(
                logs_dir=Path(tmp),
                model_name="openai/gemma4-26b-a4b-it-q6.gguf",
                extra_env={
                    "OPENAI_API_KEY": "sk-local",
                    "OPENAI_BASE_URL": "http://host.docker.internal:8080/v1",
                },
            )
            allowlist = agent.network_allowlist()

            custom_agent = Pi(
                logs_dir=Path(tmp),
                model_name="local/gemma",
                base_url="http://127.0.0.1:11434/v1",
            )
            custom_allowlist = custom_agent.network_allowlist()

        self.assertIn("host.docker.internal", allowlist.domains)
        self.assertIn("127.0.0.1", custom_allowlist.domains)

    def test_custom_base_url_uses_single_provider_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = Pi(
                logs_dir=Path(tmp),
                model_name="local/gemma",
                base_url="http://127.0.0.1:11434/v1",
                api_key_env="LOCAL_API_KEY",
                extra_env={"LOCAL_API_KEY": "sk-local"},
            )
            environment = RecordingEnvironment()

            asyncio.run(agent.run("Fix the bug safely", environment, AgentContext()))

        setup = environment.calls[0]
        run = environment.calls[1]

        self.assertIn('"local": {', setup["command"])
        self.assertIn('"baseUrl": "http://127.0.0.1:11434/v1"', setup["command"])
        self.assertIn('"apiKey": "$LOCAL_API_KEY"', setup["command"])
        self.assertIn('"id": "gemma"', setup["command"])
        self.assertEqual(setup["env"]["LOCAL_API_KEY"], "sk-local")
        self.assertIn("--model local/gemma", run["command"])

    def test_json_events_create_minimal_atif_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            events = [
                {"type": "session", "version": 3, "id": "session-1", "cwd": "/app"},
                {"type": "turn_start"},
                {
                    "type": "tool_execution_end",
                    "toolCallId": "tool-1",
                    "toolName": "read",
                    "result": {
                        "content": [{"type": "text", "text": "file contents"}],
                        "isError": False,
                    },
                },
                {
                    "type": "turn_end",
                    "message": {
                        "role": "assistant",
                        "provider": "openai",
                        "model": "gemma4-26b-a4b-it-q6.gguf",
                        "timestamp": 1_700_000_000_000,
                        "content": [
                            {"type": "thinking", "thinking": "Inspect files."},
                            {
                                "type": "toolCall",
                                "id": "tool-1",
                                "name": "read",
                                "arguments": {"path": "README.md"},
                            },
                            {"type": "text", "text": "Done."},
                        ],
                        "usage": {
                            "input": 10,
                            "output": 5,
                            "cacheRead": 2,
                            "cacheWrite": 0,
                            "totalTokens": 17,
                            "cost": {
                                "input": 0.0,
                                "output": 0.0,
                                "cacheRead": 0.0,
                                "cacheWrite": 0.0,
                                "total": 0.0,
                            },
                        },
                    },
                    "toolResults": [],
                },
            ]
            (logs_dir / "pi.jsonl").write_text(
                "\n".join(json.dumps(event) for event in events),
                encoding="utf-8",
            )
            agent = Pi(
                logs_dir=logs_dir,
                model_name="openai/gemma4-26b-a4b-it-q6.gguf",
            )
            context = AgentContext()

            agent.populate_context_post_run(context)
            trajectory = json.loads((logs_dir / "trajectory.json").read_text())

        self.assertEqual(trajectory["session_id"], "session-1")
        self.assertEqual(trajectory["agent"]["name"], "pi")
        self.assertEqual(trajectory["steps"][0]["source"], "agent")
        self.assertEqual(trajectory["steps"][0]["message"], "Done.")
        self.assertEqual(trajectory["steps"][0]["reasoning_content"], "Inspect files.")
        self.assertEqual(
            trajectory["steps"][0]["tool_calls"][0]["function_name"],
            "read",
        )
        self.assertEqual(
            trajectory["steps"][0]["observation"]["results"][0]["content"],
            "file contents",
        )
        self.assertEqual(context.n_input_tokens, 12)
        self.assertEqual(context.n_output_tokens, 5)


if __name__ == "__main__":
    unittest.main()
