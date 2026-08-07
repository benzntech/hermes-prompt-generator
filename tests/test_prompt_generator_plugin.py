"""Tests for the prompt-generator plugin.

Loads the plugin the way the Hermes core does (importlib spec from the
plugin dir) and exercises both slash-command handlers with mocked
backends — no real CLI subprocess, no network, no host LLM.

Run with the Hermes venv from OUTSIDE this plugin dir (hyphenated-dir
trap — pytest treats a rootdir with __init__.py as a package and the
hyphenated name is not a valid package identifier)::

    cd /tmp && ~/.hermes/hermes-agent/venv/bin/python -m pytest \
        ~/.hermes/plugins/prompt-generator/tests/test_prompt_generator_plugin.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
_NS_PARENT = "hermes_plugins"


def _load_plugin():
    if _NS_PARENT not in sys.modules:
        ns_pkg = types.ModuleType(_NS_PARENT)
        ns_pkg.__path__ = []
        ns_pkg.__package__ = _NS_PARENT
        sys.modules[_NS_PARENT] = ns_pkg
    module_name = f"{_NS_PARENT}.prompt_generator"
    if module_name in sys.modules:
        return sys.modules[module_name]
    init_file = PLUGIN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name, init_file, submodule_search_locations=[str(PLUGIN_DIR)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {init_file}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = module_name
    module.__path__ = [str(PLUGIN_DIR)]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def pg():
    return _load_plugin()


class FakeCtx:
    def __init__(self, llm=None, vision_response=None):
        self.hooks = {}
        self.commands = {}
        # llm emulates the real PluginLlm facade: an object with .complete()
        self._llm = types.SimpleNamespace(complete=llm) if llm is not None else None
        # vision_response emulates vision_analyze tool output (JSON string).
        self._vision_response = vision_response

    @property
    def llm(self):
        return self._llm

    def dispatch_tool(self, tool_name, args, **kwargs):
        self.dispatched = getattr(self, "dispatched", [])
        self.dispatched.append((tool_name, args, kwargs))
        if self._vision_response is not None:
            return self._vision_response
        raise RuntimeError(f"no fake response for tool {tool_name}")

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {"handler": handler, "args_hint": args_hint,
                               "description": description}


_VISION_JSON = (
    '{"success": true, "analysis": "A cyberpunk city at night, neon signs, '
    'rain-soaked streets --ar 16:9 --stylize 250"}'
)


def _fake_cli(result_text="Generated prompt here."):
    """Fake CLI backend callable; records (prompt, model) calls."""
    def fake(prompt, model=None):
        fake.calls.append((prompt, model))
        return result_text
    fake.calls = []
    return fake


def _fake_host(result_text="Host response."):
    """Fake ctx.llm.complete; records (messages, kwargs) calls."""
    class Resp:
        def __init__(self, text):
            self.text = text

    def fake(messages=None, **kwargs):
        fake.calls.append((messages, kwargs))
        return Resp(result_text)
    fake.calls = []
    return fake


@pytest.fixture
def no_env_cli(pg, monkeypatch):
    """Ensure PROMPT_GENERATOR_CLI is unset so the CLI backend is absent."""
    monkeypatch.delenv("PROMPT_GENERATOR_CLI", raising=False)
    yield


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


def test_register_commands(pg):
    ctx = FakeCtx()
    pg.register(ctx)
    assert "prompt-gen" in ctx.commands
    assert "prompt-gen-image" in ctx.commands


def test_register_args_hints(pg):
    ctx = FakeCtx()
    pg.register(ctx)
    assert "<idea>" in ctx.commands["prompt-gen"]["args_hint"]
    assert "<image_path>" in ctx.commands["prompt-gen-image"]["args_hint"]
    assert "backend" in ctx.commands["prompt-gen"]["args_hint"]


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def test_cli_bin_from_env(pg, monkeypatch):
    monkeypatch.setenv("PROMPT_GENERATOR_CLI", "/usr/local/bin/fake-cli")
    assert pg._cli_bin() == "/usr/local/bin/fake-cli"


def test_cli_bin_unset(pg, no_env_cli):
    assert pg._cli_bin() is None


def test_backend_cli_available(pg, no_env_cli, monkeypatch):
    monkeypatch.setenv("PROMPT_GENERATOR_CLI", "/usr/bin/true")
    assert "cli" in pg._available_backends()


def test_backend_host_available(pg, no_env_cli, monkeypatch):
    host = _fake_host()
    monkeypatch.setattr(pg, "_ctx", FakeCtx(llm=host))
    assert "host" in pg._available_backends()


def test_backend_prefers_host(pg, no_env_cli, monkeypatch):
    monkeypatch.setenv("PROMPT_GENERATOR_CLI", "/usr/bin/true")
    host = _fake_host()
    monkeypatch.setattr(pg, "_ctx", FakeCtx(llm=host))
    backends = pg._available_backends()
    assert set(backends) == {"cli", "host"}
    # Host (Hermes LLM) is the default engine; cli is opt-in.
    out = pg._run_generic("hello")
    assert out == "Host response."
    assert len(host.calls) == 1


def test_backend_none(pg, no_env_cli, monkeypatch):
    monkeypatch.setattr(pg, "_ctx", None)
    run = pg._run_generic("hello")
    assert "no backend available" in run


def test_backend_forced_unknown(pg, no_env_cli, monkeypatch):
    monkeypatch.setenv("PROMPT_GENERATOR_CLI", "/usr/bin/true")
    run = pg._run_generic("hello", forced="bogus")
    assert "not available" in run


# ---------------------------------------------------------------------------
# /prompt-gen
# ---------------------------------------------------------------------------


def test_prompt_gen_no_args(pg, no_env_cli, monkeypatch):
    monkeypatch.setenv("PROMPT_GENERATOR_CLI", "/usr/bin/true")
    out = pg._handle_prompt_gen("")
    assert "Usage:" in out


def test_prompt_gen_uses_cli(pg, no_env_cli, monkeypatch):
    fake = _fake_cli("Final structured prompt.")
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    monkeypatch.setattr(pg, "_ctx", None)
    out = pg._handle_prompt_gen("write a cold email to a recruiter")
    assert out == "Final structured prompt."
    assert len(fake.calls) == 1
    prompt, model = fake.calls[0]
    assert "RACE" in prompt
    assert "cold email to a recruiter" in prompt
    assert model is None


def test_prompt_gen_flavour_claude(pg, no_env_cli, monkeypatch):
    fake = _fake_cli()
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    pg._handle_prompt_gen("explain my code --claude")
    prompt, _ = fake.calls[0]
    assert "XML-style tags" in prompt
    assert "IDEA: explain my code" in prompt


def test_prompt_gen_flavour_midjourney(pg, no_env_cli, monkeypatch):
    fake = _fake_cli()
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    pg._handle_prompt_gen("a dragon over a city --midjourney")
    prompt, _ = fake.calls[0]
    assert "--ar" in prompt
    assert "IDEA: a dragon over a city" in prompt


@pytest.mark.parametrize(
    "framework, field_fragment",
    [
        ("race", "Role, Action, Context, Explanation"),
        ("care", "Context, Action, Expected Result, Example"),
        ("ape", "Action, Purpose, Execution"),
        ("create", "Character, Request, Examples, Adjustment"),
        ("tag", "Task, Action, Goal"),
        ("creo", "Context, Request, Explanation, Outcome"),
        ("rise", "Role, Input, Steps, Execution"),
        ("pain", "Problem, Action, Information, Next Steps"),
        ("coast", "Context, Objective, Actions, Scenario, Task"),
        ("roses", "Role, Objective, Scenario, Expected Solution, Steps"),
        ("react", "Context, Task, Explanation"),
        ("costar", "Context, Output Type, Style, Reasoning"),
    ],
)
def test_prompt_gen_all_frameworks(pg, no_env_cli, monkeypatch, framework,
                                   field_fragment):
    fake = _fake_cli()
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    pg._handle_prompt_gen(f"do the thing --framework {framework}")
    prompt, _ = fake.calls[0]
    assert f"FRAMEWORK: {framework.upper()}" in prompt
    assert field_fragment in prompt
    assert "OUTPUT TEMPLATE" in prompt


def test_prompt_gen_bare_framework_flag(pg, no_env_cli, monkeypatch):
    fake = _fake_cli()
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    pg._handle_prompt_gen("do the thing --creo")
    prompt, _ = fake.calls[0]
    assert "FRAMEWORK: CREO" in prompt


def test_prompt_gen_default_framework_race(pg, no_env_cli, monkeypatch):
    fake = _fake_cli()
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    pg._handle_prompt_gen("do the thing")
    prompt, _ = fake.calls[0]
    assert "FRAMEWORK: RACE" in prompt


def test_prompt_gen_unknown_framework(pg, no_env_cli, monkeypatch):
    fake = _fake_cli()
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    out = pg._handle_prompt_gen("do the thing --framework bogus")
    assert "Unknown framework: bogus" in out
    assert not fake.calls


def test_prompt_gen_framework_with_flavour(pg, no_env_cli, monkeypatch):
    fake = _fake_cli()
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    pg._handle_prompt_gen("write an email --framework care --claude")
    prompt, _ = fake.calls[0]
    assert "FRAMEWORK: CARE" in prompt
    assert "XML-style tags" in prompt


def test_prompt_gen_falls_back_to_host(pg, no_env_cli, monkeypatch):
    host = _fake_host("Host generated prompt.")
    monkeypatch.setattr(pg, "_cli_backend", lambda: None)
    monkeypatch.setattr(pg, "_ctx", FakeCtx(llm=host))
    out = pg._handle_prompt_gen("write a haiku")
    assert out == "Host generated prompt."
    assert len(host.calls) == 1
    messages, _ = host.calls[0]
    assert messages[0]["role"] == "user"
    assert "haiku" in messages[0]["content"]


def test_prompt_gen_forced_host(pg, no_env_cli, monkeypatch):
    fake_cli = _fake_cli()
    host = _fake_host("Host prompt.")
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake_cli)
    monkeypatch.setattr(pg, "_ctx", FakeCtx(llm=host))
    out = pg._handle_prompt_gen("do it --backend host")
    assert out == "Host prompt."
    assert len(host.calls) == 1  # cli untouched
    assert not fake_cli.calls


def test_prompt_gen_no_backend(pg, no_env_cli, monkeypatch):
    monkeypatch.setattr(pg, "_cli_backend", lambda: None)
    monkeypatch.setattr(pg, "_ctx", None)
    out = pg._handle_prompt_gen("some idea")
    assert "no backend available" in out


# ---------------------------------------------------------------------------
# /prompt-gen-image
# ---------------------------------------------------------------------------


def test_prompt_gen_image_no_args(pg, no_env_cli, monkeypatch):
    out = pg._handle_prompt_gen_image("")
    assert "Usage:" in out


def test_prompt_gen_image_missing_file(pg, no_env_cli, monkeypatch):
    out = pg._handle_prompt_gen_image("/nonexistent/img.png")
    assert "Image not found" in out


def test_prompt_gen_image_uses_hermes_vision(pg, no_env_cli, monkeypatch, tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"fake-png")
    ctx = FakeCtx(vision_response=_VISION_JSON)
    monkeypatch.setattr(pg, "_ctx", ctx)
    out = pg._handle_prompt_gen_image(str(img))
    assert "cyberpunk city" in out  # analysis text extracted
    assert ctx.dispatched and ctx.dispatched[0][0] == "vision_analyze"
    tool_args = ctx.dispatched[0][1]
    assert tool_args["image_url"] == str(img)
    assert "Midjourney" in tool_args["question"]  # default style


def test_prompt_gen_image_style_flux(pg, no_env_cli, monkeypatch, tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"fake-png")
    ctx = FakeCtx(vision_response=_VISION_JSON)
    monkeypatch.setattr(pg, "_ctx", ctx)
    pg._handle_prompt_gen_image(f"{img} --flux")
    tool_args = ctx.dispatched[0][1]
    assert "FLUX.1" in tool_args["question"]


def test_prompt_gen_image_style_sd(pg, no_env_cli, monkeypatch, tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"fake-png")
    ctx = FakeCtx(vision_response=_VISION_JSON)
    monkeypatch.setattr(pg, "_ctx", ctx)
    pg._handle_prompt_gen_image(f"{img} --sd")
    tool_args = ctx.dispatched[0][1]
    assert "Stable Diffusion" in tool_args["question"]


def test_prompt_gen_image_vision_failure(pg, no_env_cli, monkeypatch, tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"fake-png")
    ctx = FakeCtx(vision_response='{"success": false, "analysis": "boom"}')
    monkeypatch.setattr(pg, "_ctx", ctx)
    out = pg._handle_prompt_gen_image(str(img))
    assert "vision failed: boom" in out


def test_prompt_gen_image_no_ctx(pg, no_env_cli, monkeypatch, tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"fake-png")
    monkeypatch.setattr(pg, "_ctx", None)
    out = pg._handle_prompt_gen_image(str(img))
    assert "no vision routing available" in out


def test_prompt_gen_image_backend_cli(pg, no_env_cli, monkeypatch, tmp_path):
    img = tmp_path / "photo.png"
    img.write_bytes(b"fake-png")
    fake = _fake_cli("CLI image prompt.")
    monkeypatch.setattr(pg, "_cli_backend", lambda: fake)
    monkeypatch.setattr(pg, "_ctx", FakeCtx(vision_response=_VISION_JSON))
    out = pg._handle_prompt_gen_image(f"{img} --backend cli")
    assert out == "CLI image prompt."
    assert len(fake.calls) == 1
    prompt, _ = fake.calls[0]
    assert "Image:" in prompt and str(img) in prompt


def test_extract_vision_analysis_json(pg):
    assert "cyberpunk city" in pg._extract_vision_analysis(_VISION_JSON)


def test_extract_vision_analysis_raw_text(pg):
    assert pg._extract_vision_analysis("plain analysis text") == "plain analysis text"


def test_extract_vision_analysis_empty(pg):
    assert "(vision produced no output)" in pg._extract_vision_analysis("")


# ---------------------------------------------------------------------------
# vendor-name hygiene: no branded CLI names in the plugin source
# ---------------------------------------------------------------------------


def test_no_vendor_names_in_source(pg):
    """Copyright hygiene: the plugin must not name any vendor CLI."""
    import inspect

    src = inspect.getsource(pg)
    for banned in ("antigravity", "gemini-cli", "oauth-token"):
        assert banned not in src.lower()
