"""prompt-generator plugin — structured prompt generation on the Hermes LLM.

Backend (no vendor names, no external services, no extra API keys):
  host — the Hermes host LLM via ``ctx.llm`` (the session's configured
         model). This is the PRIMARY engine.
  cli  — optional external CLI delegation: runs ``<bin> -p "<prompt>"`` and
         relays stdout. Binary configured via the PROMPT_GENERATOR_CLI env
         var. Opt-in with ``--backend cli``.

Preference: host (Hermes LLM) by default; ``--backend`` forces a choice.

Commands:
    /prompt-gen <idea> [--framework race|care|ape|create|tag|creo|rise|pain|coast|roses|react|costar] [--claude|--gpt|--midjourney|--flux|--sd] [--backend cli|host]
        -> structured prompt using any of the 12 Prompt-Framework templates
           (RACE default)
    /prompt-gen-image <image> [--midjourney|--flux|--sd] [--backend cli|host]
        -> vision description of the image assembled into a paste-ready
           image-generation prompt; default engine is Hermes' configured
           vision model (vision_analyze routing)

Bypass prefixes: /quick, *simple, #basic (same convention as prompt-optimizer).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

_CLI_TIMEOUT_S = int(os.getenv("PROMPT_GENERATOR_CLI_TIMEOUT", "300"))
_ctx = None  # set at register() time for the host-model fallback


# ---------------------------------------------------------------------------
# Backends — generic, vendor-neutral
# ---------------------------------------------------------------------------

def _cli_bin() -> str | None:
    """Resolve the configured CLI binary, or None when not configured."""
    env = (os.getenv("PROMPT_GENERATOR_CLI", "") or "").strip()
    if not env:
        return None
    if os.path.sep in env or os.path.isfile(env):
        return env
    return shutil.which(env)  # bare name on PATH


def _cli_backend():
    """Return a (prompt, model) -> str callable for the CLI backend, or None."""
    bin_path = _cli_bin()
    if bin_path is None:
        return None

    def run(prompt: str, model=None):
        cmd = [bin_path, "-p", prompt]
        logger.info("prompt-generator: running %s -p (cli backend)", bin_path)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_CLI_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return (
                f"CLI backend timed out after {_CLI_TIMEOUT_S}s. "
                f"Raise PROMPT_GENERATOR_CLI_TIMEOUT or simplify the idea."
            )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            return (
                f"CLI backend failed (exit {proc.returncode}): {err[:500]}\n"
                f"Check PROMPT_GENERATOR_CLI points at your CLI."
            )
        return (proc.stdout or "").strip() or "(CLI produced no output)"

    return run


def _host_backend():
    """Return a (messages, **kwargs) -> response callable, or None."""
    if _ctx is None:
        return None
    llm = getattr(_ctx, "llm", None)
    if llm is None:
        return None
    return llm.complete


def _available_backends() -> dict:
    backends = {}
    cli = _cli_backend()
    if cli is not None:
        backends["cli"] = cli
    host = _host_backend()
    if host is not None:
        backends["host"] = host
    return backends


def _run_generic(prompt: str, forced: str | None = None) -> str:
    """Run a prompt through the best available backend; return plain text."""
    backends = _available_backends()
    if not backends:
        return (
            "prompt-generator: no backend available. Set PROMPT_GENERATOR_CLI "
            "to your CLI binary, or run inside a Hermes session with a "
            "configured model."
        )
    key = (forced or "").strip().lower()
    if not key:
        # Preference: the Hermes host LLM (ctx.llm) is the primary engine;
        # the configured CLI is an explicit opt-in via --backend cli.
        key = "host" if "host" in backends else "cli"
    if key not in backends:
        return (
            f"prompt-generator: backend '{key}' not available "
            f"(have: {', '.join(sorted(backends)) or 'none'})."
        )
    try:
        if key == "cli":
            return backends["cli"](prompt, None)
        result = backends["host"](
            [{"role": "user", "content": prompt}]
        )
        text = getattr(result, "text", None) or getattr(result, "content", None)
        return str(text).strip() if text else "(empty response)"
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt-generator: generation failed: %s", exc)
        return f"prompt-generator: generation failed: {exc}"


# ---------------------------------------------------------------------------
# Generation templates
# ---------------------------------------------------------------------------

_TEXT_GEN_SYSTEM = (
    "You are a prompt engineering expert. Turn the user's rough idea into a "
    "single, well-structured prompt following the framework and fields "
    "specified below.\n"
    "Rules: preserve ALL intent and details from the idea; be specific and "
    "actionable; use imperative verbs; no filler. Output ONLY the final "
    "prompt, ready to paste into any AI chat — include every field listed."
)

# Prompt framework definitions (fields + output template), mirrored from the
# Prompt-Framework library (github.com/Subhagatoadak/Prompt_Framework).
# value = (field list for the LLM, exact output template).
_FRAMEWORKS = {
    "race": (
        "Role, Action, Context, Explanation",
        "Role: <role>\nAction: <action>\nContext: <context>\nExplanation: <explanation>",
    ),
    "care": (
        "Context, Action, Expected Result, Example",
        "Context: <context>\nAction: <action>\nExpected Result: <expected result>\nExample: <example>",
    ),
    "ape": (
        "Action, Purpose, Execution",
        "Action: <action>\nPurpose: <purpose>\nExecution: <execution>",
    ),
    "create": (
        "Character, Request, Examples, Adjustment, Type of Output",
        "Character: <character>\nRequest: <request>\nExamples: <examples>\nAdjustment: <adjustment>\nType of Output: <output type>",
    ),
    "tag": (
        "Task, Action, Goal",
        "Task: <task>\nAction: <action>\nGoal: <goal>",
    ),
    "creo": (
        "Context, Request, Explanation, Outcome",
        "Context: <context>\nRequest: <request>\nExplanation: <explanation>\nOutcome: <outcome>",
    ),
    "rise": (
        "Role, Input, Steps, Execution",
        "Role: <role>\nInput: <input>\nSteps: <steps>\nExecution: <execution>",
    ),
    "pain": (
        "Problem, Action, Information, Next Steps",
        "Problem: <problem>\nAction: <action>\nInformation: <information>\nNext Steps: <next steps>",
    ),
    "coast": (
        "Context, Objective, Actions, Scenario, Task",
        "Context: <context>\nObjective: <objective>\nActions: <actions>\nScenario: <scenario>\nTask: <task>",
    ),
    "roses": (
        "Role, Objective, Scenario, Expected Solution, Steps",
        "Role: <role>\nObjective: <objective>\nScenario: <scenario>\nExpected Solution: <expected solution>\nSteps: <steps>",
    ),
    "react": (
        "Context, Task, Explanation",
        "Context: <context>\nTask: <task>\nExplanation: <explanation>",
    ),
    "costar": (
        "Context, Output Type, Style, Reasoning",
        "Context: <context>\nOutput Type: <output type>\nStyle: <style>\nReasoning: <reasoning>",
    ),
}
_DEFAULT_FRAMEWORK = "race"

_MODEL_FLAVOURS = {
    "claude": (
        "Format the prompt for Anthropic Claude: use XML-style tags "
        "(<task>, <context>, <output>) to mark structure."
    ),
    "gpt": (
        "Format the prompt for OpenAI GPT: explicit output-format spec, "
        "numbered steps for multi-part tasks."
    ),
    "midjourney": (
        "Format as a Midjourney prompt: rich sensory scene description "
        "followed by style keywords and --flags (e.g. --ar 16:9 --stylize 250)."
    ),
    "flux": (
        "Format as a FLUX.1 prompt: detailed plain-description style, "
        "natural language, no flags."
    ),
    "sd": (
        "Format as a Stable Diffusion prompt: keyword-rich description plus "
        "a Negative prompt: line."
    ),
}

_IMAGE_GEN_QUESTIONS = {
    "midjourney": (
        "Analyze this image in detail (subject, composition, lighting, colors, "
        "mood, textures, camera angle). Then write a single ready-to-paste "
        "Midjourney prompt: a rich sensory scene description followed by "
        "style keywords and --flags (--ar 16:9 --stylize 250 --chaos 20). "
        "Output ONLY the prompt."
    ),
    "flux": (
        "Analyze this image in detail (subject, composition, lighting, colors, "
        "mood, textures). Then write a single ready-to-paste FLUX.1 prompt: "
        "detailed natural-language plain description, no flags. Output ONLY "
        "the prompt."
    ),
    "sd": (
        "Analyze this image in detail (subject, composition, lighting, colors, "
        "mood, textures). Then write a single ready-to-paste Stable Diffusion "
        "prompt: keyword-rich description plus a 'Negative prompt:' line. "
        "Output ONLY the prompt."
    ),
}
_IMAGE_DEFAULT_STYLE = "midjourney"
_IMAGE_FALLBACK_QUESTION = (
    "Describe this image in detail and write a prompt that could recreate it."
)


# ---------------------------------------------------------------------------
# Shared flag parsing
# ---------------------------------------------------------------------------

def _parse_flags(tokens: list, value_flags: set, known_flags: set):
    """Generic flag splitter.

    Returns (kept_positional, flag_values, unknown_flags).
    value_flags: flags that consume the next token (e.g. backend).
    known_flags: boolean-style flags we recognise (e.g. claude, flux).
    """
    kept = []
    values = {}
    unknown = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:].lower()
            if key in value_flags:
                if i + 1 < len(tokens):
                    values[key] = tokens[i + 1]
                    i += 1
                else:
                    unknown.append(t)
            elif key in known_flags:
                values[key] = True
            else:
                unknown.append(t)
        else:
            kept.append(t)
        i += 1
    return kept, values, unknown


# ---------------------------------------------------------------------------
# /prompt-gen — idea -> structured prompt
# ---------------------------------------------------------------------------

def _handle_prompt_gen(raw_args: str) -> str:
    kept, flags, _ = _parse_flags(
        raw_args.strip().split(),
        value_flags={"backend", "framework"},
        known_flags=set(_MODEL_FLAVOURS) | set(_FRAMEWORKS),
    )
    idea = " ".join(kept).strip()
    flavour = next((f for f in _MODEL_FLAVOURS if flags.get(f)), None)
    # Framework: --framework <name> takes precedence, then a bare --<name>
    # flag, then the default (race).
    framework = flags.get("framework")
    if not framework:
        framework = next((f for f in _FRAMEWORKS if flags.get(f)), None)
    if not framework:
        framework = _DEFAULT_FRAMEWORK
    if framework not in _FRAMEWORKS:
        return (
            f"Unknown framework: {framework}. Available: "
            f"{', '.join(sorted(_FRAMEWORKS))}"
        )
    if not idea:
        return (
            "Usage: /prompt-gen <idea> "
            "[--framework race|care|ape|create|tag|creo|rise|pain|coast|"
            "roses|react|costar] "
            "[--claude|--gpt|--midjourney|--flux|--sd] [--backend cli|host]\n"
            "Example: /prompt-gen write a cold email to a recruiter --framework care --claude"
        )
    fields, template = _FRAMEWORKS[framework]
    system = (
        f"{_TEXT_GEN_SYSTEM}\n\n"
        f"FRAMEWORK: {framework.upper()}\n"
        f"FIELDS: {fields}\n"
        f"OUTPUT TEMPLATE:\n{template}"
    )
    if flavour:
        system += "\n\n" + _MODEL_FLAVOURS[flavour]
    prompt = f"{system}\n\nIDEA: {idea}\n\nFinal prompt:"
    return _run_generic(prompt, flags.get("backend"))


# ---------------------------------------------------------------------------
# /prompt-gen-image — image -> image-generation prompt
# ---------------------------------------------------------------------------

def _handle_prompt_gen_image(raw_args: str) -> str:
    kept, flags, _ = _parse_flags(
        raw_args.strip().split(),
        value_flags={"backend"},
        known_flags=set(_IMAGE_GEN_QUESTIONS),
    )
    if not kept:
        return (
            "Usage: /prompt-gen-image <image_path> "
            "[--midjourney|--flux|--sd] [--backend cli|host]\n"
            "Example: /prompt-gen-image /tmp/photo.png --flux"
        )
    image = kept[0]
    style = next((s for s in _IMAGE_GEN_QUESTIONS if flags.get(s)),
                 _IMAGE_DEFAULT_STYLE)
    import os as _os

    if not _os.path.isfile(_os.path.expanduser(image)):
        return f"Image not found: {image}"
    question = _IMAGE_GEN_QUESTIONS.get(style, _IMAGE_FALLBACK_QUESTION)
    backend = (flags.get("backend") or "").strip().lower()
    if backend == "cli":
        # Explicit opt-in: the raw CLI path (no vision routing).
        return _run_generic(f"{question}\n\nImage: {image}", forced="cli")
    # Default: dispatch through Hermes' vision_analyze tool, which routes via
    # the vision model configured in Hermes (auxiliary.vision — or whatever
    # override is registered, e.g. a CLI vision backend). The host LLM may
    # not be multimodal, so the configured vision model is the right engine
    # for the image path.
    if _ctx is None or not hasattr(_ctx, "dispatch_tool"):
        return (
            "prompt-generator: no vision routing available. Set "
            "--backend cli with PROMPT_GENERATOR_CLI configured, or run "
            "inside a Hermes session with a vision model configured."
        )
    try:
        raw = _ctx.dispatch_tool(
            "vision_analyze",
            {"image_url": image, "question": question},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("prompt-generator: vision_analyze dispatch failed: %s", exc)
        return f"prompt-generator: vision dispatch failed: {exc}"
    return _extract_vision_analysis(raw)


def _extract_vision_analysis(raw) -> str:
    """Parse the vision_analyze tool response into plain text.

    vision_analyze returns a JSON string {"success": bool, "analysis": str};
    tolerate raw text responses (some backends return analysis directly).
    """
    import json as _json

    if not raw:
        return "(vision produced no output)"
    if isinstance(raw, str):
        text = raw.strip()
        try:
            payload = _json.loads(text)
        except ValueError:
            return text  # not JSON — plain analysis text
        if isinstance(payload, dict):
            if payload.get("success") is False:
                return f"vision failed: {payload.get('analysis', 'unknown error')}"
            analysis = payload.get("analysis", "")
            return analysis.strip() or "(vision produced no output)"
        return text
    return str(raw).strip() or "(vision produced no output)"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    global _ctx
    _ctx = ctx
    logger.info("prompt-generator: initializing (backends: %s)",
                ", ".join(sorted(_available_backends())) or "none")
    ctx.register_command(
        "prompt-gen",
        handler=_handle_prompt_gen,
        description=(
            "Generate a structured prompt from a rough idea using any of the "
            "12 Prompt-Framework templates (race, care, ape, create, tag, "
            "creo, rise, pain, coast, roses, react, costar). Runs on the "
            "Hermes host LLM by default; optional --claude|--gpt|"
            "--midjourney|--flux|--sd."
        ),
        args_hint="<idea> [--framework race|care|ape|create|tag|creo|rise|pain|coast|roses|react|costar] [--claude|--gpt|--midjourney|--flux|--sd] [--backend cli|host]",
    )
    ctx.register_command(
        "prompt-gen-image",
        handler=_handle_prompt_gen_image,
        description=(
            "Turn an image into a paste-ready image-generation prompt "
            "(Midjourney/FLUX/SD). Uses Hermes' configured vision model by "
            "default; --backend cli opts into the raw CLI path."
        ),
        args_hint="<image_path> [--midjourney|--flux|--sd] [--backend cli|host]",
    )
    logger.info("prompt-generator: registered /prompt-gen and /prompt-gen-image")
