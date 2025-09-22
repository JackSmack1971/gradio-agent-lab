import os, json, time, requests
import gradio as gr
from functools import lru_cache

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}" if OPENROUTER_API_KEY else "",
    "HTTP-Referer": "http://localhost:7860",  # optional attribution
    "X-Title": "Agent Lab (Gradio 5)",        # optional attribution
}

DEFAULT_MODEL_ID = "openrouter/auto"
DEFAULT_MODEL_CHOICES: tuple[tuple[str, str], ...] = ((DEFAULT_MODEL_ID, DEFAULT_MODEL_ID),)
DEFAULT_MODELS_RAW = [{"id": DEFAULT_MODEL_ID, "name": DEFAULT_MODEL_ID}]

# =========================================================
# Models API helpers (sidebar metadata)  ??????????????????
# =========================================================

@lru_cache(maxsize=1)
def fetch_models_raw():
    """Fetch full model list once (cached)."""
    url = f"{BASE_URL}/models"
    headers = {k: v for k, v in HEADERS.items() if v}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response else None
        # Security: ensure we do not leak auth failures—warn user and fall back safely.
        if status in {401, 403}:
            gr.Warning("OpenRouter API key was rejected while loading models; using defaults.")
        else:
            gr.Warning("Unable to load OpenRouter models; using default list.")
        return DEFAULT_MODELS_RAW
    except requests.RequestException:
        # Network failures should not break the UI; warn and fall back to the default.
        gr.Warning("OpenRouter models endpoint unreachable; using default list.")
        return DEFAULT_MODELS_RAW
    try:
        data = response.json().get("data", [])
    except ValueError:
        gr.Warning("Received invalid response from OpenRouter; using default model list.")
        return DEFAULT_MODELS_RAW
    return data or DEFAULT_MODELS_RAW

def list_model_choices():
    """Dropdown choices -> (label, value)."""
    data = fetch_models_raw()
    choices = []
    for m in data:
        mid = m.get("id") or m.get("name")
        if not mid:
            continue
        choices.append((mid, mid))
    if not choices:
        return list(DEFAULT_MODEL_CHOICES)
    # preferred first
    preferred = [
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash",
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-70b-instruct",
        "deepseek/deepseek-chat",
        "qwen/qwen2.5-72b-instruct",
        "openrouter/auto",
    ]
    priority = {m:i for i,m in enumerate(preferred)}
    choices.sort(key=lambda x: (priority.get(x[1], 10_000), x[1]))
    return choices

def find_model(md_id: str):
    """Return the raw model object by id."""
    for m in fetch_models_raw():
        if (m.get("id") or m.get("name")) == md_id:
            return m
    return None

def _fmt_price(p):
    """Format pricing dict entries per 1M tokens when present."""
    # OpenRouter often exposes pricing like {'prompt': 3.0, 'completion': 15.0, ...} (USD per 1M).
    # Be defensive across providers.
    if not isinstance(p, dict):
        return "—"
    prompt = p.get("prompt")
    completion = p.get("completion")
    extras = {k:v for k,v in p.items() if k not in ("prompt", "completion")}
    parts = []
    if prompt is not None: parts.append(f"Input: ${prompt}/1M tok")
    if completion is not None: parts.append(f"Output: ${completion}/1M tok")
    for k, v in extras.items():
        # e.g., image, audio, cached, etc.
        parts.append(f"{k.capitalize()}: ${v}")
    return ", ".join(parts) if parts else "—"

def _bool(v):
    return "Yes" if bool(v) else "No"

def build_model_markdown(md_id: str) -> str:
    """Generate sidebar Markdown for the selected model."""
    m = find_model(md_id)
    if not m:
        return f"### Model\n`{md_id}`\n\n*No metadata found from `/models`.*"
    # Best-effort extraction across providers
    name = m.get("name") or m.get("id") or md_id
    provider = (m.get("owned_by") or m.get("organization") or m.get("provider") or "—")
    link = m.get("permalink") or f"https://openrouter.ai/models/{m.get('id', md_id)}"
    # Context and pricing
    ctx = (m.get("context_length") or m.get("max_context") or m.get("context") or "—")
    pricing = _fmt_price(m.get("pricing", {}))
    # Modalities & caps
    in_modal = ", ".join(m.get("input_modalities", []) or m.get("modality", []) or []) or "text"
    out_modal = ", ".join(m.get("output_modalities", []) or []) or "text"
    params = m.get("supported_parameters", [])
    caps = []
    if m.get("supports_images") or ("image" in (m.get("input_modalities") or [])):
        caps.append("Image-in")
    if m.get("tools") or ("tools" in params):
        caps.append("Tools")
    if m.get("function_calling") or ("function_calling" in params):
        caps.append("Func-calls")
    if m.get("json_output") or ("response_format" in params):
        caps.append("JSON-mode")
    caps_str = ", ".join(caps) if caps else "—"
    # Notes
    notes = m.get("description") or ""
    if notes:
        # Trim long blurbs
        notes = notes.strip()
        if len(notes) > 560:
            notes = notes[:560].rstrip() + "…"
    else:
        notes = "—"

    # Markdown block
    md = []
    md.append(f"### Model\n[`{name}`]({link})")
    md.append("")
    md.append("**Provider:** " + f"`{provider}`")
    md.append("")
    md.append("**Context length:** " + f"`{ctx}` tokens")
    md.append("")
    md.append("**Pricing:** " + (pricing or "—"))
    md.append("")
    md.append("**Modalities:**")
    md.append(f"- Input: `{in_modal}`")
    md.append(f"- Output: `{out_modal}`")
    md.append("")
    if params:
        # compact list
        short = ", ".join(sorted(set(params))[:12])
        more = "" if len(params) <= 12 else f" (+{len(params)-12} more)"
        md.append(f"**Supported params:** {short}{more}")
        md.append("")
    md.append("**Capabilities:** " + caps_str)
    md.append("")
    md.append("**Notes:**")
    md.append(notes)
    md.append("")
    md.append("> Data from OpenRouter **Models API**. Fields vary by provider. ")
    return "\n".join(md)

# =========================================================
# Chat streaming to OpenRouter  ???????????????????????????
# =========================================================

def stream_openrouter(model, system_prompt, messages, temperature, max_tokens, top_p, seed):
    url = f"{BASE_URL}/chat/completions"
    chat = []
    if system_prompt and system_prompt.strip():
        chat.append({"role": "system", "content": system_prompt.strip()})
    chat.extend(messages)
    payload = {
        "model": model,
        "messages": chat,
        "temperature": float(temperature) if temperature is not None else None,
        "max_tokens": int(max_tokens) if max_tokens else None,
        "top_p": float(top_p) if top_p else None,
        "seed": int(seed) if seed not in (None, "", 0) else None,
        "stream": True,
    }
    with requests.post(url, headers=HEADERS, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                data = raw[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = (
                        chunk.get("choices", [{}])[0]
                              .get("delta", {})
                              .get("content", "")
                    )
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue

def reply_fn(history, model, system_prompt, temperature, max_tokens, top_p, seed):
    return stream_openrouter(model, system_prompt, history, temperature, max_tokens, top_p, seed)

# =========================================================
# UI  ?????????????????????????????????????????????????????
# =========================================================

MODEL_CHOICES = list_model_choices() or list(DEFAULT_MODEL_CHOICES)

with gr.Blocks(title="Agent Lab · OpenRouter", theme="soft") as demo:
    gr.Markdown("# Agent Lab · OpenRouter\nQuickly A/B system prompts and models (streaming)")

    # ?? Sidebar: Model Metadata
    with gr.Sidebar(label="Model Metadata", open=True, width=320):
        gr.Markdown("## Model Metadata")
        sidebar_model_id = gr.Textbox(label="Model ID (readonly)", interactive=False)
        sidebar_md = gr.Markdown(value="*(select a model to load metadata)*")
        refresh_btn = gr.Button("? Refresh Models")

    # ?? Main controls
    with gr.Row():
        model_dd = gr.Dropdown(
            choices=MODEL_CHOICES,
            value=MODEL_CHOICES[0][1] if MODEL_CHOICES else "openrouter/auto",
            label="Model (OpenRouter)"
        )
        temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="temperature")
        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="top_p")
    with gr.Row():
        max_tokens = gr.Number(value=1024, precision=0, label="max_tokens (response cap)")
        seed = gr.Number(value=0, precision=0, label="seed (0/blank or 0 = random)")

    system_prompt = gr.Textbox(
        label="System Instructions",
        placeholder="e.g., You are a meticulous code-review copilot. Use evidence-tagged findings.",
        lines=5,
    )

    def chat_fn(messages, model, system, temp, max_tok, top_p_val, seed_val):
        return reply_fn(messages, model, system, temp, max_tok, top_p_val, seed_val)

    chat = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        fill_height=True,
        autofocus=True,
        submit_btn="Send",
        stop_btn="Stop",
        additional_inputs=[model_dd, system_prompt, temperature, max_tokens, top_p, seed],
    )

    # ?? Wiring: update sidebar when model changes
    def update_sidebar(model_id):
        try:
            md = build_model_markdown(model_id)
        except Exception as e:
            md = f"### Model\n`{model_id}`\n\n*Error parsing metadata.*"
        return gr.Textbox(value=model_id), gr.Markdown(value=md)

    model_dd.change(fn=update_sidebar, inputs=model_dd, outputs=[sidebar_model_id, sidebar_md])

    # ?? Refresh button: bust cache, reload choices and sidebar
    def refresh_models(current_model):
        fetch_models_raw.cache_clear()
        new_choices = list_model_choices() or list(DEFAULT_MODEL_CHOICES)
        # If current selection disappeared, fall back to first safe choice.
        new_value = (
            current_model
            if any(v == current_model for _, v in new_choices)
            else (new_choices[0][1] if new_choices else DEFAULT_MODEL_ID)
        )
        md = build_model_markdown(new_value)
        return (
            gr.Dropdown(choices=new_choices, value=new_value),
            gr.Textbox(value=new_value),
            gr.Markdown(value=md),
        )

    refresh_btn.click(
        fn=refresh_models,
        inputs=model_dd,
        outputs=[model_dd, sidebar_model_id, sidebar_md],
    )

if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable.")
    demo.launch()
