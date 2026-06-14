# Coding Agent CLI Harness

**Hard-typed, semantically flexible. Every line earns its keep.**

This harness defines a typed semantic state substrate: user input is parsed into control signals, and those controls determine which operations are allowed over the current project state.

The goal is not to make an LLM “act like” a project manager, planner, reviewer, or implementer. The goal is to make project state explicit, validate every transition, and use model calls only as bounded operators inside a governed workflow.

The agent is not the system. The system is the governed recurrence between typed state, control input, validated transition, and feedback.
reference repo: https://github.com/duck-lint/coding-agent-harness

## Source Shape
```
operator supplies:
  task text
  selected agent contract

agent contract supplies:
  provider
  model
  input policy
  output policy
  schema refs

runtime config supplies:
  default runtime budget

compilers/runners handle:
  context resolution
  provider payload rendering
  provider execution
  output validation
```

## Package Plan Route

```powershell
python -m harness plan "Review the current project trajectory."
```

This is the operator-facing Project Manager route. The built-in PM contract
supplies provider, model, input policy, output policy, and schema refs. The
package CLI supplies task authority and orchestrates the already-proven
compiler, provider-runner, and output-validation seams.

If you need the lower-level selector directly, `--agent` still works:

```powershell
python -m harness --agent harness/agents/project_manager.agent.json "Review the current project trajectory."
```

## Provider-neutral compiler scripts

```powershell
python harness/project_spec/static_context_packet_compiler.py
```

This scaffold is currently exposed through directly executable compiler scripts,
not through the package CLI.

Compiles every JSON source declared by
`harness/project_spec/static_context_packet.manifest.json` into the `sources`
map in `static_context_packet.json`. Source selection, requiredness,
cardinality, scope, and paths come from the manifest. The compiler parses the
declared JSON without applying source-specific Pydantic validation or
normalization.

Manifest paths are repository-root relative. `harness_global` entries resolve
from `--harness-root`; `target_repo` entries resolve from
`--target-repo-root`.

```powershell
python harness/agents/agent_context_compiler.py --agent harness/agents/project_manager.agent.json
```

Compiles the selected agent contract plus inputs declared by that agent's
`agent_input_policy` into `agent_context_packet.json`. For PM,
`static_context_packet` is resolved because the PM agent policy declares it.

Build a provider-neutral API call packet for a direct task:

```powershell
python harness/runtime/api_call_packet_builder.py --task "Review the current project trajectory." --direct
```

This collects git context automatically from `--repo-root`. Use
`--no-git-context` to disable that for debugging or fixture work.

Build a provider-neutral API call packet for a direct task with explicit static
context attached as supplementary context:

```powershell
python harness/runtime/api_call_packet_builder.py --task "Review the current project trajectory." --direct --static-context harness/runs/static_context_packet.json
```

Build a provider-neutral API call packet for an agent-routed task:

```powershell
python harness/runtime/api_call_packet_builder.py --task "Review the current project trajectory." --agent harness/agents/project_manager.agent.json
```

Git context is packet-level ambient context and is collected automatically from
`--repo-root` for both direct and agent-routed calls. It is not declared in
`agent_input_policy`.

Resolve effective model selection:

```powershell
python harness/runtime/model_resolution.py --api-call-packet harness/runs/api_call_packet.json --provider-runtime-policy harness/runtime/provider_runtime.policy.json
```

This records which provider/model would be used before provider-specific
payload rendering. Agent-routed calls use the selected agent contract model as
primary authority. Direct calls use the provider runtime policy default.
Runtime budget constrains token behavior and does not select the model. This
artifact is inspectable, but it is not required for agent-routed OpenAI payload
rendering.

Render OpenAI provider payload for an agent-routed packet:

```powershell
python harness/providers/openai/openai_response_payload_compiler.py --api-call-packet harness/runs/api_call_packet.json --output harness/runs/provider_payload.json
```

For agent-routed calls, `request.model` is rendered directly from
`api_call_packet.agent_context_packet.agent_contract.model`.

Render OpenAI provider payload for a direct packet with an explicit model:

```powershell
python harness/providers/openai/openai_response_payload_compiler.py --api-call-packet harness/runs/api_call_packet.json --model gpt-5.4-mini --output-schema harness/contracts/ProjectManagerReport.schema.json --output harness/runs/provider_payload.json
```

Or render a direct packet using `provider_runtime.policy.json` only as the
source of `default_direct_model`:

```powershell
python harness/providers/openai/openai_response_payload_compiler.py --api-call-packet harness/runs/api_call_packet.json --provider-runtime-policy harness/runtime/provider_runtime.policy.json --output-schema harness/contracts/ProjectManagerReport.schema.json --output harness/runs/provider_payload.json
```

This emits an inspectable OpenAI Responses API payload. It does not call the
model.
Before any live runner exists, the compiler strips embedded schema compatibility
keys like `$schema` / `$id` and rejects obvious unsupported Structured Outputs
keywords during preflight.

Run OpenAI provider call:

```powershell
python harness/providers/openai/openai_call_runner.py --provider-payload harness/runs/provider_payload.json --output harness/runs/raw_model_response.json
```

This sends the already-rendered OpenAI payload and captures the raw model
response. It does not validate the response as a `ProjectManagerReport`.

Validate Project Manager output:

```powershell
python harness/contracts/project_manager_report_extractor.py --raw-response harness/runs/raw_model_response.json --schema harness/contracts/ProjectManagerReport.schema.json --output harness/runs/project_manager_report.json
```

This extracts `output_text` from the raw OpenAI response, parses it as JSON,
validates it against `ProjectManagerReport.schema.json`, and writes
`project_manager_report.json` plus `project_manager_report.validation.json`
only if validation succeeds.

Compile a repo snapshot from one file:

```powershell
python harness/repo_snapshot/repo_snapshot_compiler.py --repo-root . --path README.md
```

Compile a repo snapshot from globs:

```powershell
python harness/repo_snapshot/repo_snapshot_compiler.py --repo-root . --glob "tests/*.py"
```

Compile all admissible repo files:

```powershell
python harness/repo_snapshot/repo_snapshot_compiler.py --repo-root . --all-admissible
```

Compile all admissible repo files and explicitly include the harness itself:

```powershell
python harness/repo_snapshot/repo_snapshot_compiler.py --repo-root . --all-admissible --include-harness
```

Agent-routed calls receive repo snapshots only when the selected `.agent.json`
declares `repo_snapshot_packet` in `agent_input_policy`.

For agent-routed repo selection, declare `repo_snapshot_packet` with a
`resolution` object in the agent contract, for example `mode: "paths"` plus the
requested `paths`. Git context is separate from repo snapshot selection and
does not need an agent-policy entry.

Direct calls can attach an already compiled repo snapshot as supplementary
context.

By default, repo snapshots exclude the `harness/` tree across `paths`, `globs`,
and `all_admissible`. Use `--include-harness` only for self-hosting or
meta-level harness development.

These scripts emit the current bounded-slice artifacts:

- `static_context_packet.json`
- `agent_context_packet.json`
- `api_call_packet.json`
- `repo_snapshot_packet.json`
- `effective_model_selection.json`
- `provider_payload.json`
- `raw_model_response.json`
- `project_manager_report.json`
- `project_manager_report.validation.json`

The package route above is the normal operator entrypoint. The PM route writes
the validated report plus its validation evidence sidecar. The direct scripts
remain available for seam-level debugging.
