# Plugin Contract

A plugin is a small adapter implementing `PipelinePlugin`.

It declares:

- a stable, versioned `key`;
- a human-readable name;
- an `output_family` used to locate the resulting run;
- a `workflow_stage` and optional prerequisite stage;
- the device roles allowed to execute it;
- typed parameters with defaults and groups;
- a command builder.

The API applies role and prerequisite checks before queueing. The command builder
must use validated parameters and return an immutable command tuple. Pipeline
logic belongs in the source package, not in the adapter.

To register a plugin:

1. Add the adapter under `backend/battryml/plugins/`.
2. Register it in `build_default_registry()`.
3. Extend the catalog scanner when the plugin produces an analysis run. Pure
   preprocessing stages expose filesystem readiness instead.
4. Add a normalized data service only when its output is not SOH-compatible.
5. Add role, command, scanner, and API tests as appropriate.
