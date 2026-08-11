# Security Policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub
issues.**

Use GitHub's private vulnerability reporting instead:
<https://github.com/kyleADHD/AirCanvas/security/advisories/new>

Include what you can: affected version, a reproduction, and impact. You can
expect an acknowledgement within a few days; please allow a reasonable
disclosure window before publishing details.

## Threat model notes for reporters

Areas of particular interest:

- **Checkpoint parsing.** AirCanvas parses safetensors headers with its own
  reader (`sharding/`) and treats model checkpoints as untrusted input.
  Anything that turns a malicious checkpoint into memory corruption, path
  traversal (shard cache writes), or code execution is in scope.
- **Shard-cache integrity.** Crash-safe writes rely on temp-file → rename +
  `.done` markers. Bypasses that make a partial/tampered shard load as valid
  are in scope.
- **Not in scope:** prompts and generated content (AirCanvas does not filter
  them — it is a memory/placement layer over diffusers), and vulnerabilities
  in upstream dependencies (report those upstream, though a heads-up is
  appreciated if AirCanvas's usage makes one exploitable).
