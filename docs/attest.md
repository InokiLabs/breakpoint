# Signed reports

A BreakPoint report (`breakpoint.report/v1`) can be signed with an ed25519 key so
that anyone holding the signer's public key can check it, without ever seeing the
policy weights.

```
pip install "breakpoint-mjx[attest]"                 # adds the cryptography package
breakpoint keys new --out ~/.breakpoint/keys         # breakpoint.key (0600) + breakpoint.pub
breakpoint sign report.json --key ~/.breakpoint/keys/breakpoint.key [--signer self|inoki] [--out signed.json]
breakpoint verify report.json --pubkey breakpoint.pub [--policy ./checkpoint.pkl]
```

`verify` prints `PASS: ...` or `FAIL: <reason>` and exits 1 on FAIL (2 on a usage
error such as a missing file). `sign` rewrites the report in place unless `--out`
is given.

## What is signed

The canonical JSON of the report: UTF-8, keys sorted at every level, separators
`(",", ":")`, with the top-level `attestation` field removed. The signature is
ed25519 over those exact bytes. The attestation block added to the report is:

```json
{"alg": "ed25519", "key_id": "<first 16 hex of sha256(raw 32-byte public key)>",
 "signer": "self | inoki", "payload_sha256": "<sha256 of the canonical bytes>",
 "signature": "<base64>"}
```

Whitespace and key order in the file do not matter. Any change to a value, to list
order, or an added or removed field does. Signing a report that already has an
attestation replaces it. `verify` also requires the attestation's `signer` to match
the signed `run.runner` field, so the label cannot be changed after signing.

`--policy` is for teams that hold the weights: it additionally checks that
`policy.sha256` in the report equals the sha256 of a local checkpoint file.

## What a valid signature proves

- The holder of that private key produced this exact report (in canonical form),
  and nobody edited it afterwards.
- The report names a policy by its sha256, so it is bound to one specific
  checkpoint even though the checkpoint itself is never shared.

## What it does NOT prove

- **That the simulation transfers to hardware.** A report is a statement about a
  fixed simulated battery. Passing cells in MJX do not mean the robot passes on
  real ground.
- **That the signer ran it honestly, unless the signer is Inoki.** A `self`
  signature only says the team produced the file. A team could edit a report before
  signing it, or run a different checkpoint than the one it names. Only an `inoki`
  signature, checked against Inoki's published key (`docs/keys/`), means Inoki Labs
  ran the battery itself.
- **Who the signer is**, beyond "whoever holds this key". Trust comes from where
  you got the public key, not from the `signer` label.
- That the battery is a good test. The report records `battery.sha256` so you can
  check which battery was used; judging the battery is up to you.

## Keys

- Private keys are written only outside a git work tree (`keys new` refuses a path
  inside one), are never printed, and are never overwritten. They are unencrypted
  PEM (PKCS8) with mode 0600 where the OS supports it. On Windows the mode is not
  enforced, so protect the folder with its ACLs.
- Public keys are PEM (SubjectPublicKeyInfo); `openssl pkey -pubin -in x.pub -text`
  reads them.
- Inoki's public key is published at `/keys/inoki.pub` on the site and in
  `docs/keys/`; see `docs/keys/README.md`.
