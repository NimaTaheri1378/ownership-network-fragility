# GitHub release checklist

Before pushing a public release:

- [ ] Confirm raw WRDS/vendor data are not staged.
- [ ] Confirm derived security-level panels are not staged.
- [ ] Confirm local logs and machine-specific schema contracts are not staged.
- [ ] Confirm no credentials, private keys, or tokens are staged.
- [ ] Confirm public tables are aggregate-only and contain no security-level identifiers.
- [ ] Confirm public figures are aggregate/research summaries only.
- [ ] Confirm staged files contain no private local paths or cluster hostnames.
