# Publication scope

The release includes curated source, tests using synthetic fixtures, configuration,
small invented table examples, aggregate historical metrics and engineering docs.

Excluded from the repository and its history:

- Research recordings, videos and extracted audio.
- Private human labels, observer forms and transcribed research utterances.
- Administrative documents and identified research reports.
- Local logs, model weights, caches, generated tables and temporary renderings.
- Personal absolute paths, credentials and authentication helpers.

The original research folder is left in place and is not edited, renamed or moved by
the migration. The curated copy has an independent Git root and does not use an
ancestor Git repository. The original inventory contains 127 files. Local preservation
snapshots and private verification logs are kept outside the published repository.

`scripts/audit_release.py` checks candidate and tracked files for excluded types,
large files, personal paths and common secret signatures. This limited check
complements review and is not a guarantee against every possible sensitive item.

## Ownership and license

No blanket license is assigned while institutional code ownership and permission for
reuse remain unconfirmed. Public portfolio publication of the curated work is
authorized by its contributor; that does not automatically grant redistribution or
commercial reuse rights. Obtain the appropriate research-owner confirmation before
adding a permissive license. Upstream checkpoints and datasets have separate terms.

Do not attach private recordings or annotations to public issues or pull requests.
