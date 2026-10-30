# Security

This is a research prototype. Use release 0.2 or newer; the 0.1 inference pins
included dependencies with known advisories. Updates require compatibility tests.

Do not attach recordings, transcripts, tokens or personal paths to public issues.
For a suspected vulnerability, use GitHub's private reporting feature if available;
otherwise open a minimal issue requesting a private contact channel without exploit
details or sensitive attachments. Do not post secrets to demonstrate a finding.

The demo accepts short PCM WAV uploads, limits size/queue/runtime, and deletes
request files when processing ends. It has no accounts, persistent analysis store,
public file-serving endpoint or transcript export. Provider logs and infrastructure
are outside its retention guarantees. Run it locally for sensitive research data.

The public app is not a multi-tenant production service. Authentication, infrastructure
rate limiting, operational monitoring and a provider-specific privacy review would
be needed before broader deployment. Dependency scans detect known advisories;
neither they nor secret scans prove an absence of vulnerabilities or personal data.

See [review scope and findings](docs/security-review.md).
