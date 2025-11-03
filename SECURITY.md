# Security Policy

Thank you for helping to keep **SFT2D** and its users safe.
This document describes how to report security issues and outlines our security support policy.

---

## 🛡️ Supported Versions

The following versions of **SFT2D** currently receive security updates and patches:

| Version                   | Supported | Notes                 |
| ------------------------- | --------- | --------------------- |
| main (development branch) | ✅         | Actively maintained   |
| v1.x                      | ✅         | Latest stable release |
| v0.x                      | ❌         | No longer supported   |

If you are using an older version, please upgrade to the latest release to ensure you receive fixes and improvements.

---

## 🐞 Reporting a Vulnerability

If you discover a security vulnerability in **SFT2D**, please **do not** open a public issue.
Instead, report it responsibly via email to:

📧 **[sdash@nso.edu](mailto:sdash@nso.edu)**

Please include:

* A description of the vulnerability and its potential impact
* Steps to reproduce (if applicable)
* Any suggested fixes or mitigations
* Your contact information (optional, for follow-up)

You will receive an acknowledgment within **3–5 business days**, and updates on the progress of the fix thereafter.

---

## 🔐 Handling Process

1. The maintainer reviews the report and verifies the issue.
2. A private branch is created for investigation and mitigation.
3. Once fixed, a new release is issued, and users are notified via the release notes.
4. Credit is given to the reporter if desired (unless anonymity is requested).

---

## ⚙️ Security Best Practices for Contributors

To maintain a secure codebase:

* Avoid committing credentials, tokens, or sensitive data.
* Do not include external dependencies from untrusted sources.
* Review code for potential injection, deserialization, or resource exhaustion risks.
* Follow principle of least privilege for all I/O and file operations.
* Ensure tests do not expose internal data or run unsafe shell commands.

---

## 🧩 Responsible Disclosure

We appreciate responsible disclosure — please allow time for the maintainer to investigate and patch before publicly disclosing the issue.
Premature public disclosure may put users at risk.

---

## 🤝 Acknowledgments

We thank all researchers, developers, and users who contribute to keeping **SFT2D** secure and reliable.
Your vigilance and cooperation are invaluable to maintaining a trustworthy scientific software ecosystem.

---

*Last updated: November 2025*
