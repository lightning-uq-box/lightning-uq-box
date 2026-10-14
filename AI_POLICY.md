# AI contribution policy

AI tools may be used to prepare contributions to Lightning-UQ-Box. Human review
and understanding are required to protect contribution quality and maintainer
time. This policy applies to code, tests, documentation, issues, pull requests,
and comments.

## Human supervision

Every contribution must have a responsible human who reviews and understands all
submitted changes before opening a pull request, including a draft pull request.
The contributor must verify relevant testing or other validation, be able to
explain the implementation, and address maintainer feedback. These requirements
also apply to subsequent updates.

We do not accept fully autonomous contributions. Giving an agent an initial
prompt, allowing it to open a pull request, or relying solely on passing automated
checks does not satisfy the requirement for human supervision. An agent may
prepare changes locally, but a human must review the specific changes and approve
their submission before the agent opens a pull request.

## Disclosure and communication

Disclose material AI assistance in the pull request description, including the
tools used and their role in producing code, tests, or documentation. Routine
spelling corrections and translation do not require disclosure. Prompt transcripts
are not required.

Describe the problem, implementation, and validation in your own words. Do not
delegate responses to maintainers to an unsupervised agent. AI-assisted
proofreading and translation are permitted.

## Quality and licensing

Keep contributions focused and concise. Verify generated claims, references,
examples, and test results, and remove unrelated changes or unnecessary complexity
before submission. Clearly state any validation that was not performed.

Contributors must have permission to submit their contributions under the
project's Apache License 2.0 and preserve any required third-party attribution.
AI assistance does not remove this requirement.

## Enforcement and approved automation

Maintainers may request clarification or close contributions that do not meet
this policy without detailed review. Missing disclosure can be corrected; repeated
violations may lead to restrictions on further contributions. Enforcement is based
on the contribution and the contributor's engagement, rather than assumptions
about writing style or the presence of an AI disclosure marker alone.

Maintainer-approved repository automation, such as the configured Dependabot
updates, is exempt within its approved scope. This exception does not authorize
unsolicited autonomous AI contributions.

This policy is informed by the [TorchGeo AI policy](https://github.com/torchgeo/governance/blob/main/AI-POLICY.md)
and the [PyTorch AI policy](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md).
