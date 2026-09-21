# Azure deployment design — not implemented

This document describes a plausible production deployment direction for the
portfolio prototype. No Azure resources, deployment pipeline, Terraform/Bicep,
managed identities, private endpoints, or cloud monitoring were created or
verified by this repository.

## Intended topology

- Run the Docker image in Azure Container Apps or Azure App Service with a minimum
  of two instances where availability requirements justify it.
- Use Azure Database for PostgreSQL Flexible Server for investigation audit data,
  with TLS, backups, retention, private networking, and access policies configured
  by the owning organization.
- Use approved Azure OpenAI chat and embedding deployments. Keep deployment names
  in environment configuration and credentials in Azure Key Vault; prefer managed
  identity where the selected Azure service and SDK path support it.
- Store an approved policy corpus and generated vector index in a controlled data
  service. The repository's JSON index is suitable only for the local prototype.
- Export structured application logs, metrics, traces, and alerts to the
  organization's approved Azure Monitor/Application Insights workspace.
- Put an authenticated API gateway in front of the service and enforce identity,
  authorization, throttling, request-size limits, and audit correlation IDs.

## Suggested delivery stages

1. Add versioned PostgreSQL migrations, connection pooling, readiness checks, and
   backup/restore tests.
2. Add workload identity and Key Vault integration without placing secrets in
   images, CI variables, or source control.
3. Add infrastructure as code plus separate development/staging/production
   environments and policy-index promotion controls.
4. Run live Azure compatibility, failure-injection, load, security, and retrieval
   evaluations using synthetic data before any wider trial.
5. Add a real reviewer workflow with authenticated assignment, decision capture,
   reason codes, overrides, and immutable audit history.

## Production safeguards still required

The prototype must not be promoted unchanged. A real system would require model
and rules governance, legal/compliance approval, threat modeling, data protection
impact assessment, red-team testing, monitoring for drift and retrieval failures,
human-review service-level objectives, incident response, and explicit controls
preventing recommendations from directly causing customer-impacting actions.

