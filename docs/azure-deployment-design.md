# Azure deployment design — prepared, not deployed

This repository includes opt-in scripts for a small Azure Container Apps demo
and describes a plausible production deployment direction for the portfolio
prototype. No Azure resources were created or verified while preparing these
files, and no automatic Azure deployment pipeline, Terraform/Bicep, managed
identities, private endpoints, or cloud monitoring are included.

## Demo deployment

The manual deployment script uses resource group `fincrime-demo-rg` in
`uksouth` and creates or updates the `fincrime-api` Container App from the
prebuilt public image at
`ghcr.io/joffie89/agentic-fincrime-investigation:latest`. It configures external
ingress on port 8000, stores `API_KEY` and `OPENAI_API_KEY` as Container Apps
secrets, and injects them through secret references. The app uses OpenAI
assessment and embeddings with the bundled `data/policy_index_openai.json`
index. Scaling is limited to zero or one replica.

The `publish-image.yml` GitHub Actions workflow runs pytest before building and
publishing `latest` and commit-SHA tags. Its Docker context excludes `.env` and
`.env.*`. After the first publication, the GHCR package must be changed to
**Public** once in GitHub package settings so Container Apps can pull it without
registry credentials.

Prerequisites are an Azure subscription and an authenticated Azure CLI session.
From the repository root, run:

```powershell
.\scripts\deploy_azure.ps1
```

Pass `-Image ghcr.io/joffie89/agentic-fincrime-investigation:<commit-sha>` to
deploy an immutable published build instead of `latest`.

The prompts use secure input; the script does not echo secret values, write them
to local files, or place them in command-history text. It prints the app URL
after Azure has accepted the final configuration. To remove the demo resources,
run:

```powershell
.\scripts\teardown_azure.ps1
```

The demo deliberately sets `PERSISTENCE_BACKEND=memory` and can scale to zero.
Investigation and audit records therefore disappear whenever the replica stops
or restarts, including scale-to-zero events. This low-cost demo profile is not
the production topology described below.

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

