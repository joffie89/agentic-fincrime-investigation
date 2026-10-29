[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string] $Image = "ghcr.io/joffie89/agentic-fincrime-investigation:latest"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ResourceGroup = "fincrime-demo-rg"
$Location = "uksouth"
$AppName = "fincrime-api"
$TargetPort = 8000
$ApiKeySecretName = "api-key"
$OpenAiApiKeySecretName = "openai-api-key"

function Invoke-AzureCli {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments
    )

    & az @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed with exit code $LASTEXITCODE."
    }
}

function Set-ContainerAppSecret {
    param(
        [Parameter(Mandatory)]
        [string] $SecretName,

        [Parameter(Mandatory)]
        [System.Security.SecureString] $SecureValue
    )

    $SecretPointer = [IntPtr]::Zero
    $PlainValue = $null
    $SecretAssignment = $null
    $SecretArguments = $null

    try {
        $SecretPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
        $PlainValue = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($SecretPointer)
        $SecretAssignment = "$SecretName=$PlainValue"
        $SecretArguments = @(
            "containerapp", "secret", "set",
            "--name", $AppName,
            "--resource-group", $ResourceGroup,
            "--secrets", $SecretAssignment,
            "--only-show-errors",
            "--output", "none"
        )

        # Suppress all CLI output for the only call whose in-memory arguments contain a secret.
        & az @SecretArguments *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to store Container Apps secret '$SecretName'."
        }
    }
    finally {
        if ($null -ne $SecretArguments) {
            [Array]::Clear($SecretArguments, 0, $SecretArguments.Length)
        }
        $SecretAssignment = $null
        $PlainValue = $null
        if ($SecretPointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($SecretPointer)
        }
    }
}

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI ('az') is required. Install it and run 'az login' before this script."
}

Invoke-AzureCli -Arguments @(
    "account", "show",
    "--only-show-errors",
    "--output", "none"
)

$ApiKey = $null
$OpenAiApiKey = $null
try {
    $ApiKey = Read-Host -Prompt "API_KEY" -AsSecureString
    $OpenAiApiKey = Read-Host -Prompt "OPENAI_API_KEY" -AsSecureString

    if ($ApiKey.Length -lt 20 -or $OpenAiApiKey.Length -lt 20) {
        throw "Key looks too short - paste with right-click in a normal PowerShell window."
    }

    $ApiKey.MakeReadOnly()
    $OpenAiApiKey.MakeReadOnly()

    # A Container App must exist before app-level secrets can be added. This random,
    # short-lived bootstrap value lets the first revision start without a .env file;
    # the final update replaces it with the Container Apps secret reference.
    $BootstrapApiKey = "bootstrap-$([Guid]::NewGuid().ToString('N'))"
    try {
        Invoke-AzureCli -Arguments @(
            "containerapp", "up",
            "--name", $AppName,
            "--resource-group", $ResourceGroup,
            "--location", $Location,
            "--image", $Image,
            "--ingress", "external",
            "--target-port", $TargetPort.ToString(),
            "--env-vars",
            "API_KEY=$BootstrapApiKey",
            "SERVICE_MODE=local",
            "EMBEDDING_MODE=local",
            "POLICY_INDEX_PATH=data/policy_index.json",
            "PERSISTENCE_BACKEND=memory",
            "LLM_ASSESSMENT_ENABLED=false",
            "--output", "none"
        )
    }
    finally {
        $BootstrapApiKey = $null
    }

    Set-ContainerAppSecret -SecretName $ApiKeySecretName -SecureValue $ApiKey
    Set-ContainerAppSecret -SecretName $OpenAiApiKeySecretName -SecureValue $OpenAiApiKey

    Invoke-AzureCli -Arguments @(
        "containerapp", "update",
        "--name", $AppName,
        "--resource-group", $ResourceGroup,
        "--set-env-vars",
        "API_KEY=secretref:$ApiKeySecretName",
        "OPENAI_API_KEY=secretref:$OpenAiApiKeySecretName",
        "SERVICE_MODE=openai",
        "EMBEDDING_MODE=openai",
        "POLICY_INDEX_PATH=data/policy_index_openai.json",
        "PERSISTENCE_BACKEND=memory",
        "LLM_ASSESSMENT_ENABLED=true",
        "OPENAI_ASSESSMENT_TEMPERATURE=0",
        "--min-replicas", "0",
        "--max-replicas", "1",
        "--only-show-errors",
        "--output", "none"
    )

    $Fqdn = & az containerapp show `
        --name $AppName `
        --resource-group $ResourceGroup `
        --query "properties.configuration.ingress.fqdn" `
        --output tsv `
        --only-show-errors
    if ($LASTEXITCODE -ne 0) {
        throw "Could not retrieve the Container App URL."
    }

    $Fqdn = ([string] $Fqdn).Trim()
    if ([string]::IsNullOrWhiteSpace($Fqdn)) {
        throw "Azure returned an empty Container App hostname."
    }

    Write-Host "Application URL: https://$Fqdn"
}
finally {
    if ($null -ne $ApiKey) {
        $ApiKey.Dispose()
    }
    if ($null -ne $OpenAiApiKey) {
        $OpenAiApiKey.Dispose()
    }
}
