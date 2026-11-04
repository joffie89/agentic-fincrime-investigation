[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ApplicationDisplayName = "fincrime-github-deploy"
$FederatedCredentialName = "github-main"
$FederatedCredentialIssuer = "https://token.actions.githubusercontent.com"
$FederatedCredentialSubject = `
    "repo:joffie89/agentic-fincrime-investigation:ref:refs/heads/main"
$ImmutableFederatedCredentialName = "github-main-immutable"
$ImmutableFederatedCredentialSubject = `
    "repo:joffie89@54671870/agentic-fincrime-investigation@1379635336:ref:refs/heads/main"
$FederatedCredentialAudience = "api://AzureADTokenExchange"
$ResourceGroup = "fincrime-demo-rg"
$RoleName = "Container Apps Contributor"

function Invoke-AzureCli {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments
    )

    # Keep native stderr separate from stdout. Do not use 2>&1 or *> because
    # Azure CLI writes non-fatal warnings to stderr and Windows PowerShell 5.1
    # can otherwise mix those warnings into captured JSON.
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $OutputLines = @(& az @Arguments)
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        $CommandName = ($Arguments | Select-Object -First 3) -join " "
        throw "Azure CLI command 'az $CommandName' failed with exit code $ExitCode."
    }

    return $OutputLines
}

function Invoke-AzureCliJson {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments
    )

    $OutputText = (Invoke-AzureCli -Arguments $Arguments) -join [Environment]::NewLine
    if ([string]::IsNullOrWhiteSpace($OutputText)) {
        throw "Azure CLI returned no JSON output."
    }

    try {
        return ($OutputText | ConvertFrom-Json | ForEach-Object { $_ })
    }
    catch {
        throw "Azure CLI returned invalid JSON: $($_.Exception.Message)"
    }
}

function Ensure-FederatedCredential {
    param(
        [Parameter(Mandatory)]
        [string] $ApplicationId,

        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [object[]] $ExistingCredentials,

        [Parameter(Mandatory)]
        [string] $Name,

        [Parameter(Mandatory)]
        [string] $Issuer,

        [Parameter(Mandatory)]
        [string] $Subject,

        [Parameter(Mandatory)]
        [string] $Audience,

        [Parameter(Mandatory)]
        [string] $Description
    )

    $MatchingCredentials = @(
        $ExistingCredentials | Where-Object {
            $Audiences = @($_.audiences)
            [string]::Equals(
                [string] $_.name,
                $Name,
                [StringComparison]::Ordinal
            ) -and
            [string]::Equals(
                [string] $_.issuer,
                $Issuer,
                [StringComparison]::Ordinal
            ) -and
            [string]::Equals(
                [string] $_.subject,
                $Subject,
                [StringComparison]::Ordinal
            ) -and
            $Audiences.Count -eq 1 -and
            [string]::Equals(
                [string] $Audiences[0],
                $Audience,
                [StringComparison]::Ordinal
            )
        }
    )

    if ($MatchingCredentials.Count -gt 0) {
        Write-Host "Reusing the GitHub Actions federated credential '$Name'."
        return
    }

    $ConflictingCredentials = @(
        $ExistingCredentials | Where-Object {
            [string]::Equals(
                [string] $_.name,
                $Name,
                [StringComparison]::Ordinal
            )
        }
    )
    if ($ConflictingCredentials.Count -gt 0) {
        throw "Federated credential '$Name' exists with different trust settings."
    }

    Write-Host "Creating the GitHub Actions federated credential '$Name'..."
    $FederatedCredential = @{
        name = $Name
        issuer = $Issuer
        subject = $Subject
        description = $Description
        audiences = @($Audience)
    }
    $FederatedCredentialJson = $FederatedCredential | ConvertTo-Json -Depth 4 -Compress
    $FederatedCredentialFile = Join-Path `
        ([IO.Path]::GetTempPath()) `
        ("fincrime-github-oidc-{0}.json" -f [Guid]::NewGuid().ToString("N"))
    try {
        # A temporary JSON file avoids Windows PowerShell 5.1 stripping the
        # JSON quotes while marshalling arguments to the native az process.
        [IO.File]::WriteAllText(
            $FederatedCredentialFile,
            $FederatedCredentialJson,
            (New-Object Text.UTF8Encoding($false))
        )
        [void](Invoke-AzureCliJson -Arguments @(
            "ad", "app", "federated-credential", "create",
            "--id", $ApplicationId,
            "--parameters", $FederatedCredentialFile,
            "--output", "json"
        ))
    }
    finally {
        if (Test-Path -LiteralPath $FederatedCredentialFile -PathType Leaf) {
            Remove-Item -LiteralPath $FederatedCredentialFile -Force
        }
        $FederatedCredentialFile = $null
        $FederatedCredentialJson = $null
        $FederatedCredential = $null
    }
}

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI 2.91 or later is required. Install it and run 'az login'."
}

$Account = Invoke-AzureCliJson -Arguments @(
    "account", "show",
    "--output", "json"
)
$SubscriptionId = ([string] $Account.id).Trim()
$TenantId = ([string] $Account.tenantId).Trim()
if (
    [string]::IsNullOrWhiteSpace($SubscriptionId) -or
    [string]::IsNullOrWhiteSpace($TenantId)
) {
    throw "No active Azure subscription was found. Run 'az login' first."
}

$Applications = @(Invoke-AzureCliJson -Arguments @(
    "ad", "app", "list",
    "--display-name", $ApplicationDisplayName,
    "--output", "json"
))
$MatchingApplications = @(
    $Applications | Where-Object {
        [string]::Equals(
            [string] $_.displayName,
            $ApplicationDisplayName,
            [StringComparison]::Ordinal
        )
    }
)

if ($MatchingApplications.Count -gt 1) {
    throw "More than one Entra application is named '$ApplicationDisplayName'."
}

if ($MatchingApplications.Count -eq 0) {
    Write-Host "Creating Microsoft Entra application '$ApplicationDisplayName'..."
    $Application = Invoke-AzureCliJson -Arguments @(
        "ad", "app", "create",
        "--display-name", $ApplicationDisplayName,
        "--sign-in-audience", "AzureADMyOrg",
        "--output", "json"
    )
}
else {
    Write-Host "Reusing Microsoft Entra application '$ApplicationDisplayName'."
    $Application = $MatchingApplications[0]
}

$ClientId = ([string] $Application.appId).Trim()
if ([string]::IsNullOrWhiteSpace($ClientId)) {
    throw "The Entra application is missing its application (client) ID."
}

$ServicePrincipals = @(Invoke-AzureCliJson -Arguments @(
    "ad", "sp", "list",
    "--filter", "appId eq '$ClientId'",
    "--output", "json"
))
if ($ServicePrincipals.Count -gt 1) {
    throw "More than one service principal was found for client ID '$ClientId'."
}

if ($ServicePrincipals.Count -eq 0) {
    Write-Host "Creating the service principal..."
    $ServicePrincipal = Invoke-AzureCliJson -Arguments @(
        "ad", "sp", "create",
        "--id", $ClientId,
        "--output", "json"
    )
}
else {
    Write-Host "Reusing the service principal."
    $ServicePrincipal = $ServicePrincipals[0]
}

$ServicePrincipalObjectId = ([string] $ServicePrincipal.id).Trim()
if ([string]::IsNullOrWhiteSpace($ServicePrincipalObjectId)) {
    throw "The service principal is missing its object ID."
}

$FederatedCredentials = @(Invoke-AzureCliJson -Arguments @(
    "ad", "app", "federated-credential", "list",
    "--id", $ClientId,
    "--output", "json"
))
Ensure-FederatedCredential `
    -ApplicationId $ClientId `
    -ExistingCredentials $FederatedCredentials `
    -Name $FederatedCredentialName `
    -Issuer $FederatedCredentialIssuer `
    -Subject $FederatedCredentialSubject `
    -Audience $FederatedCredentialAudience `
    -Description "GitHub Actions deployment from the repository name and main branch"

Ensure-FederatedCredential `
    -ApplicationId $ClientId `
    -ExistingCredentials $FederatedCredentials `
    -Name $ImmutableFederatedCredentialName `
    -Issuer $FederatedCredentialIssuer `
    -Subject $ImmutableFederatedCredentialSubject `
    -Audience $FederatedCredentialAudience `
    -Description "GitHub Actions deployment from immutable repository IDs and main branch"

$ResourceGroupScope = "/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup"
$RoleAssignments = @(Invoke-AzureCliJson -Arguments @(
    "role", "assignment", "list",
    "--assignee-object-id", $ServicePrincipalObjectId,
    "--role", $RoleName,
    "--scope", $ResourceGroupScope,
    "--fill-principal-name", "false",
    "--fill-role-definition-name", "false",
    "--output", "json"
))

if ($RoleAssignments.Count -eq 0) {
    Write-Host "Assigning '$RoleName' at resource-group scope..."
    [void](Invoke-AzureCliJson -Arguments @(
        "role", "assignment", "create",
        "--assignee-object-id", $ServicePrincipalObjectId,
        "--assignee-principal-type", "ServicePrincipal",
        "--role", $RoleName,
        "--scope", $ResourceGroupScope,
        "--output", "json"
    ))
}
else {
    Write-Host "Reusing the existing '$RoleName' role assignment."
}

Write-Host ""
Write-Host "Role: $RoleName"
Write-Host "Scope: $ResourceGroupScope"
Write-Host "This role can update Container Apps without granting broad Contributor access."
Write-Host ""
Write-Host "Add these values as GitHub repository secrets:"
Write-Host "AZURE_CLIENT_ID=$ClientId"
Write-Host "AZURE_TENANT_ID=$TenantId"
Write-Host "AZURE_SUBSCRIPTION_ID=$SubscriptionId"
