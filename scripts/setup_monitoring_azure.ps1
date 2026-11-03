[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ResourceGroup = "fincrime-demo-rg"
$EnvironmentName = "fincrime-api-env"
$ContainerAppName = "fincrime-api"
$ActionGroupName = "fincrime-api-monitoring-ag"
$ActionGroupShortName = "FinCrimeAPI"
$Http5xxAlertName = "fincrime-api-http-5xx"
$LatencyAlertName = "fincrime-api-p95-latency"
$WorkbookDeploymentName = "fincrime-api-monitoring-workbook"
$WorkbookPath = [IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\monitoring\workbook.json")
)

function Invoke-AzureCli {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments
    )

    # Keep native stderr separate from stdout. In particular, do not use 2>&1 or
    # *> because Azure CLI writes non-fatal warnings to stderr and Windows
    # PowerShell 5.1 can otherwise mix those warnings into captured JSON.
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
        $CommandName = ($Arguments | Select-Object -First 2) -join " "
        throw "Azure CLI command 'az $CommandName' failed with exit code $ExitCode."
    }

    return $OutputLines
}

function Invoke-AzureCliText {
    param(
        [Parameter(Mandatory)]
        [string[]] $Arguments
    )

    $OutputText = (Invoke-AzureCli -Arguments $Arguments) -join [Environment]::NewLine
    return $OutputText.Trim()
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
        return $OutputText | ConvertFrom-Json
    }
    catch {
        throw "Azure CLI returned invalid JSON: $($_.Exception.Message)"
    }
}

function New-ScheduledQueryAlert {
    param(
        [Parameter(Mandatory)]
        [string] $Name,

        [Parameter(Mandatory)]
        [string] $DisplayName,

        [Parameter(Mandatory)]
        [string] $Description,

        [Parameter(Mandatory)]
        [string] $Query,

        [Parameter(Mandatory)]
        [int] $Threshold,

        [Parameter(Mandatory)]
        [string] $WorkspaceId,

        [Parameter(Mandatory)]
        [string] $WorkspaceLocation,

        [Parameter(Mandatory)]
        [string] $ActionGroupId
    )

    $Rule = @{
        location = $WorkspaceLocation
        kind = "LogAlert"
        properties = @{
            displayName = $DisplayName
            description = $Description
            severity = 2
            enabled = $true
            evaluationFrequency = "PT15M"
            windowSize = "PT15M"
            scopes = @($WorkspaceId)
            targetResourceTypes = @("Microsoft.OperationalInsights/workspaces")
            criteria = @{
                allOf = @(
                    @{
                        query = $Query
                        timeAggregation = "Count"
                        operator = "GreaterThan"
                        threshold = $Threshold
                        failingPeriods = @{
                            numberOfEvaluationPeriods = 1
                            minFailingPeriodsToAlert = 1
                        }
                    }
                )
            }
            autoMitigate = $true
            actions = @{
                actionGroups = @($ActionGroupId)
            }
        }
    }

    $RuleJson = $Rule | ConvertTo-Json -Depth 20 -Compress
    $RuleFile = Join-Path `
        ([IO.Path]::GetTempPath()) `
        ("fincrime-monitoring-alert-{0}.json" -f [Guid]::NewGuid().ToString("N"))
    try {
        [IO.File]::WriteAllText(
            $RuleFile,
            $RuleJson,
            (New-Object Text.UTF8Encoding($false))
        )
        [void](Invoke-AzureCli -Arguments @(
            "resource", "create",
            "--resource-group", $ResourceGroup,
            "--name", $Name,
            "--resource-type", "Microsoft.Insights/scheduledQueryRules",
            "--api-version", "2023-12-01",
            "--is-full-object",
            "--properties", "@$RuleFile",
            "--output", "none"
        ))
    }
    finally {
        if (Test-Path -LiteralPath $RuleFile -PathType Leaf) {
            Remove-Item -LiteralPath $RuleFile -Force
        }
        $RuleJson = $null
        $Rule = $null
    }
}

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI ('az') is required. Install Azure CLI 2.91 or later and run 'az login'."
}

if (-not (Test-Path -LiteralPath $WorkbookPath -PathType Leaf)) {
    throw "Workbook template not found: $WorkbookPath"
}

$SubscriptionId = Invoke-AzureCliText -Arguments @(
    "account", "show",
    "--query", "id",
    "--output", "tsv"
)
if ([string]::IsNullOrWhiteSpace($SubscriptionId)) {
    throw "No active Azure subscription was found. Run 'az login' first."
}

$WorkspaceCustomerId = Invoke-AzureCliText -Arguments @(
    "containerapp", "env", "show",
    "--name", $EnvironmentName,
    "--resource-group", $ResourceGroup,
    "--query", "properties.appLogsConfiguration.logAnalyticsConfiguration.customerId",
    "--output", "tsv"
)
if (
    [string]::IsNullOrWhiteSpace($WorkspaceCustomerId) -or
    $WorkspaceCustomerId -eq "None" -or
    $WorkspaceCustomerId -eq "null"
) {
    throw "Container Apps environment '$EnvironmentName' is not connected to Log Analytics."
}

$Workspaces = @(Invoke-AzureCliJson -Arguments @(
    "monitor", "log-analytics", "workspace", "list",
    "--output", "json"
))
$MatchingWorkspaces = @(
    $Workspaces | Where-Object {
        [string]::Equals(
            [string] $_.customerId,
            $WorkspaceCustomerId,
            [StringComparison]::OrdinalIgnoreCase
        )
    }
)

if ($MatchingWorkspaces.Count -eq 0) {
    throw "Could not find the Log Analytics workspace used by '$EnvironmentName'."
}
if ($MatchingWorkspaces.Count -gt 1) {
    throw "More than one Log Analytics workspace matched '$EnvironmentName'."
}

$Workspace = $MatchingWorkspaces[0]
$WorkspaceId = ([string] $Workspace.id).Trim()
$WorkspaceLocation = ([string] $Workspace.location).Trim()
if (
    [string]::IsNullOrWhiteSpace($WorkspaceId) -or
    [string]::IsNullOrWhiteSpace($WorkspaceLocation)
) {
    throw "The matching Log Analytics workspace is missing its resource ID or location."
}

$AlertEmail = ([string](Read-Host "Alert email address")).Trim()
if ([string]::IsNullOrWhiteSpace($AlertEmail)) {
    throw "An alert email address is required."
}
try {
    $ParsedEmail = New-Object System.Net.Mail.MailAddress($AlertEmail)
    if ($ParsedEmail.Address -ne $AlertEmail) {
        throw "Enter a single email address without a display name."
    }
}
catch {
    throw "Enter a valid alert email address."
}

try {
    $ActionGroupId = Invoke-AzureCliText -Arguments @(
        "monitor", "action-group", "create",
        "--name", $ActionGroupName,
        "--resource-group", $ResourceGroup,
        "--short-name", $ActionGroupShortName,
        "--action", "email", "MonitoringEmail", $AlertEmail, "usecommonalertschema",
        "--query", "id",
        "--output", "tsv"
    )
    if ([string]::IsNullOrWhiteSpace($ActionGroupId)) {
        throw "Azure returned an empty action group resource ID."
    }

    $Http5xxQuery = @"
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "$ContainerAppName"
| extend LogEntry = parse_json(Log_s)
| where tostring(LogEntry.event) == "http_request"
| extend StatusCode = toint(LogEntry.status_code)
| where StatusCode >= 500 and StatusCode < 600
"@

    $LatencyQuery = @"
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "$ContainerAppName"
| extend LogEntry = parse_json(Log_s)
| where tostring(LogEntry.event) == "http_request"
| extend DurationMs = todouble(LogEntry.duration_ms)
| where isnotnull(DurationMs)
| summarize P95LatencyMs = percentile(DurationMs, 95)
| where P95LatencyMs > 8000
"@

    New-ScheduledQueryAlert `
        -Name $Http5xxAlertName `
        -DisplayName "Fincrime API: more than 5 HTTP 5xx responses in 15 minutes" `
        -Description "Alerts when fincrime-api emits more than five HTTP 5xx responses in a 15-minute window." `
        -Query $Http5xxQuery `
        -Threshold 5 `
        -WorkspaceId $WorkspaceId `
        -WorkspaceLocation $WorkspaceLocation `
        -ActionGroupId $ActionGroupId

    New-ScheduledQueryAlert `
        -Name $LatencyAlertName `
        -DisplayName "Fincrime API: p95 latency above 8 seconds for 15 minutes" `
        -Description "Alerts when fincrime-api p95 request latency exceeds 8,000 ms in a 15-minute window." `
        -Query $LatencyQuery `
        -Threshold 0 `
        -WorkspaceId $WorkspaceId `
        -WorkspaceLocation $WorkspaceLocation `
        -ActionGroupId $ActionGroupId

    [void](Invoke-AzureCli -Arguments @(
        "deployment", "group", "create",
        "--name", $WorkbookDeploymentName,
        "--resource-group", $ResourceGroup,
        "--template-file", $WorkbookPath,
        "--parameters", "workspaceResourceId=$WorkspaceId",
        "--output", "none"
    ))

    Write-Host "Monitoring setup complete."
    Write-Host "Log Analytics workspace: $($Workspace.name)"
    Write-Host "Alerts: $Http5xxAlertName, $LatencyAlertName"
    Write-Host "Workbook deployment: $WorkbookDeploymentName"
}
finally {
    $AlertEmail = $null
}
