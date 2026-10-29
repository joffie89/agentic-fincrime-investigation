[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ResourceGroup = "fincrime-demo-rg"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI ('az') is required. Install it and run 'az login' before this script."
}

$Confirmation = Read-Host "Type '$ResourceGroup' to permanently delete the resource group and everything in it"
if ($Confirmation -cne $ResourceGroup) {
    Write-Host "Teardown cancelled; no Azure resources were changed."
    exit 0
}

& az group delete `
    --name $ResourceGroup `
    --yes `
    --only-show-errors `
    --output none

if ($LASTEXITCODE -ne 0) {
    throw "Azure CLI could not delete resource group '$ResourceGroup'."
}

Write-Host "Deleted resource group '$ResourceGroup'."
