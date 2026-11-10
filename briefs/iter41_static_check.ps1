# Dependency-free source and block-boundary checks. This does not parse Python,
# execute Torch, or establish floating-point/CUDA correctness.
$ErrorActionPreference = 'Stop'
$root41check = Split-Path -Parent $PSScriptRoot
$old41check = [IO.File]::ReadAllText((Join-Path $root41check 'iter40.py')).Replace("`r`n", "`n")
$new41check = [IO.File]::ReadAllText((Join-Path $root41check 'iter41.py')).Replace("`r`n", "`n")
$pattern41check = '(?ms)^(?:@[^\n]+\n)*(?:def|class) (?<name>\w+)[\s\S]*?(?=^(?:@[^\n]+\n)*(?:def|class) |^if __name__|\z)'
$before41check = @{}
$after41check = @{}
foreach ($match41check in [regex]::Matches($old41check, $pattern41check)) {
    $before41check[$match41check.Groups['name'].Value] = $match41check.Value.Trim()
}
foreach ($match41check in [regex]::Matches($new41check, $pattern41check)) {
    $name41check = $match41check.Groups['name'].Value
    if ($after41check.ContainsKey($name41check)) { throw "Duplicate definition: $name41check" }
    $after41check[$name41check] = $match41check.Value.Trim()
}
$allowed41check = @('FusedBuckets', 'table_proposals', 'LocalMixer', 'Model', 'epoch_cpu', 'prepare_model', 'self_tests', 'run')
$changed41check = @()
foreach ($name41check in $before41check.Keys) {
    if (-not $after41check.ContainsKey($name41check)) { throw "Missing definition: $name41check" }
    if ($before41check[$name41check] -cne $after41check[$name41check]) {
        if ($name41check -notin $allowed41check) { throw "Unexpected change: $name41check" }
        $changed41check += $name41check
    }
}
$writePattern41check = '(?ms)^    def write\(self, code, y\):[\s\S]*\z'
$writeBefore41check = [regex]::Match($before41check['FusedBuckets'], $writePattern41check).Value
$writeAfter41check = [regex]::Match($after41check['FusedBuckets'], $writePattern41check).Value
if (-not $writeBefore41check -or $writeBefore41check -cne $writeAfter41check) { throw 'Fused write implementation changed' }
$cases41check = 0
$lengths41check = @(0..1100) + @(4095, 4096, 4097, 8193)
foreach ($n41check in $lengths41check) {
    foreach ($stride41check in @(1, 2, 3, 7, 16, 63, 257, 511, 512, 513, 1000, 2049, 8192)) {
        $cursor41check = 0
        for ($left41check = 0; $left41check -lt $n41check; $left41check += 512) {
            $right41check = [Math]::Min($left41check + 512, $n41check)
            $first41check = (($stride41check - ($left41check % $stride41check)) % $stride41check)
            $start41check = [int](($left41check + $first41check) / $stride41check)
            $stop41check = [int][Math]::Floor(($right41check + $stride41check - 1) / $stride41check)
            if ($start41check -ne $cursor41check) { throw 'Sample destination has gap or overlap' }
            $count41check = 0
            for ($offset41check = $first41check; $offset41check -lt $right41check - $left41check; $offset41check += $stride41check) {
                if ($left41check + $offset41check -ne ($cursor41check + $count41check) * $stride41check) { throw 'Wrong global sample' }
                $count41check++
            }
            if ($count41check -ne $stop41check - $start41check) { throw 'Sample source/destination length differs' }
            $cursor41check = $stop41check
        }
        if ($cursor41check -ne [int][Math]::Floor(($n41check + $stride41check - 1) / $stride41check)) { throw 'Wrong total sample count' }
        $cases41check++
    }
}
$parseErrors41check = $null
$tokens41check = $null
[void][Management.Automation.Language.Parser]::ParseFile((Join-Path $root41check 'run_iter41.ps1'), [ref]$tokens41check, [ref]$parseErrors41check)
if ($parseErrors41check.Count) { throw ($parseErrors41check | Out-String) }
$report41check = [ordered]@{
    status = 'passed'
    unchanged_iter40_sha256 = (Get-FileHash -LiteralPath (Join-Path $root41check 'iter40.py') -Algorithm SHA256).Hash
    inherited_definitions_checked = $before41check.Count
    changed_inherited_definitions = @($changed41check | Sort-Object)
    fused_write_source_unchanged = $true
    block_sampling_scalar_cases = $cases41check
    runner_powershell_syntax = 'passed'
    python_syntax = 'not checked: required executable blocked'
    torch_runtime = 'not checked: required executable blocked'
}
$report41check | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $root41check 'results/iter41_static_checks.json') -Encoding UTF8
$report41check | ConvertTo-Json -Depth 5
