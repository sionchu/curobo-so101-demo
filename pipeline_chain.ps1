# Detached: wait until the GPU is free (other jobs share it), then the full-chain evals. Progress: logs\pipeline_chain.txt
Set-Location D:\Workspace\curobo-demo
$env:HF_HUB_OFFLINE = '1'
$cu = '.venv\Scripts\python.exe'
$log = 'logs\pipeline_chain.txt'
$free = 0
while ($free -lt 3) {  # 3 consecutive minutes with < 4.5 GB used by others
  $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if ($used -lt 4500) { $free++ } else { $free = 0 }
  "$(Get-Date -Format o) gpu_used_mb=$used free_checks=$free" | Out-File -Append $log
  if ($free -lt 3) { Start-Sleep 60 }
}
function Step($name, [scriptblock]$body) {
  "$(Get-Date -Format o) START $name" | Out-File -Append $log
  $t0 = Get-Date
  & $body
  "$(Get-Date -Format o) END $name exit=$LASTEXITCODE wall_s=$([int]((Get-Date)-$t0).TotalSeconds)" | Out-File -Append $log
}
Step 'chain_policy' { & $cu -u so101\chain_isaac.py --grasp policy --tag chain_policy *> logs\chain_policy.log }
Step 'chain_curobo' { & $cu -u so101\chain_isaac.py --grasp curobo --tag chain_curobo *> logs\chain_curobo.log }
"$(Get-Date -Format o) PIPELINE DONE" | Out-File -Append $log
