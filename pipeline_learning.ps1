# Detached: wait until the GPU is free (other jobs share it), then the runs behind videos\learning_story.mp4.
# Progress: logs\pipeline_learning.txt
Set-Location D:\Workspace\curobo-demo
$env:HF_HUB_OFFLINE = '1'
$cu = '.venv\Scripts\python.exe'
$lr = '.venv-lerobot\Scripts\python.exe'
$log = 'logs\pipeline_learning.txt'
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
$S5 = 'train\smolvla_simft\checkpoints\005000\pretrained_model'
$S10 = 'train\smolvla_simft\checkpoints\010000\pretrained_model'
$SR = 'train\smolvla\checkpoints\020000\pretrained_model'
$ACT = 'train\act\checkpoints\025000\pretrained_model'
$TRJ = '4,6,10,14'
Step 'offline_ckpts' { $env:EVAL_OUT = 'policy_eval_ckpts'; & $lr -u so101\eval_policies.py $ACT 'ACT' $SR 'SmolVLA real' $S5 'SmolVLA real+sim 5k' $S10 'SmolVLA real+sim 10k' *> logs\offline_ckpts.log; Remove-Item Env:\EVAL_OUT }
Step 'closedloop_sim_ft_5k' { & $lr -u so101\eval_isaac_closedloop.py --policy $S5 --tag sim_ft_5k *> logs\eval_sim_ft_5k.log }
Step 'traj_sim_ft' { & $lr -u so101\eval_isaac_closedloop.py --policy $S10 --tag trj_sim_ft --only $TRJ *> logs\eval_trj_sim_ft.log }
Step 'traj_real_only' { & $lr -u so101\eval_isaac_closedloop.py --policy $SR --tag trj_real_only --only $TRJ *> logs\eval_trj_real_only.log }
Step 'traj_expert' { & $cu -u so101\eval_isaac_closedloop.py --expert --tag trj_expert --only $TRJ *> logs\eval_trj_expert.log }
Step 'traj_base' { & $lr -u so101\eval_isaac_closedloop.py --policy models\smolvla_base --tag trj_base --only $TRJ *> logs\eval_trj_base.log }
"$(Get-Date -Format o) PIPELINE DONE" | Out-File -Append $log
