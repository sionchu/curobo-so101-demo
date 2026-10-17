# Detached pipeline: sim data -> LeRobot dataset -> SmolVLA co-training -> Isaac closed-loop evals -> offline re-eval.
# Runs independently of any interactive session. Progress: logs\pipeline_simft.txt
Set-Location D:\Workspace\curobo-demo
$env:HF_HUB_OFFLINE = '1'
$lr = '.venv-lerobot\Scripts\python.exe'
$cu = '.venv\Scripts\python.exe'
$log = 'logs\pipeline_simft.txt'
function Step($name, [scriptblock]$body) {
  "$(Get-Date -Format o) START $name" | Out-File -Append $log
  $t0 = Get-Date
  & $body
  "$(Get-Date -Format o) END $name exit=$LASTEXITCODE wall_s=$([int]((Get-Date)-$t0).TotalSeconds)" | Out-File -Append $log
}
Step 'convert' { & $lr so101\sim_to_lerobot.py *> logs\sim_to_lerobot.log }
$eps = '[' + (((0..44) + (50..169)) -join ',') + ']'
Step 'train_simft' {
  $vram = 'logs\vram_smolvla_simft.txt'
  $mon = Start-Job -ArgumentList $vram { param($f) while($true){ "$(Get-Date -Format o),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" | Out-File -Append "D:\Workspace\curobo-demo\$f"; Start-Sleep 5 } }
  & $lr -m lerobot.scripts.lerobot_train --dataset.repo_id=local/so101_real_plus_sim `
    --dataset.root=D:\Workspace\curobo-demo\data\so101_real_plus_sim "--dataset.episodes=$eps" --dataset.video_backend=pyav `
    --policy.path=D:\Workspace\curobo-demo\train\smolvla\checkpoints\020000\pretrained_model --policy.device=cuda `
    --policy.push_to_hub=false --policy.empty_cameras=1 `
    '--rename_map={"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}' `
    --batch_size=8 --steps=10000 --save_freq=5000 --log_freq=100 --num_workers=2 --wandb.enable=false `
    --output_dir=D:\Workspace\curobo-demo\train\smolvla_simft --job_name=smolvla_simft *> logs\train_smolvla_simft.log
  Stop-Job $mon; Remove-Job $mon
}
Step 'eval_real_only' { & $lr -u so101\eval_isaac_closedloop.py --policy train\smolvla\checkpoints\020000\pretrained_model --tag real_only *> logs\eval_real_only.log }
Step 'eval_sim_ft' { & $lr -u so101\eval_isaac_closedloop.py --policy train\smolvla_simft\checkpoints\010000\pretrained_model --tag sim_ft *> logs\eval_sim_ft.log }
Step 'eval_expert' { & $cu -u so101\eval_isaac_closedloop.py --expert --tag curobo_expert *> logs\eval_expert.log }
Step 'offline_eval' { & $lr so101\eval_policies.py train\act\checkpoints\025000\pretrained_model ACT train\smolvla\checkpoints\020000\pretrained_model SmolVLA train\smolvla_simft\checkpoints\010000\pretrained_model SmolVLA_simft *> logs\offline_eval_simft.log }
"$(Get-Date -Format o) PIPELINE DONE" | Out-File -Append $log
