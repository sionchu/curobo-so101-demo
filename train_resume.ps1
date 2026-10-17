# Train ACT (from scratch) then fine-tune SmolVLA on lerobot/svla_so101_pickplace, episodes 0-44 (45-49 held out).
Set-Location D:\Workspace\curobo-demo
$env:HF_HUB_OFFLINE = '1'
$py = '.venv-lerobot\Scripts\python.exe'
$eps = '[' + ((0..44) -join ',') + ']'
$common = @('-m','lerobot.scripts.lerobot_train','--dataset.repo_id=lerobot/svla_so101_pickplace',
  '--dataset.root=D:\Workspace\curobo-demo\data\svla_so101_pickplace', "--dataset.episodes=$eps",
  '--dataset.video_backend=pyav','--policy.device=cuda','--policy.push_to_hub=false','--num_workers=2',
  '--wandb.enable=false','--log_freq=100')
$resume = @('-m','lerobot.scripts.lerobot_train','--config_path=D:\Workspace\curobo-demo\train\act\checkpoints\last\pretrained_model\train_config.json','--resume=true','--num_workers=2')
function Run($name, $extra) {
  $vram = "logs\vram_$name.txt"; Remove-Item $vram -ErrorAction SilentlyContinue
  $mon = Start-Job -ArgumentList $vram { param($f) while($true){ "$(Get-Date -Format o),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" | Out-File -Append "D:\Workspace\curobo-demo\$f"; Start-Sleep 5 } }
  $t0 = Get-Date
  if ($name -eq 'act_resume') { & $py @resume *> "logs\train_$name.log" } else { & $py @common @extra "--output_dir=D:\Workspace\curobo-demo\train\$name" "--job_name=$name" *> "logs\train_$name.log" }
  $code = $LASTEXITCODE
  Stop-Job $mon; Remove-Job $mon
  "$name exit=$code wall_s=$([int]((Get-Date)-$t0).TotalSeconds)" | Out-File -Append logs\train_summary.txt
}
Run 'act_resume' @()
Run 'smolvla' @('--policy.path=D:\Workspace\curobo-demo\models\smolvla_base','--policy.empty_cameras=1',
  '--rename_map={"observation.images.up": "observation.images.camera1", "observation.images.side": "observation.images.camera2"}',
  '--batch_size=8','--steps=20000','--save_freq=5000')

