param([string]$Name, [int]$Port, [string[]]$PyArgs)
Set-Location D:\Workspace\curobo-demo
$out = "logs\rec_$Name.out"; $err = "logs\rec_$Name.err"
$p = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList (@('-u') + $PyArgs) -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
$deadline = (Get-Date).AddMinutes(10)
while ((Get-Date) -lt $deadline) { if ((Select-String -Path $out -Pattern 'waiting for a browser' -ErrorAction SilentlyContinue) -or $p.HasExited) { break }; Start-Sleep 3 }
$c = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList @('-u','so101\headless_client.py',"http://localhost:$Port",'1800') -RedirectStandardOutput "logs\headless_$Name.out" -RedirectStandardError "logs\headless_$Name.err" -WindowStyle Hidden -PassThru
$deadline = (Get-Date).AddMinutes(25)
while ((Get-Date) -lt $deadline) { if ($p.HasExited) { break }; Start-Sleep 5 }
Stop-Process -Id $c.Id -Force -ErrorAction SilentlyContinue
"[$Name] exited=$($p.HasExited)"
Get-Content $out | Select-String 'wrote|map:' | ForEach-Object Line
Get-Content $err | Select-String -Pattern 'Error|Traceback|line \d+' | Select-Object -Last 8 | ForEach-Object Line
