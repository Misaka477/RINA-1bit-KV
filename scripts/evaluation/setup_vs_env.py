"""Set VS environment variables in PowerShell profile for triton."""
import os

# Find or create profile directory
profile_dirs = [
    os.path.expanduser("~\\Documents\\PowerShell"),
    os.path.expanduser("~\\Documents\\WindowsPowerShell"),
]
for d in profile_dirs:
    os.makedirs(d, exist_ok=True)

p = os.path.join(profile_dirs[1], "Microsoft.PowerShell_profile.ps1")

lines = [
    '$env:CC = "D:\\Software_Development\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64\\cl.exe"',
    '$env:INCLUDE = "D:\\Software_Development\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\include;D:\\Windows Kits\\10\\Include\\10.0.26100.0\\ucrt;D:\\Windows Kits\\10\\Include\\10.0.26100.0\\shared;D:\\Windows Kits\\10\\Include\\10.0.26100.0\\um"',
    '$env:LIB = "D:\\Software_Development\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\lib\\x64;D:\\Windows Kits\\10\\Lib\\10.0.26100.0\\ucrt\\x64;D:\\Windows Kits\\10\\Lib\\10.0.26100.0\\um\\x64"',
    "",
]

with open(p, "w") as f:
    f.write("\n".join(lines))
print(f"Written to {p}")
print("Open a new PowerShell terminal for changes to take effect.")
