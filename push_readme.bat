@echo off
python d:\orchestra\log_readme.py
del d:\orchestra\log_readme.py
if not exist "d:\orchestra\assets" mkdir d:\orchestra\assets
copy "C:\Users\yugda\.gemini\antigravity\brain\0d7f6835-002f-4a4c-988e-ef9611d8c0f5\hackerrank_orchestrate_banner_1777632569573.png" "d:\orchestra\assets\banner.png"
git add -A
git commit -m "docs: improvise README with banner, badges, and mermaid diagrams"
git push origin main
echo SUCCESS_ALL
