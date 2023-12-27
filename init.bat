@echo off
cls

echo $ rmdir /q /s .git
rmdir /q /s .git

echo.

echo $ git init
git init

echo.

echo $ git branch -M main
git branch -M main

echo.

echo $ git add --all
git add --all

echo.

echo $ git commit -m "초기화"
echo ----------------------------------------
git commit -m "초기화"

echo.

echo $ git remote add origin git@github.com:hyk2202/megait.git
echo ----------------------------------------
git remote add origin git@github.com:hyk2202/megait.git

echo.

echo $ git push --force --set-upstream origin main
echo ----------------------------------------
git push --force --set-upstream origin main

echo.

pause