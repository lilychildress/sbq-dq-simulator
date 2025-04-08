#!/bin/bash

oldNameUnderscore='bff_simulator'
oldNameHyphen=$(echo "$oldNameUnderscore" | tr _ -)
echo "what should $oldNameUnderscore be renamed to?"
read newNameUnderscore
newNameHyphen=$(echo "$newNameUnderscore" | tr _ -)

echo "Renaming folder $oldNameUnderscore to $newNameUnderscore..."
mv ./$oldNameUnderscore ./"$newNameUnderscore"

echo "Changing $oldNameUnderscore to $newNameUnderscore..."
find . -path "./.history" -prune -o -path "./.git" -prune -o -path "./poetry.lock" -prune -o -path "./.venv" -prune -o -type f -exec sed -i 's/'$oldNameUnderscore'/'$newNameUnderscore'/g' {} +

echo "Changing $oldNameHyphen to $newNameHyphen..."
find . -path "./.history" -prune -o -path "./.git" -prune -o -path "./poetry.lock" -prune -o -path "./.venv" -prune -o -type f -exec sed -i 's/'$oldNameHyphen'/'$newNameHyphen'/g' {} +

echo "Rename complete"
