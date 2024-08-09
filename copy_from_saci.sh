


rsync -av --exclude-from='.ignoresaci' f.dias@saci:$1 dados2 --verbose
sleep 1
file=$(find dados2 -maxdepth 2 -type f -name '*config*.py')

# Check if exactly one file matches
if [ $(echo "$file" | wc -l) -eq 1 ]; then
  # Rename the file
  mv "$file" "config_dx.py"
  echo "Renamed $file to config_dx.py"
else
  echo "Error: Either no files or multiple files match the pattern."
fi