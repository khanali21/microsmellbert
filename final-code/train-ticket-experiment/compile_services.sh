#!/bin/bash
REPO_PATH="train-ticket"
for service_dir in "$REPO_PATH"/ts-*; do
    if [ -d "$service_dir" ]; then
        echo "Compiling $service_dir..."
        cd "$service_dir" || continue
        mvn clean compile -DskipTests  # Skip tests to speed up
        cd - || exit
    fi
done
echo "Compilation complete."