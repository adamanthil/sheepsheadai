#!/usr/bin/env bash
# Content-derived image tags: the same files give the same tag at any commit.
#
# The tag is a hash of `git ls-tree` over a service's paths, so it needs no
# history, never changes length, and a revert reproduces the old tag (and
# reuses the old image). CI and hand deploys both run this script, so they
# cannot disagree. Output is KEY=VALUE lines, ready for $GITHUB_OUTPUT.
# See notebooks/CD_Deployment_Plan_202608.md section 3.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

tag() { git ls-tree -r HEAD -- "$@" | git hash-object --stdin | cut -c1-12; }

api_paths=()
while IFS= read -r line; do
  [[ -z $line || $line == \#* ]] || api_paths+=("$line")
done < app/deploy/api-paths.txt

echo "api=$(tag "${api_paths[@]}")"
echo "web=$(tag app/web)"
echo "sha=$(git rev-parse HEAD)"
