each_lockfile() {
  "$@"
  while IFS= read -r script; do
    "$@" --script "$script"
  done < <(grep -lr '^# /// script$' .mise/tasks)
}
