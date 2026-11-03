# Provenance of the commit history

On 3 November 2026 the author and committer of every commit were set to Asif Rahman and tool trailers were removed from commit messages. File contents were not changed. Every rewritten commit has the same tree (the same files, byte for byte) as the commit it replaces, so plan digests, result files and their recorded times are unaffected. Commit times were kept.

Rewriting gives every commit a new ID. GitHub's push record (`github_activity.json`, exported from the GitHub API before the rewrite) names the original IDs and the time each push arrived. `commit_map.tsv` pairs each original ID with its new ID. The original history is held by the author as a git bundle and is available on request. With it, anyone can check that each new commit has the same tree as the original and that the original was pushed to GitHub at the recorded time.
