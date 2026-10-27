# Problem overcome: leaked SSH private key (notes for write-up)

**Date:** 27 Oct 2026

## What happened
While debugging the telemetry dashboard, a review of the project folder found the EC2 SSH
private key (`samuelojo-key.pem`) sitting in the repo folder. `.gitignore` had no rule for
`*.pem`, and `git log --all -- samuelojo-key.pem` showed it had been committed in the very
first commit (aaea4c8, "Initial clean commit: Dashboard, CI/CD pipeline, and offline fallback",
5 Oct 2026). Anyone with a copy of the repo could have logged into the EC2 server.

## How it was fixed
1. Generated a new ed25519 key pair outside the project folder (`C:\Users\samue\.ssh\ac-ec2`).
2. Logged in with the old key and appended the new public key to `~/.ssh/authorized_keys`
   on the EC2 instance (appended, not overwritten, to avoid locking myself out).
3. Verified the new key worked from a second terminal before changing anything else.
4. Backed up `authorized_keys`, removed the old key's line, and confirmed the old key was refused.

## Lessons
- Keep private keys outside the repo; add `*.pem` to `.gitignore` before the first commit.
- Rotate first, clean up history second: rewriting git history does not help if the key was
  already cloned or pushed.
- Always add the new key and test it in a second session before removing the old one.

## Still to do (update before using this in a write-up)
- [ ] Add `*.pem` to `.gitignore` and `git rm --cached samuelojo-key.pem`
- [ ] Purge the key from git history (`git filter-repo --invert-paths --path samuelojo-key.pem`)
- [ ] Check whether the repo has a remote / is public (`git remote -v`)
- [ ] Update any CI/CD secret that used the old key
- [ ] Restrict the EC2 security group's SSH (port 22) rule to my own IP
- [ ] Delete the old .pem from the project folder
