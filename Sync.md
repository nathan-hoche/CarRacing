# To synchronize both GitHub and Kent GitLab repositories

## Create a new remote called "all" with the URL of the primary repo.
git remote add all git@github.com:nathan-hoche/CarRacing.git
## Re-register the remote as a push URL.
git remote set-url --add --push all git@github.com:nathan-hoche/CarRacing.git
# Add the secondary repo as a push URL.
git remote set-url --add --push all git@git.cs.kent.ac.uk:gh384/aisystemimplementation-project.git

## Push to both remotes.
git push all main

## Pull from both remotes.
### You cannot pull from two remotes so you only pull from the primary repo.
git pull