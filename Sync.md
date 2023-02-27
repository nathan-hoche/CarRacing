# How to synchronize both GitHub and Kent GitLab repositories

> ***Don't forget to set your ssh key in the Gitlab settings !***

## Create a new remote called "all" with the URL of the primary repo.
```bash
git remote add all git@github.com:nathan-hoche/CarRacing.git
```
> _This will add a new remote called "all" with the URL of the primary repo._
## Re-register the remote as a push URL.
```bash
git remote set-url --add --push all git@github.com:nathan-hoche/CarRacing.git
```
> _When pushing to the "all" remote, you will push to both the origin and the secondary repo._
## Add the secondary repo as a push URL.
```bash
git remote set-url --add --push all git@git.cs.kent.ac.uk:gh384/aisystemimplementation-project.git
```

## Push to both remotes.
```bash
git push all main
```
> _This will push to both the origin and the secondary repo._

## Pull from both remotes.
> You cannot pull from two remotes so you only pull from the primary repo.
```bash
git pull origin main
```