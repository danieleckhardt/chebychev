# Development Container

This repository contains the Docker configuration for the NLMaves project. 

## Add as submodule

```
git submodule add git@gitlab.kit.edu:kit/ianm/ag-numerik/projects/wellen-a4/mfem-development-container.git .devcontainer
git submodule update --init --recursive
git submodule status 
```

## Using SSH keys
If you want to forward the ssh_key to your container, the following steps are the best practice (click [here](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials) for more details). 

First, start the SSH Agent in the background by running the following in a terminal:

```
eval "$(ssh-agent -s)"

```

Then add these lines to your ~/.bash_profile or ~/.zprofile (for Zsh) so it starts on login:

```bash
if [ -z "$SSH_AUTH_SOCK" ]; then
# Check for a currently running instance of the agent
RUNNING_AGENT="`ps -ax | grep 'ssh-agent -s' | grep -v grep | wc -l | tr -d '[:space:]'`"
if [ "$RUNNING_AGENT" = "0" ]; then
        # Launch a new instance of the agent
        ssh-agent -s &> $HOME/.ssh/ssh-agent
fi
eval `cat $HOME/.ssh/ssh-agent`
fi
```
Then add your local SSH keys to the agent 

```bash
ssh-add $PATH_TO_YOUR_PRIVATE_KEY

```
where you replace `$PATH_TO_YOUR_PRIVATE_KEY` by e.g `$HOME/.ssh/gitlab_rsa`.

