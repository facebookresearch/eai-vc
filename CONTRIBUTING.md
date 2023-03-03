# Contributing to eai-vc

## Pull Requests

1. Clone the repo, and create your branch from `main`. Title it `<unixname>/<branch-name>`.
    - If this is the first time, clone the repo: `git clone --recursive git@github.com:facebookresearch/eai-vc.git`
    - If you need to update the repo with new changes, pull main: `git checkout main && git pull origin main && git submodule update --init --recursive`
    - Create your branch: `git checkout -b <unixname>/<branch-name>`
1. Push your branch to the repo, and create a pull request.
    - We use [black](https://github.com/psf/black) to autoformat our code. You can either
        2. Push your code; if there are formatting violations, our bot will automatically fix them. You can then pull the branch.
        1. Run `black .` before pushing your code. You can automatically [set this to run on save in VSCode](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0).
    - Push your branch: `git push --set-upstream origin <unixname>/<branch-name>`
    - Create a pull request: https://github.com/facebookresearch/eai-vc/pull/new/unixname/branch-name
    - Fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
1. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to eai-vc, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
