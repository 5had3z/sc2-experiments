# sc2-experiments
Repo contains ML experiments on SC2-serializer datasets

### Git hooks
The CI will run several checks on the new code pushed to the repository. These checks can also be run locally without waiting for the CI by following the steps below:

1. [install pre-commit](https://pre-commit.com/#install),
2. Install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit. The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`.

> Note: when you commit, you may have to explicitly run `pre-commit run --all-files` twice to make it pass, as each formatting tool will first format the code and fail the first time but should pass the second time.
