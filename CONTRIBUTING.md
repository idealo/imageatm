# Contribution Guide

We welcome any contributions whether it's,

- Submitting feedback
- Fixing bugs
- Or implementing a new feature.

Please read this guide before making any contributions.


#### Submit Feedback
The feedback should be submitted by creating an issue at [GitHub issues](https://github.com/idealo/imageatm/issues).
Select the related template (bug report, feature request, or custom) and add the corresponding labels.

#### Fix Bugs:
You may look through the [GitHub issues](https://github.com/idealo/imageatm/issues) for bugs.

#### Implement Features
You may look through the [GitHub issues](https://github.com/idealo/imageatm/issues) for feature requests.

## Rules of Engagement
- Code flow should always go into one direction `dev` -> master
- Every PR needs to reference an issue
- Issues need to be referenced to the Image ATM project when creating it
- Feature branches should be cloned from latest `dev` version
- Issues should be referenced to the commits
- Assign independent review for merging PR feature branches -> `dev`
- dev -> master is based on new version bump or decided adhoc on our regular meeting
- Feature branches should be deleted after the merge
- Every contributor should be able to move `dev` to `master` and deploy it to PyPI


## Pull Requests (PR)
1. Fork the repository and a create a new branch from the master branch.
2. For bug fixes, add new tests and for new features please add changes to the documentation.
3. Do a PR from your new branch to our `dev` branch of the original Image ATM repo.

## Documentation
- Make sure any new function or class you introduce has proper docstrings. We use the [Google Python Styling Guide](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for our docstrings.

## Testing
- We use [pytest](https://docs.pytest.org/en/latest/) for our testing. Make sure to write tests for any new feature and/or bug fixes.

## Main Contributor List
We maintain a list of main contributors to appreciate all the contributions.
