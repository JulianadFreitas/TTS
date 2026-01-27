## 📊 Feature Definitions for Pull Request Analysis


In this directory you will find the raw data files for each repository directly as they were gotten from mining.

You will also find the preprocessed data files categorised by group e.g group one is ***`PR_Lifetime_group1.csv`*** 


Below is the table of all the columns and their definitions

| Original Column Name | Reference | Definition / Meaning | Category |
|:---|:---|:---|:---|
| `number of commits` | kikas2016using, rees2017better | Total number of commits in the pull request branch. | **CODE** |
| `lines of code changed` | kersten2006using | Total lines of code added or deleted. | **CODE** |
| `number of files changed` | kersten2006using | Total number of distinct files modified. | **CODE** |
| `code changes ratio` | kersten2006using | Ratio comparing lines added/deleted/changed. | **CODE** |
| `number of revisions` | kikas2016using | Times PR was updated with new commits after review started. | **CODE** |
| `number of milestones` | | A binary count if the Pull Request was associated with a specific developmental milestone. | **CODE** |
| `test coverage` | | Percentage of the PR's new or changed code that is covered by automated unit tests. | **CODE** |
|---|---|---|---|
| `number of comments` | kikas2016using | Total number of comments on the entire thread. | **ACTIVITY** |
| `number of review comments` | zhang2022pull | Total number of inline comments on the code diff during review. | **ACTIVITY** |
| `number of reviewers` | kikas2016using | Number of unique people who submitted a formal review. | **ACTIVITY** |
| `number of approvals` | moreira2021factors | Total number of formal 'Approval' reviews submitted. | **ACTIVITY** |
| `time to first response` | hasan2023understanding | Time elapsed between the PR being opened and the first comment/review. | **ACTIVITY** |
| `number of changes requested` | moreira2021factors | Total number of formal 'Request Changes' reviews submitted. | **ACTIVITY** |
| `number of reviews requested` | zhang2022pull | Number of unique people formally requested to review (pending review submission). | **ACTIVITY** |
| `PR comments text` | zhang2022pull | Text content of all discussion comments (for NLP analysis). | **ACTIVITY** |
| `review comments` | zhang2022pull | Text content of all inline code review comments (for NLP analysis). | **ACTIVITY** |
| `comments` | rees2017better | Binary to determine the presence of comments. | **ACTIVITY** |
| `comment author number` | kikas2016using | A count of the number of unique comment authors. | **ACTIVITY** |
| `PR comments text length` | zhang2022pull | Length of all comments combined. | **ACTIVITY** |
| `PR comments text wordiness` | kikas2016using | Measure of how verbose the discussion comments are. | **ACTIVITY** |
| `time since last commit` | | Time since most recent commit pushed to the PR branch. | **ACTIVITY** |
| `review duration` | | The total time from when the PR was initially created until it was last updated. | **ACTIVITY** |
|---|---|---|---|
| `repo` | kikas2016using | Identifier for the repository (Organization context). | **CONTEXT** |
| `labels` | zhang2022pull | List of all labels applied to the PR. | **CONTEXT** |
| `number of assignees` | cosentino2017systematic | Total number of unique people officially assigned to the PR. | **CONTEXT** |
| `number of build runs` | zhang2022pull | Number of Continuous Integration (CI) build executions. | **CONTEXT** |
| `number of build failures` | zhang2022pull | Number of CI build failures. | **CONTEXT** |
| `PR text` | kikas2016using | Main description text of the Pull Request (for NLP). | **CONTEXT** |
| `PR text wordiness` | kikas2016using | Measure of how verbose the main PR description is. | **CONTEXT** |
| `label count` | kikas2016using | Total count of labels applied to the PR. | **CONTEXT** |
| `day of week` | anbalagan2009days | The day of the week the PR was opened. | **CONTEXT** |
| `weekday` | sliwerski2005changes | Binary indicator: Was the PR opened on a weekday. | **CONTEXT** |
| `Number of linked Issues` | | The count of related software Issues explicitly referenced and linked to the PR. | **CONTEXT** |
|---|---|---|---|
| `PRs Closed in Last 2 Weeks` | kikas2016using, rees2017better | Number of PRs closed in the repo recently. | **ORG/AUTHOR** |
| `PRs Opened in Last 2 Weeks` | kikas2016using, rees2017better | Number of PRs opened in the repo recently. | **ORG/AUTHOR** |
| `Author PRs Opened` | kikas2016using, rees2017better | Total PRs previously opened by this author (experience). | **ORG/AUTHOR** |
| `Open PRs at Open Date` | | Total PRs open at the time the PR was created. | **ORG/AUTHOR** |
|---|---|---|---|
| `PR Lifetime` | | Time until the PR was closed/merged (target variable). | **TARGET** |
