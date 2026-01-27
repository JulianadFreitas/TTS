# A list of observed limitations:

- Scraping repositories by popularity (i.e. number of stars) and language is quite
  complicated. GitHub API sometimes can not order millions of repositories written
  in a given language and return exactly what is the order of these repositories
  in seconds. For this reason, a same URL may return more than once in different
  requests.

  This problem is discussed by GitHub API developers at the following page:
  developer.github.com/v3/search/#timeouts-and-incomplete-results